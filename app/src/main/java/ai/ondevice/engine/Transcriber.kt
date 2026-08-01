package ai.ondevice.engine

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.media.MediaRecorder
import ai.ondevice.core.BackendId
import ai.ondevice.core.SparseParams
import ai.ondevice.core.TranscriptSegment
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.flow.onCompletion
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.concurrent.withLock
import kotlin.math.abs
import kotlin.math.min

/**
 * Speech to text, for real (SPEC §6).
 *
 * Recording and transcribing are separate things here, and they used not to be.
 * [listen] captures: microphone in, WAV out, input levels for the waveform, and
 * no decoding at all. [transcribeFile] decodes any container Android can demux,
 * resampled to the 16 kHz mono float whisper requires. The screen records, then
 * transcribes the take.
 *
 * The capture loop used to re-decode a ten-second trailing window every three
 * seconds, whether or not anyone had spoken, to show partial text. It cost about
 * half of eight cores for the whole recording — measured in a silent room — and
 * every word it showed was provisional, because whisper has no incremental
 * decode: each pass was a fresh transcription of overlapping audio, so earlier
 * words kept changing under the reader. Transcribing once, at the end, is
 * cheaper and produces the better transcript: real segment boundaries, which a
 * sliding window can never give.
 *
 * The microphone is opened inside the flow and released in `onCompletion`, so
 * cancelling the recording actually frees the device rather than leaving it hot.
 */
class Transcriber(
    private val context: Context,
    private val computeDevice: ComputeDevice,
) {
    private val json = Json { ignoreUnknownKeys = true; isLenient = true }

    /**
     * Held across every native call, so nothing frees the context while a
     * decode is inside it.
     *
     * `handle` being `@Volatile` made each read current without making the pair
     * of them atomic: [decode] checked it, and `unload` — on the load path, on
     * Stop, on a model switch — could free the context in the window before the
     * native call used the value already read. That is a use-after-free on a
     * pointer, which is a SIGSEGV rather than an exception: the app vanishes.
     * Reachable by switching speech model, or reloading, while a take is still
     * being transcribed.
     *
     * A free therefore waits for the decode in flight. With a large model that
     * is a real pause on the caller's thread, and it is the right trade: the
     * alternative is not a faster unload, it is a crash.
     */
    private val nativeLock = java.util.concurrent.locks.ReentrantLock()

    @Volatile
    private var handle: Long = 0L

    @Volatile
    var loadedModelId: String? = null
        private set

    /**
     * The device the loaded model is actually on, so a changed setting is a
     * reason to reload rather than a badge nobody acts on.
     *
     * Settings → Compute device was read at load and never again, and every
     * caller skipped the load when the model id had not changed — so switching
     * from GPU to NPU did nothing at all until the process died. The model id
     * alone was never the whole question.
     */
    @Volatile
    var loadedBackend: BackendId? = null
        private set

    val available: Boolean get() = WhisperBridge.available
    val isLoaded: Boolean get() = handle != 0L

    /**
     * Whether the loaded model is the one asked for, *on the device asked for*.
     *
     * Callers use this rather than comparing ids, which is the check that let a
     * changed Compute device go unnoticed.
     */
    suspend fun isCurrent(modelId: String): Boolean =
        isLoaded && loadedModelId == modelId &&
            loadedBackend == computeDevice.chosen(RuntimeRegistry.WHISPER)

    suspend fun load(
        modelId: String,
        path: String,
        params: SparseParams = SparseParams.EMPTY,
    ): Result<String> =
        withContext(Dispatchers.IO) {
            runCatching {
                check(WhisperBridge.available) {
                    WhisperBridge.loadError ?: "The whisper.cpp runtime is not installed in this build."
                }
                // Settings → Compute device, asked of whisper's own binary.
                // whisper offered exactly one way to say this and it was a
                // hardcoded `true`, which meant "whichever accelerator ggml
                // registered first" — fine while that was only ever OpenCL, a
                // silent choice now that the NPU registers alongside it.
                val device = computeDevice.chosen(RuntimeRegistry.WHISPER)
                val backend = device.registryNames.first()

                nativeLock.withLock {
                    unload()
                    android.util.Log.i(TAG, "loading ${File(path).name} on $backend")
                    val newHandle = WhisperBridge.nativeLoad(path, backend)
                    check(newHandle != 0L) { "The runtime returned no handle for $path." }
                    handle = newHandle
                    loadedModelId = modelId
                    loadedBackend = device
                    if (!params.isEmpty) {
                        WhisperBridge.nativeApplyParams(handle, params.toJsonString())
                    }
                    android.util.Log.i(TAG, "loaded ${File(path).name} backend=$backend")
                    WhisperBridge.nativeInfo(handle)
                }
                // A half-loaded context is worse than none: it reports a model
                // id the callers will trust and skip the reload for.
            }.onFailure { unload() }
        }

    fun unload() = nativeLock.withLock {
        if (handle != 0L) {
            WhisperBridge.nativeFree(handle)
            handle = 0L
        }
        loadedModelId = null
        loadedBackend = null
    }

    fun applyParams(params: SparseParams): ParamReport = nativeLock.withLock {
        if (handle == 0L) return ParamReport(rejected = params.keys.toList())
        val report = json.parseToJsonElement(
            WhisperBridge.nativeApplyParams(handle, params.toJsonString()),
        ).jsonObject
        return ParamReport(
            applied = report["applied"]?.jsonArray?.map { it.jsonPrimitive.content } ?: emptyList(),
            rejected = report["rejected"]?.jsonArray?.map { it.jsonPrimitive.content } ?: emptyList(),
        )
    }

    // — capture —

    /**
     * Record until cancelled, writing the take to [captureTo].
     *
     * Capture only: no model is needed and none is touched. What comes back is
     * input level, for the waveform and the clock. The transcription happens
     * once, afterwards, over the finished file.
     */
    @SuppressLint("MissingPermission")
    fun listen(captureTo: java.io.File? = null): Flow<CaptureEvent> = flow {
        if (!hasMicPermission()) {
            emit(CaptureEvent.Failed("Recording needs the microphone permission."))
            return@flow
        }

        val minBuffer = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL, ENCODING)
        if (minBuffer <= 0) {
            emit(CaptureEvent.Failed("This device has no 16 kHz mono input."))
            return@flow
        }
        val record = AudioRecord(
            MediaRecorder.AudioSource.VOICE_RECOGNITION,
            SAMPLE_RATE,
            CHANNEL,
            ENCODING,
            maxOf(minBuffer, SAMPLE_RATE * 2),
        )
        if (record.state != AudioRecord.STATE_INITIALIZED) {
            record.release()
            emit(CaptureEvent.Failed("The microphone could not be opened — another app may hold it."))
            return@flow
        }

        activeRecord = record
        record.startRecording()

        val buffer = ShortArray(minBuffer / 2)
        var totalSamples = 0L
        val writer = captureTo?.let {
            runCatching { ai.ondevice.speech.WavWriter(it, SAMPLE_RATE) }.getOrNull()
        }
        captureWriter = writer

        try {
            while (true) {
                currentCoroutineContext().ensureActive()
                val read = record.read(buffer, 0, buffer.size)
                if (read <= 0) continue

                // Paused keeps the microphone open and the file honest: the
                // samples read while paused are thrown away rather than
                // written, so a pause is a gap in the take rather than a
                // silence the length of the pause.
                if (paused) continue
                writer?.append(buffer, read)

                var peak = 0f
                for (i in 0 until read) {
                    val sample = abs(buffer[i] / 32768f)
                    if (sample > peak) peak = sample
                }
                totalSamples += read

                emit(
                    CaptureEvent.Level(
                        peak = peak,
                        elapsedMillis = totalSamples * 1000 / SAMPLE_RATE,
                    ),
                )
            }
        } finally {
            // The microphone is a shared, exclusive device. Releasing it here —
            // rather than trusting a caller to remember — is the difference
            // between cancelling a recording and hanging every other app that
            // wants to record.
            runCatching { record.stop() }
            runCatching { record.release() }
            activeRecord = null
            // A backstop, not the usual path: [finishCapture] closes the take
            // the moment Stop is pressed, and both are idempotent. This one
            // covers the capture ending or failing on its own.
            finishCapture()
            paused = false
        }
    }.flowOn(Dispatchers.IO).onCompletion {
        activeRecord?.let { runCatching { it.stop() }; runCatching { it.release() } }
        activeRecord = null
    }

    /**
     * End the take now, without waiting for the loop to unwind.
     *
     * Cancelling the capture does not stop it promptly: the collector may be
     * inside a native decode of the last window, and with a large model that is
     * tens of seconds during which the header still claims zero samples and the
     * microphone is still held. So Stop closes the writer and releases the
     * device itself, and the loop's `finally` finds both already done — both
     * calls are idempotent. Without this, a three-second recording handed to a
     * player read 00:00, because the length is written at close.
     */
    fun finishCapture() {
        val writer = captureWriter
        captureWriter = null
        runCatching { writer?.close() }
        activeRecord?.let { runCatching { it.stop() } }
    }

    /**
     * Paused keeps the microphone open rather than releasing it.
     *
     * Stopping and restarting AudioRecord loses the device for as long as the
     * handover takes, and on some phones another app takes it in the gap. A
     * flag costs one branch per buffer.
     */
    /** The take being written, so Stop can close it without waiting. */
    @Volatile
    private var captureWriter: ai.ondevice.speech.WavWriter? = null

    @Volatile
    var paused: Boolean = false
        private set

    fun pause() { paused = true }

    fun resume() { paused = false }

    @Volatile
    private var activeRecord: AudioRecord? = null

    // — file —

    /** Decode any container Android can demux, then transcribe the whole thing. */
    suspend fun transcribeFile(file: File): Result<List<TranscriptSegment>> =
        withContext(Dispatchers.Default) {
            runCatching {
                check(handle != 0L) { "No speech model is loaded." }
                val samples = decodeToPcm(file)
                check(samples.isNotEmpty()) { "No audio could be decoded from ${file.name}." }
                decode(samples)
            }
        }

    /**
     * The same decode, at whatever rate the caller needs.
     *
     * Exposed for OmniVoice's voice cloning, which wants the reference clip at
     * 24 kHz rather than whisper's 16. Going through 16 kHz and back up would
     * throw away everything above 8 kHz and then invent it again, and that band
     * is a large part of what makes one voice sound different from another —
     * which is the entire point of the clip.
     */
    suspend fun decodeAudio(file: File, targetRate: Int): Result<FloatArray> =
        withContext(Dispatchers.Default) {
            runCatching {
                val samples = decodeToPcm(file, targetRate)
                check(samples.isNotEmpty()) { "No audio could be decoded from ${file.name}." }
                samples
            }
        }

    /**
     * Transcribe audio the caller already has in memory, at [sampleRate].
     *
     * OmniVoice's voice cloning uses this to work out what a reference clip
     * says, which it needs before it can copy the voice — and it holds that clip
     * at 24 kHz, so the rate whisper wants is converted here rather than being
     * a number the caller has to know.
     */
    suspend fun transcribeSamples(
        samples: FloatArray,
        sampleRate: Int,
    ): Result<List<TranscriptSegment>> =
        withContext(Dispatchers.Default) {
            runCatching {
                check(handle != 0L) { "No speech model is loaded." }
                check(samples.isNotEmpty()) { "There is no audio to transcribe." }
                decode(toWhisperRate(samples, sampleRate))
            }
        }

    private fun toWhisperRate(samples: FloatArray, sampleRate: Int): FloatArray {
        if (sampleRate == SAMPLE_RATE) return samples
        val ratio = sampleRate.toDouble() / SAMPLE_RATE
        return FloatArray((samples.size / ratio).toInt()) { i ->
            val position = i * ratio
            val left = position.toInt()
            val right = (left + 1).coerceAtMost(samples.size - 1)
            val fraction = (position - left).toFloat()
            samples[left] * (1f - fraction) + samples[right] * fraction
        }
    }

    private fun decode(samples: FloatArray): List<TranscriptSegment> = nativeLock.withLock {
        if (handle == 0L) return emptyList()
        val result = json.parseToJsonElement(WhisperBridge.nativeTranscribe(handle, samples)).jsonObject
        return result["segments"]?.jsonArray.orEmpty().map { element ->
            val obj = element.jsonObject
            TranscriptSegment(
                startMillis = obj["startMillis"]?.jsonPrimitive?.content?.toLongOrNull() ?: 0L,
                endMillis = obj["endMillis"]?.jsonPrimitive?.content?.toLongOrNull() ?: 0L,
                text = obj["text"]?.jsonPrimitive?.content?.trim().orEmpty(),
                confidence = obj["confidence"]?.jsonPrimitive?.content?.toFloatOrNull() ?: 1f,
            )
        }.filter { it.text.isNotBlank() }
    }

    /**
     * Container → mono float at [targetRate], defaulting to the 16 kHz whisper
     * is the only consumer of.
     *
     * The resample averages the samples falling in each output period. For
     * speech at 44.1 or 48 kHz down to 16 kHz that is audibly poor but
     * transcription-equivalent — whisper's own front end throws most of that
     * bandwidth away building the mel spectrogram — and it is a genuine
     * low-pass rather than nearest-neighbour, so it holds up for the 24 kHz a
     * voice clone asks for too.
     */
    private fun decodeToPcm(file: File, targetRate: Int = SAMPLE_RATE): FloatArray {
        val extractor = MediaExtractor()
        extractor.setDataSource(file.absolutePath)

        val trackIndex = (0 until extractor.trackCount).firstOrNull { index ->
            extractor.getTrackFormat(index).getString(MediaFormat.KEY_MIME)?.startsWith("audio/") == true
        } ?: run {
            extractor.release()
            error("${file.name} has no audio track.")
        }

        val format = extractor.getTrackFormat(trackIndex)
        extractor.selectTrack(trackIndex)
        val mime = format.getString(MediaFormat.KEY_MIME)!!
        val sourceRate = format.getInteger(MediaFormat.KEY_SAMPLE_RATE)
        val channels = format.getInteger(MediaFormat.KEY_CHANNEL_COUNT)

        val codec = MediaCodec.createDecoderByType(mime)
        codec.configure(format, null, null, 0)
        codec.start()

        val out = ArrayList<Float>(sourceRate * 8)
        val info = MediaCodec.BufferInfo()
        var inputDone = false
        var outputDone = false
        var accumulator = 0.0
        var accumulated = 0
        val ratio = sourceRate.toDouble() / targetRate

        var position = 0.0

        try {
            while (!outputDone) {
                if (!inputDone) {
                    val index = codec.dequeueInputBuffer(10_000)
                    if (index >= 0) {
                        val buffer = codec.getInputBuffer(index)!!
                        val size = extractor.readSampleData(buffer, 0)
                        if (size < 0) {
                            codec.queueInputBuffer(index, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                            inputDone = true
                        } else {
                            codec.queueInputBuffer(index, 0, size, extractor.sampleTime, 0)
                            extractor.advance()
                        }
                    }
                }

                val index = codec.dequeueOutputBuffer(info, 10_000)
                if (index >= 0) {
                    val buffer = codec.getOutputBuffer(index)!!
                    buffer.order(ByteOrder.LITTLE_ENDIAN)
                    val shorts = buffer.asShortBuffer()
                    val frames = shorts.remaining() / channels
                    for (frame in 0 until frames) {
                        // Downmix to mono by averaging, which is what a phone's
                        // stereo recording of one speaker wants.
                        var sum = 0f
                        for (c in 0 until channels) sum += shorts.get() / 32768f
                        val mono = sum / channels

                        accumulator += mono
                        accumulated++
                        position += 1.0
                        if (position >= ratio) {
                            position -= ratio
                            out.add((accumulator / accumulated).toFloat())
                            accumulator = 0.0
                            accumulated = 0
                        }
                    }
                    codec.releaseOutputBuffer(index, false)
                    if (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) outputDone = true
                } else if (index == MediaCodec.INFO_TRY_AGAIN_LATER && inputDone) {
                    // Guard against a decoder that never signals end-of-stream.
                    if (out.size > targetRate * MAX_FILE_SECONDS) outputDone = true
                }

                if (out.size > targetRate * MAX_FILE_SECONDS) outputDone = true
            }
        } finally {
            runCatching { codec.stop() }
            runCatching { codec.release() }
            runCatching { extractor.release() }
        }

        return FloatArray(out.size) { out[it] }
    }

    private fun hasMicPermission(): Boolean =
        context.checkSelfPermission(Manifest.permission.RECORD_AUDIO) ==
            android.content.pm.PackageManager.PERMISSION_GRANTED

    private fun ArrayDeque<Float>.toFloatArray(): FloatArray {
        val array = FloatArray(size)
        var i = 0
        forEach { array[i++] = it }
        return array
    }

    private companion object {
        const val TAG = "Transcriber"
        const val SAMPLE_RATE = 16_000
        const val CHANNEL = AudioFormat.CHANNEL_IN_MONO
        const val ENCODING = AudioFormat.ENCODING_PCM_16BIT

        /** An hour. Past that the whole-file decode stops being a sane shape. */
        const val MAX_FILE_SECONDS = 3600
    }
}

sealed interface CaptureEvent {
    /** Waveform input level, emitted per read so the bars move with the voice. */
    data class Level(val peak: Float, val elapsedMillis: Long) : CaptureEvent

    data class Failed(val message: String) : CaptureEvent
}
