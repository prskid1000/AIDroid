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

/** Speech to text, for real (SPEC §6). */
class Transcriber(private val context: Context) {
    private val json = Json { ignoreUnknownKeys = true; isLenient = true }

    /** Held across every native call, so nothing frees the context while a decode is inside it. */
    private val nativeLock = java.util.concurrent.locks.ReentrantLock()

    @Volatile
    private var handle: Long = 0L

    @Volatile
    var loadedModelId: String? = null
        private set

    val available: Boolean get() = WhisperBridge.available
    val isLoaded: Boolean get() = handle != 0L

    fun isCurrent(modelId: String): Boolean = isLoaded && loadedModelId == modelId

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
                nativeLock.withLock {
                    unload()
                    EngineLog.i(TAG, "loading ${File(path).name}")
                    val newHandle = WhisperBridge.nativeLoad(path)
                    check(newHandle != 0L) { "The runtime returned no handle for $path." }
                    handle = newHandle
                    loadedModelId = modelId
                    if (!params.isEmpty) {
                        WhisperBridge.nativeApplyParams(handle, params.toJsonString())
                    }
                    EngineLog.i(TAG, "loaded ${File(path).name}")
                    WhisperBridge.nativeInfo(handle)
                }
                // A half-loaded context is worse than none: it reports a model
                // id the callers will trust and skip the reload for.
            }.onFailure { unload() }
        }

    fun unload() = nativeLock.withLock {
        if (handle != 0L) EngineLog.i(TAG, "unloading ${loadedModelId.orEmpty()}")
        if (handle != 0L) {
            WhisperBridge.nativeFree(handle)
            handle = 0L
        }
        loadedModelId = null
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

    /** Record until cancelled, writing the take to [captureTo]. */
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
            // The microphone is a shared, exclusive device.
            runCatching { record.stop() }
            runCatching { record.release() }
            activeRecord = null
            // A backstop, not the usual path: [finishCapture] closes the take the moment Stop is pressed, and both are idempotent.
            finishCapture()
            paused = false
        }
    }.flowOn(Dispatchers.IO).onCompletion {
        activeRecord?.let { runCatching { it.stop() }; runCatching { it.release() } }
        activeRecord = null
    }

    /** End the take now, without waiting for the loop to unwind. */
    fun finishCapture() {
        val writer = captureWriter
        captureWriter = null
        runCatching { writer?.close() }
        activeRecord?.let { runCatching { it.stop() } }
    }

    /** Paused keeps the microphone open rather than releasing it. */
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

    /** The same decode, at whatever rate the caller needs. */
    suspend fun decodeAudio(file: File, targetRate: Int): Result<FloatArray> =
        withContext(Dispatchers.Default) {
            runCatching {
                val samples = decodeToPcm(file, targetRate)
                check(samples.isNotEmpty()) { "No audio could be decoded from ${file.name}." }
                samples
            }
        }

    /** Transcribe audio the caller already has in memory, at [sampleRate]. */
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

    /**
     * Stop the pass in flight.
     *
     * Deliberately takes no [nativeLock]: that lock is held for the whole
     * length of the decode this is trying to stop, so waiting for it would
     * mean the Stop could only arrive once it was no longer needed.
     */
    fun cancel() {
        if (handle != 0L) WhisperBridge.nativeCancel(handle)
    }

    private fun decode(samples: FloatArray): List<TranscriptSegment> = nativeLock.withLock {
        if (handle == 0L) return emptyList()

        // A take is one uninterruptible JNI call, so there is no step inside it
        // to hint: the unit is the whole take, counted in audio samples, and the
        // rate has to be carried between calls or it would be learned only after
        // the run it was meant to steer had finished. The first two takes after a
        // load are therefore unhinted — that is CpuHints refusing to target a
        // number it has only seen once — and every one after them is hinted.
        // Keyed on the model because tiny and large-v3 do not decode a second of
        // audio at the same rate.
        val hints = CpuHints.open(context, TAG, carryOver = "whisper:${loadedModelId.orEmpty()}")
        val startedAt = System.currentTimeMillis()
        val transcribed = try {
            hints.unit(samples.size.toLong()) { WhisperBridge.nativeTranscribe(handle, samples) }
        } finally {
            hints.close()
        }
        // The decode said nothing about itself, which made the two ways it goes
        // wrong indistinguishable from the outside: audio that arrived empty or
        // at the wrong rate, and audio that arrived fine and decoded to nothing.
        // The realtime factor is the number worth having — under 1.0 means this
        // device is transcribing faster than the recording plays.
        val audioSeconds = samples.size.toFloat() / SAMPLE_RATE
        val tookSeconds = (System.currentTimeMillis() - startedAt) / 1000f
        EngineLog.i(
            TAG,
            "decoded ${"%.1f".format(audioSeconds)}s of audio in " +
                "${"%.1f".format(tookSeconds)}s " +
                "(${"%.2f".format(if (audioSeconds > 0f) tookSeconds / audioSeconds else 0f)}x realtime) " +
                "${samples.signalSummary(head = 0)}",
        )

        val result = json.parseToJsonElement(transcribed).jsonObject
        // A stopped pass has no segments and is not a failure. Reported as an
        // empty list, which is what every caller already does nothing with.
        if (result["cancelled"]?.jsonPrimitive?.content == "true") return emptyList()
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

    /** Container → mono float at [targetRate], defaulting to the 16 kHz whisper is the only consumer of. */
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
