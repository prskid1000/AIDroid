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
import kotlin.math.abs
import kotlin.math.min

/**
 * Speech to text, for real (SPEC §6).
 *
 * Two modes, one decoder:
 *
 *  - **Live** re-decodes a sliding window every `step_ms`. Whisper has no
 *    incremental decode, so a "partial" is genuinely a fresh transcription of
 *    overlapping audio — which is exactly why the UI fades text by confidence
 *    and says it may still change. The alternative, pretending each partial is
 *    final, would be a lie the model itself contradicts a second later.
 *  - **File** decodes any container Android can demux, resampled to the 16 kHz
 *    mono float whisper requires.
 *
 * The microphone is opened inside the flow and released in `onCompletion`, so
 * cancelling the recording actually frees the device rather than leaving it hot.
 */
class Transcriber(
    private val context: Context,
) {
    private val json = Json { ignoreUnknownKeys = true; isLenient = true }

    @Volatile
    private var handle: Long = 0L

    @Volatile
    var loadedModelId: String? = null
        private set

    val available: Boolean get() = WhisperBridge.available
    val isLoaded: Boolean get() = handle != 0L

    suspend fun load(modelId: String, path: String, params: SparseParams = SparseParams.EMPTY): Result<String> =
        withContext(Dispatchers.IO) {
            runCatching {
                check(WhisperBridge.available) {
                    WhisperBridge.loadError ?: "The whisper.cpp runtime is not installed in this build."
                }
                unload()
                val newHandle = WhisperBridge.nativeLoad(path)
                check(newHandle != 0L) { "The runtime returned no handle for $path." }
                handle = newHandle
                loadedModelId = modelId
                if (!params.isEmpty) WhisperBridge.nativeApplyParams(handle, params.toJsonString())
                WhisperBridge.nativeInfo(handle)
            }
        }

    fun unload() {
        if (handle != 0L) {
            WhisperBridge.nativeFree(handle)
            handle = 0L
        }
        loadedModelId = null
    }

    fun applyParams(params: SparseParams): ParamReport {
        if (handle == 0L) return ParamReport(rejected = params.keys.toList())
        val report = json.parseToJsonElement(
            WhisperBridge.nativeApplyParams(handle, params.toJsonString()),
        ).jsonObject
        return ParamReport(
            applied = report["applied"]?.jsonArray?.map { it.jsonPrimitive.content } ?: emptyList(),
            rejected = report["rejected"]?.jsonArray?.map { it.jsonPrimitive.content } ?: emptyList(),
        )
    }

    // — live —

    /**
     * Record and transcribe until cancelled.
     *
     * @param stepMillis how often to re-decode. Below about a second whisper
     *   spends more time on mel spectrograms than on new audio, so the manifest
     *   defaults this to 3 s and the screen says what it is.
     * @param windowMillis how much trailing audio each pass sees. Longer gives
     *   the decoder more context to correct itself with — which is why earlier
     *   words in a partial keep changing.
     */
    @SuppressLint("MissingPermission")
    fun listen(stepMillis: Int = 3_000, windowMillis: Int = 10_000): Flow<CaptureEvent> = flow {
        if (handle == 0L) {
            emit(CaptureEvent.Failed("No speech model is loaded."))
            return@flow
        }
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

        val window = ArrayDeque<Float>()
        val windowSamples = windowMillis * SAMPLE_RATE / 1000
        val stepSamples = stepMillis * SAMPLE_RATE / 1000
        val buffer = ShortArray(minBuffer / 2)
        var sinceLastDecode = 0
        var totalSamples = 0L

        try {
            while (true) {
                currentCoroutineContext().ensureActive()
                val read = record.read(buffer, 0, buffer.size)
                if (read <= 0) continue

                var peak = 0f
                for (i in 0 until read) {
                    val sample = buffer[i] / 32768f
                    window.addLast(sample)
                    if (abs(sample) > peak) peak = abs(sample)
                }
                while (window.size > windowSamples) window.removeFirst()
                sinceLastDecode += read
                totalSamples += read

                emit(
                    CaptureEvent.Level(
                        peak = peak,
                        elapsedMillis = totalSamples * 1000 / SAMPLE_RATE,
                    ),
                )

                if (sinceLastDecode >= stepSamples) {
                    sinceLastDecode = 0
                    val samples = window.toFloatArray()
                    val segments = withContext(Dispatchers.Default) { decode(samples) }
                    emit(CaptureEvent.Partial(segments, totalSamples * 1000 / SAMPLE_RATE))
                }
            }
        } finally {
            // The microphone is a shared, exclusive device. Releasing it here —
            // rather than trusting a caller to remember — is the difference
            // between cancelling a recording and hanging every other app that
            // wants to record.
            runCatching { record.stop() }
            runCatching { record.release() }
            activeRecord = null
        }
    }.flowOn(Dispatchers.IO).onCompletion {
        activeRecord?.let { runCatching { it.stop() }; runCatching { it.release() } }
        activeRecord = null
    }

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

    private fun decode(samples: FloatArray): List<TranscriptSegment> {
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

    /** A fresh decode of the trailing window. Earlier words may still change. */
    data class Partial(val segments: List<TranscriptSegment>, val elapsedMillis: Long) : CaptureEvent

    data class Failed(val message: String) : CaptureEvent
}
