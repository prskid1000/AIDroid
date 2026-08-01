package ai.ondevice.engine

import ai.ondevice.data.prefs.AppPrefs
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.flow.distinctUntilChanged
import kotlinx.coroutines.flow.drop
import kotlinx.coroutines.launch

/**
 * Drops every loaded model the moment Settings → Compute device changes.
 *
 * A context is built for one device and cannot move. Making each engine notice
 * that at its next load — "the id matches but the backend does not, so reload" —
 * is correct and is not enough on its own, for two reasons.
 *
 * The first is memory. Until something asks for the model again, the old
 * context is still resident: several gigabytes on a device the user has just
 * stopped using, and on this phone that can be an HTP session holding the NPU's
 * limited allocation against the load that is about to want it. The reload then
 * has to find room beside the thing it is replacing.
 *
 * The second is that "at its next load" is a promise three separate call sites
 * have to keep. Whisper, sd.cpp and llama.cpp each decide when to reload, and a
 * setting whose effect depends on every caller remembering to ask the right
 * question is a setting that will quietly stop working again the next time one
 * of them is touched. Unloading centrally means the *only* possible next state
 * is a fresh load, whoever asks for it.
 *
 * So the change is made to happen here, once, rather than inferred in three
 * places. The next use of any engine reloads on the new device because there is
 * nothing left to reuse.
 */
class ComputeDeviceSwitch(
    private val prefs: AppPrefs,
    private val engines: EngineManager,
    private val transcriber: Transcriber,
    private val diffusion: DiffusionEngine,
    private val synthesizer: ai.ondevice.speech.SpeechSynthesizer,
) {

    fun start(scope: CoroutineScope) {
        scope.launch {
            prefs.backendMode
                .distinctUntilChanged()
                // The first emission is the stored value at startup, not a
                // change to it. Unloading on that would tear down whatever the
                // app had just been asked to load.
                .drop(1)
                .collect { mode ->
                    android.util.Log.i(TAG, "compute device changed to $mode — unloading everything")
                    engines.unload()
                    transcriber.unload()
                    diffusion.unload()
                    // ONNX Runtime has no Hexagon provider, so the voice engines
                    // cannot act on this setting yet — but they are released
                    // anyway. They hold sessions worth hundreds of megabytes,
                    // and a device switch is the clearest signal the user is
                    // rearranging what runs where.
                    synthesizer.release()
                }
        }
    }

    private companion object {
        const val TAG = "ComputeDeviceSwitch"
    }
}
