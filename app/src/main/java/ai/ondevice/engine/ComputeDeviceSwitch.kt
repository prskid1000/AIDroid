package ai.ondevice.engine

import ai.ondevice.data.prefs.AppPrefs
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.flow.distinctUntilChanged
import kotlinx.coroutines.flow.drop
import kotlinx.coroutines.launch

/** Drops every loaded model the moment Settings → Compute device changes. */
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
                // The first emission is the stored value at startup, not a change to it.
                .drop(1)
                .collect { mode ->
                    android.util.Log.i(TAG, "compute device changed to $mode — unloading everything")
                    engines.unload()
                    transcriber.unload()
                    diffusion.unload()
                    // ONNX Runtime has no GPU provider on Android, so the two voice engines cannot act on this setting at all — but they are released anyway.
                    synthesizer.release()
                }
        }
    }

    private companion object {
        const val TAG = "ComputeDeviceSwitch"
    }
}
