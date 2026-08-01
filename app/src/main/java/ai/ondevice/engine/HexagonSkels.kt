package ai.ondevice.engine

import android.content.Context
import android.util.Log
import java.io.File

/**
 * Puts the NPU's device-side code somewhere the DSP can open it.
 *
 * The Hexagon backend is two programs. The half in `libondevice_llama.so` runs
 * on the CPU; the other half is `libggml-htp-v<NN>.so`, which runs on the DSP
 * and is loaded by the DSP's own loader, from a *path* — ggml asks fastRPC for
 * `file:///libggml-htp-v81.so` and the resolution happens on the far side,
 * against `ADSP_LIBRARY_PATH`.
 *
 * That is why these ship as assets rather than as jniLibs. A jniLib is not a
 * file anyone else can name: with `extractNativeLibs` off, which is the default
 * for this app, it lives inside the APK and only the dynamic linker knows the
 * trick for opening it. So they are unpacked once, into app storage, and the
 * search path is set to point there.
 *
 * All four architectures are staged rather than the one this phone needs, for
 * 2.8 MB total — the backend picks by asking the DSP its version, and a build
 * that stages by guess would be wrong on the first device that disagrees.
 *
 * Everything here is best-effort by design. No skels in the APK means no
 * `tools/build-hexagon.sh` was run, which is a build without an NPU, and that
 * is a supported build: the backend does not register, Settings says the NPU is
 * not available, and nothing else changes.
 */
object HexagonSkels {

    /**
     * Unpack the skels and point the DSP loader at them.
     *
     * Call before the first ggml call of any kind. ggml builds its registry
     * once and the Hexagon backend opens its DSP session while registering, so
     * a search path set afterwards is a search path set too late.
     */
    fun stage(context: Context) {
        if (!LlamaBridge.available) return
        runCatching {
            val skels = context.assets.list(ASSET_DIR).orEmpty()
            if (skels.isEmpty()) return

            val target = File(context.filesDir, ASSET_DIR)
            target.mkdirs()
            skels.forEach { name ->
                val file = File(target, name)
                // Size is the whole check. These are build artifacts pinned by
                // native/VERSIONS, so two files with the same name and length
                // are the same file, and a hash per launch buys nothing.
                context.assets.open("$ASSET_DIR/$name").use { input ->
                    if (file.length() == input.available().toLong()) return@use
                    file.outputStream().use(input::copyTo)
                }
            }

            // Ours first, then the platform's own, so a device whose vendor
            // ships something we would otherwise shadow still finds it.
            LlamaBridge.nativeSetDspSearchPath(
                listOf(target.absolutePath, "/vendor/lib/rfsa/adsp", "/vendor/dsp/cdsp", "/dsp")
                    .joinToString(";"),
            )
            stageQnnSkel(context, target)

            Log.i(TAG, "hexagon skels staged in ${target.absolutePath}")
        }.onFailure { Log.w(TAG, "hexagon skels not staged; NPU will not register", it) }
    }

    /**
     * The same problem again, for ONNX Runtime's half of the NPU.
     *
     * QNN is also two programs, and it also insists the two agree: the host
     * `libQnnHtp.so` in this APK is v2.42, and it refuses a DSP skel from any
     * other version. Ours ships as a jniLib, so — `extractNativeLibs` being off
     * — the DSP loader cannot open it, walks on to `/vendor/lib/rfsa/adsp`, and
     * finds the *platform's* skel instead. On this phone that is v2.39, and the
     * mismatch surfaces three layers from its cause:
     *
     *     QNN SetupBackend failed Failed to create device.
     *     Error: QNN_DEVICE_ERROR_INVALID_CONFIG: Invalid config values
     *
     * which reads like a bad option and is a wrong file. So the matching skel is
     * copied out of the APK — it is a stored zip entry, page-aligned and
     * uncompressed precisely because `extractNativeLibs` is off — into the
     * directory already first on `ADSP_LIBRARY_PATH`.
     *
     * Only the version this phone has. The APK carries six of them, v68 to v81,
     * at about 16 MB each; unpacking all six would spend 80 MB of the user's
     * storage to be ready for five DSPs this device does not have. Which one is
     * needed is not a guess: the platform ships exactly its own, so the vendor
     * directory names it.
     */
    private fun stageQnnSkel(context: Context, target: File) {
        runCatching {
            val version = File(VENDOR_ADSP).list().orEmpty()
                .firstNotNullOfOrNull { QNN_SKEL.matchEntire(it)?.groupValues?.get(1) }
                ?: return
            val name = "libQnnHtpV${version}Skel.so"
            val staged = File(target, name)

            java.util.zip.ZipFile(context.applicationInfo.sourceDir).use { apk ->
                val entry = apk.getEntry("lib/${android.os.Build.SUPPORTED_ABIS.first()}/$name")
                    ?: return
                if (staged.length() == entry.size) return
                apk.getInputStream(entry).use { input ->
                    staged.outputStream().use(input::copyTo)
                }
            }
            Log.i(TAG, "staged $name (${staged.length() / 1024 / 1024} MB) for ONNX Runtime")
        }.onFailure {
            // Not fatal to anything. ONNX Runtime falls back to its CPU
            // provider, which is what it did before this build had QNN at all.
            Log.w(TAG, "QNN skel not staged; ONNX models will run on the CPU", it)
        }
    }

    private const val TAG = "HexagonSkels"
    private const val ASSET_DIR = "hexagon"
    private const val VENDOR_ADSP = "/vendor/lib/rfsa/adsp"
    private val QNN_SKEL = Regex("""libQnnHtpV(\d+)Skel\.so""")
}
