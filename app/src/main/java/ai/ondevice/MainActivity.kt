package ai.ondevice

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.background
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.getValue
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.launch
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.ui.Modifier
import ai.ondevice.ui.OnDeviceApp
import ai.ondevice.ui.theme.NocturneColors
import ai.ondevice.ui.theme.NocturneTheme
import dagger.hilt.android.AndroidEntryPoint

/** Distinguishes "nothing was ever saved" from "not read yet". */
private const val NOTHING_SAVED = ""

@AndroidEntryPoint
class MainActivity : ComponentActivity() {

    @javax.inject.Inject
    lateinit var prefs: ai.ondevice.data.prefs.AppPrefs

    /**
     * The one permission this app asks for, and it asks for it once.
     *
     * POST_NOTIFICATIONS has been declared in the manifest since the two
     * foreground services were written and was never requested, which on
     * Android 13 and later means declared and not held. The services ran —
     * Android allows that without it — and their notifications were dropped on
     * the floor, so a download or a generation carried on with nothing in the
     * shade to say it was happening, no progress, and no way to stop it
     * without coming back into the app.
     *
     * Refusing it costs nothing but the notification: the work still runs, and
     * nothing here asks twice.
     */
    private val askForNotifications = registerForActivityResult(
        ActivityResultContracts.RequestPermission(),
    ) { /* Granted or not, the services behave the same. */ }

    override fun onCreate(savedInstanceState: Bundle?) {
        enableEdgeToEdge()
        super.onCreate(savedInstanceState)

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU &&
            checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) !=
            PackageManager.PERMISSION_GRANTED
        ) {
            askForNotifications.launch(Manifest.permission.POST_NOTIFICATIONS)
        }

        destination.value = intent?.getStringExtra(EXTRA_DESTINATION)

        setContent {
            NocturneTheme {
                // Read before the NavHost is built, because a start
                // destination cannot be changed once it is. Null means the
                // read has not landed; the app waits for it rather than
                // starting on Chat and jumping a frame later.
                // Mapped through a sentinel so the two nulls stay apart: null
                // is "the read has not landed", and blank is "it landed and
                // there was nothing saved". Collapsing them left a first run
                // waiting forever for a value that was never coming, which is
                // a blank screen rather than a slow one.
                val stored by prefs.lastRoute
                    .map { it ?: NOTHING_SAVED }
                    .collectAsStateWithLifecycle(initialValue = null)

                if (stored != null || destination.value != null) {
                    OnDeviceApp(
                        modifier = Modifier.fillMaxSize().background(NocturneColors.Bg),
                        initialDestination = destination.value,
                        startRoute = stored?.takeIf { it != NOTHING_SAVED },
                        onRouteChanged = { route ->
                            lifecycleScope.launch { prefs.setLastRoute(route) }
                        },
                    )
                }
            }
        }
    }

    /**
     * Where a notification asked us to go, as state rather than a start value.
     *
     * The activity is `singleTask` now, so a tap on a notification reuses the
     * running instance and `onCreate` does not run again — which is the point,
     * because rebuilding it destroyed the view models holding a generation and
     * cancelled it. The cost is that an extra read once at startup would stop
     * working: the intent arrives here instead.
     */
    private val destination = mutableStateOf<String?>(null)

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        intent.getStringExtra(EXTRA_DESTINATION)?.let { destination.value = it }
    }

    companion object {

        const val EXTRA_DESTINATION = "destination"
        const val DEST_DOWNLOADS = "downloads"

        /** Where a share from another app sends you, once its run has started. */
        const val DEST_WORKFLOW_RUN = "workflow-run"
    }
}
