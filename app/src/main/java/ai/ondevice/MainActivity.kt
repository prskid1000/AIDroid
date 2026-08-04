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
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.ui.Modifier
import ai.ondevice.ui.OnDeviceApp
import ai.ondevice.ui.theme.NocturneColors
import ai.ondevice.ui.theme.NocturneTheme
import dagger.hilt.android.AndroidEntryPoint

@AndroidEntryPoint
class MainActivity : ComponentActivity() {

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
                OnDeviceApp(
                    modifier = Modifier.fillMaxSize().background(NocturneColors.Bg),
                    initialDestination = destination.value,
                )
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
    }
}
