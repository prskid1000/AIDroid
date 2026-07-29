package ai.ondevice

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.ui.Modifier
import ai.ondevice.ui.OnDeviceApp
import ai.ondevice.ui.theme.NocturneColors
import ai.ondevice.ui.theme.NocturneTheme
import dagger.hilt.android.AndroidEntryPoint

@AndroidEntryPoint
class MainActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        enableEdgeToEdge()
        super.onCreate(savedInstanceState)

        val startDestination = intent?.getStringExtra(EXTRA_DESTINATION)

        setContent {
            NocturneTheme {
                OnDeviceApp(
                    modifier = Modifier.fillMaxSize().background(NocturneColors.Bg),
                    initialDestination = startDestination,
                )
            }
        }
    }

    companion object {
        const val EXTRA_DESTINATION = "destination"
        const val DEST_DOWNLOADS = "downloads"
    }
}
