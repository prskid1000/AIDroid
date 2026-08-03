package ai.ondevice.tools

import android.app.Activity
import android.content.Intent
import android.os.Bundle
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.asSharedFlow

/** What came back on the redirect, good or bad. */
sealed interface OAuthRedirectResult {
    data class Code(val code: String, val state: String) : OAuthRedirectResult

    /** The user declined, or the server refused. `error` is OAuth's own code. */
    data class Failed(val error: String, val description: String?) : OAuthRedirectResult
}

/**
 * The one place a redirect is delivered.
 *
 * A shared flow rather than a stored value: the authorisation is being awaited
 * by a view model that is already running, and a replay cache would hand a
 * stale code to the *next* authorisation as though it had just arrived.
 */
object OAuthRedirects {
    private val _results = MutableSharedFlow<OAuthRedirectResult>(extraBufferCapacity = 4)
    val results: SharedFlow<OAuthRedirectResult> = _results.asSharedFlow()

    fun deliver(result: OAuthRedirectResult) {
        _results.tryEmit(result)
    }
}

/**
 * Catches `ai.ondevice://oauth/callback` and gets out of the way.
 *
 * Its own activity rather than an intent filter on MainActivity, because the
 * redirect arrives while the app is already running and routing it through the
 * main activity would either recreate it or drop the user somewhere other than
 * the screen they started from. This one reads the query, hands it over and
 * finishes without ever drawing, so the browser closes back onto the Tools
 * screen exactly as it was left.
 *
 * `taskAffinity` is empty and `excludeFromRecents` is set in the manifest so it
 * never appears as a second entry in the recents list.
 */
class OAuthCallbackActivity : Activity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val data = intent?.data
        val error = data?.getQueryParameter("error")
        val code = data?.getQueryParameter("code")
        val state = data?.getQueryParameter("state")

        val result = when {
            error != null -> OAuthRedirectResult.Failed(
                error,
                data.getQueryParameter("error_description"),
            )
            code != null && state != null -> OAuthRedirectResult.Code(code, state)
            else -> OAuthRedirectResult.Failed(
                "invalid_redirect",
                "The sign-in came back without a code.",
            )
        }
        OAuthRedirects.deliver(result)

        // Bring the app back in front of the browser.
        //
        // Finishing alone is not enough and looked exactly like failure: this
        // activity lives in its own task, so finishing it returns to whatever
        // was underneath — the browser, still showing the consent page that
        // was approved seconds ago. The sign-in had completed, the tokens
        // were stored, and the person was left looking at an Approve button,
        // reasonably concluding nothing had happened.
        //
        // SINGLE_TOP and CLEAR_TOP so the existing task is reordered to the
        // front rather than a second copy of the app being built on top of
        // the screen they started from.
        runCatching {
            startActivity(
                Intent(this, ai.ondevice.MainActivity::class.java).addFlags(
                    Intent.FLAG_ACTIVITY_NEW_TASK or
                        Intent.FLAG_ACTIVITY_CLEAR_TOP or
                        Intent.FLAG_ACTIVITY_SINGLE_TOP,
                ),
            )
        }

        finish()
        // No animation: this activity is a seam, and a cross-fade on it looks
        // like a screen that failed to load.
        overridePendingTransition(0, 0)
    }
}
