package ai.ondevice.workflow

import ai.ondevice.engine.workflow.HandoffDispatcher
import ai.ondevice.ui.vm.WorkflowSession
import android.content.Context
import android.content.Intent
import android.os.Bundle
import androidx.activity.ComponentActivity
import dagger.hilt.android.AndroidEntryPoint
import javax.inject.Inject

/**
 * What a parked hand-off's notification opens.
 *
 * It draws nothing and finishes at once. It exists because since Android 12 a
 * notification may not trampoline an activity start through a broadcast
 * receiver or a service — the `PendingIntent` has to be a `getActivity`, and
 * this is the activity it gets. Once here the app is foreground, so the chooser
 * it fires is an ordinary start with none of the background restrictions that
 * made the result wait in the first place.
 */
@AndroidEntryPoint
class HandoffActivity : ComponentActivity() {

    @Inject lateinit var session: WorkflowSession

    @Inject lateinit var dispatcher: HandoffDispatcher

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        deliver(intent)
        finish()
    }

    private fun deliver(intent: Intent?) {
        val nodeId = intent?.getStringExtra(EXTRA_NODE) ?: return
        val handoff = session.state.value.handoffs.firstOrNull { it.nodeId == nodeId } ?: return
        runCatching { startActivity(dispatcher.chooserFor(handoff)) }
            .onSuccess {
                session.state.value = session.state.value.copy(
                    handoffs = session.state.value.handoffs.filterNot { it.nodeId == nodeId },
                )
            }
    }

    companion object {
        private const val EXTRA_NODE = "ai.ondevice.extra.HANDOFF_NODE"

        fun intentFor(context: Context, nodeId: String): Intent =
            Intent(context, HandoffActivity::class.java)
                .putExtra(EXTRA_NODE, nodeId)
                .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
    }
}
