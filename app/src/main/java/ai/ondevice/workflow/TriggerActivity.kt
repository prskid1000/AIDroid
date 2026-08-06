package ai.ondevice.workflow

import ai.ondevice.core.workflow.TriggerPayload
import ai.ondevice.core.workflow.TriggerValue
import ai.ondevice.core.workflow.Triggers
import ai.ondevice.core.workflow.WorkflowGraph
import ai.ondevice.data.AttachmentStore
import ai.ondevice.data.db.OnDeviceDatabase
import ai.ondevice.data.db.WorkflowEntity
import ai.ondevice.engine.workflow.WorkflowLauncher
import ai.ondevice.ui.theme.NocturneTheme
import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Build
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.lifecycleScope
import dagger.hilt.android.AndroidEntryPoint
import kotlinx.coroutines.launch
import javax.inject.Inject

/**
 * Where a workflow started from outside this app arrives.
 *
 * One activity, and the workflow named in the intent — because there is no
 * other shape available. An activity is declared in the manifest when the app is
 * built; a workflow is a row created long afterwards, and nothing registers an
 * activity at runtime. What *can* be made at runtime is a shortcut, and each
 * shortcut points here carrying an id. So the user sees N rows and the system
 * sees one class.
 *
 * It draws almost nothing. It is an activity for three reasons that are not
 * negotiable: `ACTION_SEND` resolves only to activities, the read permission on
 * an incoming `content://` URI is scoped to an activity's lifetime, and an
 * activity is foreground — which is what lets it legally start the inference
 * service. `OAuthCallbackActivity` is the same idea, and the same shape.
 */
@AndroidEntryPoint
class TriggerActivity : ComponentActivity() {

    @Inject lateinit var db: OnDeviceDatabase

    @Inject lateinit var attachments: AttachmentStore

    @Inject lateinit var launcher: WorkflowLauncher

    /** Watched only by the replace-in-place path, which waits for an answer. */
    @Inject lateinit var session: ai.ondevice.ui.vm.WorkflowSession

    private var screen by mutableStateOf<Screen>(Screen.Reading)

    /**
     * What this activity is showing, which is one of five things and never a tab.
     *
     * [Screen.Confirm] is the whole consent surface: any app on the device can
     * name any workflow, and a run here is minutes of somebody's battery with a
     * Send step possibly at the end of it. That is worth a sheet naming the
     * caller before anything loads.
     */
    private sealed interface Screen {
        data object Reading : Screen
        data class Pick(val payload: TriggerPayload, val options: List<WorkflowEntity>) : Screen
        data class Confirm(val payload: TriggerPayload, val workflow: WorkflowEntity) : Screen
        data class Refuse(val what: String, val because: String) : Screen

        /**
         * Held open while a run answers the selection it was handed.
         *
         * The only screen here that waits for the run rather than starting it
         * and standing aside, because `ACTION_PROCESS_TEXT` replaces the
         * caller's selection only if this activity is still alive to return a
         * result.
         */
        data class Working(val workflow: WorkflowEntity) : Screen
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            NocturneTheme {
                when (val current = screen) {
                    Screen.Reading -> TriggerReading()
                    is Screen.Pick -> TriggerPick(
                        payload = current.payload,
                        options = current.options,
                        // Through the confirmation, not around it. Picking says
                        // which workflow, and the sheet after it says what that
                        // workflow will send out — so a share that matched two
                        // workflows used to skip the one screen naming the step
                        // that mails the result somewhere.
                        onPick = { screen = Screen.Confirm(current.payload, it) },
                        onDismiss = ::finish,
                    )
                    is Screen.Confirm -> TriggerConfirm(
                        payload = current.payload,
                        workflow = current.workflow,
                        onRun = { start(current.payload, current.workflow) },
                        onDismiss = ::finish,
                    )
                    is Screen.Refuse -> TriggerRefuse(
                        what = current.what,
                        because = current.because,
                        onDismiss = ::finish,
                    )
                    is Screen.Working -> TriggerWorking(
                        workflow = current.workflow,
                        state = session.state,
                        onCancel = {
                            launcher.cancel()
                            setResult(RESULT_CANCELED)
                            finish()
                        },
                        onDone = ::returnToCaller,
                    )
                }
            }
        }
        handle(intent)
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        handle(intent)
    }

    private fun handle(intent: Intent?) {
        intent ?: return finish()
        lifecycleScope.launch {
            val payload = read(intent)
            val named = intent.getStringExtra(Intent.EXTRA_SHORTCUT_ID)
                ?: intent.getStringExtra(EXTRA_WORKFLOW)
                ?: intent.data?.takeIf { it.scheme == SCHEME }?.lastPathSegment

            if (launcher.busy) {
                screen = Screen.Refuse(
                    "Something is already running",
                    "This app holds one model's weights at a time, so it runs one workflow at a " +
                        "time. Open the app to see how far along it is.",
                )
                return@launch
            }

            /*
             * Checked before the workflow is chosen, not after.
             *
             * A file was offered and none of it could be read. Left to fall
             * through, a *named* workflow would have run anyway on an empty
             * payload — quietly using whatever was typed into its Input in the
             * editor instead of the picture somebody just shared, which looks
             * like the app ignoring them rather than failing.
             */
            if (payload.allUnreadable) {
                screen = Screen.Refuse(
                    if (payload.unreadable == 1) {
                        "That file could not be opened"
                    } else {
                        "Those files could not be opened"
                    },
                    "The app that shared it did not pass on permission to read it, or it has " +
                        "been moved since. Sharing it again usually works.",
                )
                return@launch
            }

            val chosen = named?.let { resolve(it) }
            if (named != null && chosen == null) {
                screen = Screen.Refuse(
                    "No workflow called that",
                    "Nothing here is named \"$named\". It may have been renamed or deleted.",
                )
                return@launch
            }

            if (chosen != null) {
                // Named, and it must still be able to take what arrived — a
                // shortcut outlives the edit that stopped its workflow
                // accepting pictures.
                val graph = WorkflowGraph.decode(chosen.graphJson)
                if (!payload.isEmpty && !Triggers.matches(graph, payload)) {
                    screen = Screen.Refuse(
                        "${chosen.name} cannot take that",
                        describeMismatch(graph, payload),
                    )
                    return@launch
                }
                screen = Screen.Confirm(payload, chosen)
                return@launch
            }

            val candidates = db.workflows().mostRecent(CANDIDATES).filter {
                val graph = WorkflowGraph.decode(it.graphJson)
                Triggers.sharedInputs(graph).isNotEmpty() && Triggers.matches(graph, payload)
            }
            screen = when {
                payload.isEmpty -> Screen.Refuse(
                    "Nothing arrived",
                    "That share carried no text and no file this app could read.",
                )
                candidates.isEmpty() -> Screen.Refuse(
                    "No workflow takes that",
                    "Open a workflow, add an Input step, and set where it comes from to " +
                        "\"from another app\". That is what makes it appear here.",
                )
                candidates.size == 1 -> Screen.Confirm(payload, candidates.first())
                else -> Screen.Pick(payload, candidates)
            }
        }
    }

    /**
     * Turn an intent into values, copying anything behind a URI in first.
     *
     * **The copy is the point.** An incoming `content://` grant is scoped to
     * this activity, and a run outlives every screen by design — so reading the
     * URI lazily when a step needs it fails minutes later with a permission
     * denial, in a place that has nothing to do with the cause.
     */
    private suspend fun read(intent: Intent): TriggerPayload {
        val values = mutableListOf<TriggerValue>()

        // Selected text, from the floating toolbar in another app.
        intent.getCharSequenceExtra(Intent.EXTRA_PROCESS_TEXT)?.toString()
            ?.takeIf { it.isNotBlank() }
            ?.let { values += TriggerValue(ai.ondevice.core.workflow.PortType.TEXT, text = it) }

        // Text shared as text — a caption, a link, a note. Not a file, so it
        // fills a TEXT slot and nothing else.
        intent.getCharSequenceExtra(Intent.EXTRA_TEXT)?.toString()
            ?.takeIf { it.isNotBlank() }
            ?.let { values += TriggerValue(ai.ondevice.core.workflow.PortType.TEXT, text = it) }

        var unreadable = 0
        streamsIn(intent).forEach { uri ->
            val attachment = attachments.copyIn(uri)
            if (attachment == null) {
                // Counted, not swallowed. copyIn answers null for every reason
                // there is — a refused read grant, a provider that has gone
                // away, a file deleted between the share and the tap — and
                // reporting that as "nothing arrived" describes the one cause
                // it is never going to be.
                unreadable++
            } else {
                val port = Triggers.portFor(attachment.mimeType)
                // A text file carries both: it is a file, and it is also the
                // text inside it. Which one a graph wanted is the graph's
                // business, not this boundary's.
                val text = if (
                    Triggers.readableAsText(attachment.mimeType, attachment.displayName)
                ) {
                    attachments.extractText(attachment).text
                } else {
                    ""
                }
                values += TriggerValue(
                    type = port,
                    text = text,
                    path = attachment.path,
                    displayName = attachment.displayName,
                )
            }
        }

        return TriggerPayload(
            values = values,
            fromPackage = runCatching { referrer?.host }.getOrNull(),
            readOnly = intent.getBooleanExtra(Intent.EXTRA_PROCESS_TEXT_READONLY, true),
            unreadable = unreadable,
        )
    }

    @Suppress("DEPRECATION")
    private fun streamsIn(intent: Intent): List<Uri> = when (intent.action) {
        Intent.ACTION_SEND_MULTIPLE ->
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                intent.getParcelableArrayListExtra(Intent.EXTRA_STREAM, Uri::class.java)
            } else {
                intent.getParcelableArrayListExtra(Intent.EXTRA_STREAM)
            }.orEmpty()
        else -> listOfNotNull(
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                intent.getParcelableExtra(Intent.EXTRA_STREAM, Uri::class.java)
            } else {
                intent.getParcelableExtra(Intent.EXTRA_STREAM)
            },
        )
    }.take(MAX_STREAMS)

    /** By id first, then by name — because both are things that get typed. */
    private suspend fun resolve(named: String): WorkflowEntity? {
        db.workflows().get(named)?.let { return it }
        val all = db.workflows().mostRecent(CANDIDATES)
        val exact = all.filter { it.name == named }
        if (exact.size == 1) return exact.first()
        val loose = all.filter { it.name.equals(named, ignoreCase = true) }
        // More than one match is not resolved by picking the newest. Falling
        // through to the picker costs one tap; guessing costs whatever the
        // wrong graph does over the next forty minutes.
        return loose.singleOrNull()
    }

    private fun describeMismatch(graph: WorkflowGraph, payload: TriggerPayload): String {
        val wants = Triggers.accepts(graph).joinToString(", ") { it.label.lowercase() }
        val got = payload.values.joinToString(", ") { it.type.label.lowercase() }
        return "It takes $wants, and what arrived was $got."
    }

    /**
     * Run it, and decide whether to wait for the answer or stand aside.
     *
     * Waiting is only offered where it is honest: the caller asked with
     * `ACTION_PROCESS_TEXT`, said it would accept a replacement, and the graph
     * is one this app can finish while somebody holds a phone — see
     * [Triggers.canReplaceInPlace]. Everything else starts the run and opens the
     * run screen, which is what a forty-minute graph deserves.
     */
    private fun start(payload: TriggerPayload, workflow: WorkflowEntity) {
        val graph = WorkflowGraph.decode(workflow.graphJson)
        if (
            intent?.action == Intent.ACTION_PROCESS_TEXT &&
            !payload.readOnly &&
            Triggers.canReplaceInPlace(graph)
        ) {
            if (launcher.launch(workflow.id, payload)) screen = Screen.Working(workflow)
            return
        }
        launcher.launch(workflow.id, payload)
        startActivity(
            Intent(this, ai.ondevice.MainActivity::class.java)
                .putExtra(
                    ai.ondevice.MainActivity.EXTRA_DESTINATION,
                    ai.ondevice.MainActivity.DEST_WORKFLOW_RUN,
                )
                .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_SINGLE_TOP),
        )
        finish()
    }

    /**
     * Hand the answer back so it replaces what was selected.
     *
     * `RESULT_OK` with `EXTRA_PROCESS_TEXT` is the whole contract: the calling
     * app swaps the selection for this. An empty answer returns cancelled
     * instead, because replacing somebody's paragraph with nothing is the one
     * outcome worse than doing nothing.
     */
    private fun returnToCaller(text: String?) {
        if (text.isNullOrBlank()) {
            setResult(RESULT_CANCELED)
        } else {
            setResult(
                RESULT_OK,
                Intent().putExtra(Intent.EXTRA_PROCESS_TEXT, text),
            )
        }
        finish()
    }

    companion object {
        const val ACTION_RUN_WORKFLOW = "ai.ondevice.action.RUN_WORKFLOW"
        const val EXTRA_WORKFLOW = "ai.ondevice.extra.WORKFLOW"

        /** Same scheme as the OAuth callback, a different host. */
        const val SCHEME = "ai.ondevice"

        /** How many workflows to consider when matching. */
        private const val CANDIDATES = 60

        /**
         * A ceiling on one share, not a guess at what anybody wants.
         *
         * Every stream is copied onto disk before this activity finishes, and a
         * gallery multi-select can be hundreds of pictures — which is a long
         * wait behind a transparent window that looks like nothing happening.
         */
        private const val MAX_STREAMS = 32

        fun intentFor(context: Context, workflowId: String): Intent =
            Intent(context, TriggerActivity::class.java)
                .setAction(ACTION_RUN_WORKFLOW)
                .putExtra(EXTRA_WORKFLOW, workflowId)
    }
}
