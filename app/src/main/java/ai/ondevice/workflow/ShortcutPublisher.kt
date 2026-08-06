package ai.ondevice.workflow

import ai.ondevice.R
import ai.ondevice.core.workflow.Triggers
import ai.ondevice.core.workflow.WorkflowGraph
import ai.ondevice.data.db.OnDeviceDatabase
import ai.ondevice.data.db.WorkflowEntity
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import androidx.core.content.pm.ShortcutInfoCompat
import androidx.core.content.pm.ShortcutManagerCompat
import androidx.core.graphics.drawable.IconCompat
import dagger.hilt.android.qualifiers.ApplicationContext
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Putting workflows in other apps' share sheets.
 *
 * A workflow cannot be an activity — the manifest is fixed when the app is
 * built and workflows are rows in a database made afterwards. What *can* be
 * made at runtime is a shortcut, and a shortcut is what the share sheet shows
 * as its own row. So there is one activity, one set of intent filters, and N
 * shortcuts over it.
 *
 * Which sheets a workflow appears in is not a setting. It is derived from the
 * Inputs marked *from another app*, through [Triggers.categoriesFor], and those
 * categories are matched against the `<share-target>` entries in
 * `res/xml/shortcuts.xml`. Adding the Input is what makes a workflow shareable;
 * there is nothing else to remember to turn on, and nothing to keep in step.
 */
@Singleton
class ShortcutPublisher @Inject constructor(
    @ApplicationContext private val context: Context,
    private val db: OnDeviceDatabase,
) {

    /**
     * Republish every workflow that accepts something from outside.
     *
     * The system caps how many may exist — [ShortcutManagerCompat.getMaxShortcutCountPerActivity],
     * and the launcher's own shortcuts share the budget — and the sheet
     * displays fewer still, four or five in portrait. So the cap decides what is
     * *visible*, never what is *reachable*: every workflow can still be started
     * by id from the picker or a deep link, which costs nothing.
     */
    suspend fun republish() {
        val eligible = db.workflows().mostRecent(LOOKBACK)
            .mapNotNull { workflow ->
                val categories = Triggers.categoriesFor(WorkflowGraph.decode(workflow.graphJson))
                if (categories.isEmpty()) null else workflow to categories
            }

        val budget = runCatching {
            ShortcutManagerCompat.getMaxShortcutCountPerActivity(context)
        }.getOrDefault(DEFAULT_BUDGET).coerceAtLeast(1)

        val keep = eligible.take(budget)
        runCatching {
            ShortcutManagerCompat.setDynamicShortcuts(
                context,
                keep.mapIndexed { index, (workflow, categories) ->
                    shortcutFor(workflow, categories, rank = index)
                },
            )
        }

        // Everything that no longer qualifies. Long-lived shortcuts are cached
        // by the system after they are unpublished — that is what makes their
        // ranking survive — so dropping one from the dynamic list is not enough
        // to stop it being offered.
        val stale = eligible.drop(budget).map { it.first.id }
        if (stale.isNotEmpty()) {
            runCatching { ShortcutManagerCompat.removeLongLivedShortcuts(context, stale) }
        }
    }

    /** Drop a workflow's row for good — it is gone, and its shortcut would lie. */
    fun forget(workflowId: String) {
        runCatching {
            ShortcutManagerCompat.removeLongLivedShortcuts(context, listOf(workflowId))
        }
    }

    /**
     * One workflow, as a row.
     *
     * **The id is the workflow's own id, and never reused.** That is not
     * bookkeeping: when a row is tapped in the share sheet the system does *not*
     * replay the intent stored here — it delivers the original `ACTION_SEND`
     * to the target class with `EXTRA_SHORTCUT_ID` added, and that id is the
     * only thing identifying which workflow was picked. The intent below is for
     * the launcher long-press path, where the reverse is true and no shortcut id
     * arrives.
     */
    private fun shortcutFor(
        workflow: WorkflowEntity,
        categories: Set<String>,
        rank: Int,
    ): ShortcutInfoCompat {
        val name = workflow.name.ifBlank { "Workflow" }
        return ShortcutInfoCompat.Builder(context, workflow.id)
            .setShortLabel(name)
            .setLongLabel(name)
            .setCategories(categories)
            .setLongLived(true)
            .setRank(rank)
            /*
             * Bound to the activity the share-target names, and this line is
             * load-bearing.
             *
             * Left unset, a shortcut is attributed to the app's *launcher*
             * activity — `dumpsys shortcut` said MainActivity — while
             * `shortcuts.xml` declares its targets against TriggerActivity. The
             * two have to agree for the system to offer a shortcut as a direct
             * share target, and the budget the shortcut count is drawn from is
             * per-activity as well.
             */
            .setActivity(ComponentName(context, TriggerActivity::class.java))
            .setIcon(IconCompat.createWithResource(context, R.drawable.ic_notify_generate))
            .setIntent(
                Intent(context, TriggerActivity::class.java)
                    .setAction(TriggerActivity.ACTION_RUN_WORKFLOW)
                    .putExtra(TriggerActivity.EXTRA_WORKFLOW, workflow.id),
            )
            .build()
    }

    private companion object {
        /**
         * How far back to look for candidates before the budget is applied.
         *
         * Ordered by most recently touched, and `touch` is already called at the
         * end of every run — so the ranking maintains itself with nothing here
         * to keep it fed.
         */
        const val LOOKBACK = 40

        /** Only used when the platform will not say. */
        const val DEFAULT_BUDGET = 5
    }
}
