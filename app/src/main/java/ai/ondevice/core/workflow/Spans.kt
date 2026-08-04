package ai.ondevice.core.workflow

/**
 * Where a bracket closes.
 *
 * Pulled out of the runner and given nothing but a list of type strings, so it
 * can be tested without a device, an engine or a database. Every loop and
 * condition in a workflow is decided here, and the failures it used to have
 * were the quiet kind: a span that ran one step short, or a nested loop that
 * matched the wrong end. Neither reports anything — the run finishes and says
 * it worked.
 */
object Spans {

    /** Steps that open a span. */
    val OPENERS: Set<String> = setOf(
        NodeKind.RepeatStart.type,
        NodeKind.ForEachStart.type,
        NodeKind.Batch.type,
        NodeKind.Branch.type,
    )

    /** Steps that close one. */
    val CLOSERS: Set<String> = setOf(
        NodeKind.RepeatEnd.type,
        NodeKind.ForEachEnd.type,
    )

    /**
     * The index of the step closing the bracket opened at [from], or [limit]
     * when nothing closes it.
     *
     * [limit] rather than the last index, and that is the whole of one bug.
     * Returning `size - 1` for an unclosed span meant the body ran up to but
     * not including the final step, and the run then resumed past it — so a
     * Batch at the top of a graph with a Keep at the bottom repeated everything
     * correctly and silently never kept anything. One past the end is the
     * honest answer: an unclosed bracket reaches to the end of what it is
     * allowed to see, and the caller's `end + 1` then finishes the range.
     *
     * [limit] is also what makes nesting work, because a body is run with the
     * outer bracket's end as its limit and cannot reach past it for a closer.
     */
    fun end(types: List<String>, from: Int, limit: Int = types.size): Int {
        var depth = 0
        for (i in (from + 1) until minOf(limit, types.size)) {
            val type = types[i]
            when {
                type in OPENERS -> depth++
                type in CLOSERS -> if (depth == 0) return i else depth--
            }
        }
        return limit
    }
}
