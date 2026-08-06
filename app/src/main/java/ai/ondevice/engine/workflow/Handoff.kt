package ai.ondevice.engine.workflow

import ai.ondevice.core.Export

/**
 * Something a finished step wants to leave the app.
 *
 * The runner builds one of these and stops. It does not call `startActivity`,
 * and the reason is not tidiness: a run lives on a session scope with no
 * activity behind it, and an app that is not in the foreground cannot start
 * one — the launch is dropped, silently, with the step already reported as
 * done. Apps targeting Android 15 and above no longer even grant background
 * launch privileges to the pending intents they create, so the notification
 * fallback has to opt in explicitly as well.
 *
 * So the runner describes what should go where and hands it out through
 * [RunReporter], which is the same seam the Pick step already uses to ask a
 * question of a screen that did not exist when the run began.
 */
data class Handoff(
    val nodeId: String,
    /** Where the bytes are, staged for the file provider. Null for text only. */
    val export: Export?,
    /** The body, or the whole thing when there is no file. */
    val text: String,
    val subject: String,
    val target: HandoffTarget,
    /** A specific app, when one was chosen. Null means ask every time. */
    val packageName: String? = null,
    val label: String = "",
) {
    /** What the run screen and the notification call this. */
    val describe: String
        get() = when (target) {
            HandoffTarget.CLIPBOARD -> "Copied to the clipboard"
            HandoffTarget.APP -> "Send ${subject.ifBlank { export?.suggestedName ?: "this" }}" +
                (label.takeIf { it.isNotBlank() }?.let { " to $it" } ?: "")
        }
}

/**
 * Where a Send step puts things.
 *
 * Deliberately two, not four. A folder target would want a document-tree URI
 * picked before the run and persisted across a process death, and a "save to
 * the library" target is what the Keep step already is — so both would be a
 * second way to do something this app can already do, which is the kind of
 * choice that makes a palette hard to read.
 */
enum class HandoffTarget(val label: String) {
    /** The share sheet, or one named app. Needs a person, once. */
    APP("Another app"),

    /** Immediate, silent, and the only one that needs nobody. */
    CLIPBOARD("The clipboard"),
    ;

    companion object {
        fun of(raw: String?): HandoffTarget =
            entries.firstOrNull { it.name.equals(raw, ignoreCase = true) } ?: APP
    }
}
