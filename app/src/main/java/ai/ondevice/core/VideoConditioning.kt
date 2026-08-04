package ai.ondevice.core

/**
 * Whether a loaded clip model will actually *use* a picture you give it.
 *
 * This is not a preference and not a family trait — it is a property of the
 * particular checkpoint, and the Wan variants disagree with one another. The
 * runtime consumes a start frame in an enumerated set of branches, keyed on
 * the denoiser's own description; a checkpoint outside that set matches no
 * branch, and the picture is dropped without a word. The clip then comes back
 * as though none had been supplied, which is the worst way for this to fail:
 * nothing is reported, and the result is merely disappointing rather than
 * wrong in any way a person could point at.
 *
 * So the controls are shown only where they mean something. The alternative —
 * offering "first frame" on everything and letting most of them ignore it — is
 * an affordance that lies.
 *
 * The strings are the runtime's own, read from what it announces on load
 * rather than guessed from a filename. The set below mirrors the conditions in
 * upstream's `generate_video`: the four Wan I2V/FLF2V descs, TI2V-5B, and the
 * three families gated on version rather than desc (LTX-AV, Hunyuan Video,
 * LingBot Video), which announce themselves distinctly enough to match here.
 *
 * When nothing is known — no model loaded, or one that never announced a desc
 * this recognises — both are reported unsupported, so a control appears only
 * once there is a reason to believe it does something.
 */
object VideoConditioning {

    /**
     * Descs whose branch reads a *start* frame.
     *
     * `Wan2.x-T2V-14B` and `Wan2.x-VACE-14B` are deliberately absent: those are
     * the ones that silently discard it.
     */
    private val START_FRAME = setOf(
        "wan2.1-i2v-14b",
        "wan2.2-i2v-14b",
        "wan2.1-i2v-1.3b",
        "wan2.1-flf2v-14b",
        "wan2.2-ti2v-5b",
    )

    /**
     * Descs whose branch also reads an *end* frame.
     *
     * A shorter list than [START_FRAME], and that asymmetry is the reason the
     * two are asked separately. The plain Wan 2.1/2.2 I2V checkpoints take a
     * first frame and have nowhere to put a last one — FLF2V is the variant
     * trained for both, and TI2V-5B handles it in the same branch as its start
     * frame.
     */
    private val END_FRAME = setOf(
        "wan2.1-flf2v-14b",
        "wan2.2-ti2v-5b",
    )

    /** Families gated upstream on version, which take both ends. */
    private val BOTH_ENDS_BY_PREFIX = listOf("ltx", "hunyuan", "lingbot")

    /**
     * Descs known to *ignore* a supplied frame — the ones worth hiding for.
     *
     * The gate is written as a denylist rather than an allowlist, and that is
     * a deliberate choice about which way to be wrong. An allowlist fails
     * closed: a checkpoint whose desc this build does not recognise loses the
     * control altogether, and a first frame is the difference between a clip
     * and a smear on the one model this app is known to run. Hiding it there
     * would take away a working feature to protect against a hypothetical one.
     *
     * A denylist fails open. An unrecognised model keeps the picker, which at
     * worst wastes a picture on a model that drops it — the behaviour before
     * any of this existed — while the case actually worth catching, a plain
     * text-to-video checkpoint, is still caught.
     */
    private val IGNORES_FRAMES = setOf(
        "wan2.x-t2v-14b",
        "wan2.x-vace-14b",
    )

    fun supportsStartFrame(desc: String?): Boolean {
        val key = normalise(desc) ?: return true
        if (key in IGNORES_FRAMES) return false
        return true
    }

    fun supportsEndFrame(desc: String?): Boolean {
        val key = normalise(desc) ?: return true
        if (key in IGNORES_FRAMES) return false
        // Known to take a first frame and nothing at the other end. Everything
        // unrecognised keeps both, for the reason on IGNORES_FRAMES.
        return key !in START_FRAME || key in END_FRAME
    }

    private fun normalise(desc: String?): String? =
        desc?.trim()?.lowercase()?.takeIf { it.isNotEmpty() }
}
