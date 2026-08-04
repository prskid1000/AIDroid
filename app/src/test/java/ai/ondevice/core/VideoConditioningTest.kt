package ai.ondevice.core

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Which clip models are offered a first or last frame.
 *
 * The device can only ever answer for the one checkpoint installed on it, so
 * the discrimination is pinned here instead: the strings are the runtime's own
 * descriptions, and these cases are the branches it takes on them.
 */
class VideoConditioningTest {

    /** Takes both ends — and is the one this app is known to run. */
    @Test
    fun `TI2V-5B is offered both frames`() {
        assertTrue(VideoConditioning.supportsStartFrame("Wan2.2-TI2V-5B"))
        assertTrue(VideoConditioning.supportsEndFrame("Wan2.2-TI2V-5B"))
    }

    /**
     * The case the gate exists for.
     *
     * A text-to-video checkpoint matches no branch upstream, so a supplied
     * picture is dropped without a word and the clip returns as though none
     * had been given.
     */
    @Test
    fun `a text-to-video checkpoint is offered neither`() {
        assertFalse(VideoConditioning.supportsStartFrame("Wan2.x-T2V-14B"))
        assertFalse(VideoConditioning.supportsEndFrame("Wan2.x-T2V-14B"))
        assertFalse(VideoConditioning.supportsStartFrame("Wan2.x-VACE-14B"))
    }

    /**
     * A start frame and nowhere to put an end one.
     *
     * This asymmetry is why the two are asked separately rather than as one
     * "takes pictures" flag.
     */
    @Test
    fun `a plain I2V checkpoint is offered only the first`() {
        assertTrue(VideoConditioning.supportsStartFrame("Wan2.1-I2V-14B"))
        assertFalse(VideoConditioning.supportsEndFrame("Wan2.1-I2V-14B"))
        assertTrue(VideoConditioning.supportsStartFrame("Wan2.2-I2V-14B"))
        assertFalse(VideoConditioning.supportsEndFrame("Wan2.2-I2V-14B"))
    }

    /** FLF2V is trained for both ends and is the reason END_FRAME is not empty. */
    @Test
    fun `FLF2V is offered both`() {
        assertTrue(VideoConditioning.supportsStartFrame("Wan2.1-FLF2V-14B"))
        assertTrue(VideoConditioning.supportsEndFrame("Wan2.1-FLF2V-14B"))
    }

    /**
     * Unknown fails open, which is the whole point of writing this as a
     * denylist. Hiding the picker on a checkpoint this build has not heard of
     * would take away the one control that turns a smear into a clip.
     */
    @Test
    fun `an unrecognised model keeps both`() {
        assertTrue(VideoConditioning.supportsStartFrame("SomeFutureVideo-3B"))
        assertTrue(VideoConditioning.supportsEndFrame("SomeFutureVideo-3B"))
    }

    /** Nothing loaded yet is not the same as "no". */
    @Test
    fun `no model loaded keeps both`() {
        assertTrue(VideoConditioning.supportsStartFrame(null))
        assertTrue(VideoConditioning.supportsEndFrame(null))
        assertTrue(VideoConditioning.supportsStartFrame(""))
        assertTrue(VideoConditioning.supportsEndFrame("   "))
    }

    /** The runtime's capitalisation is its own business. */
    @Test
    fun `matching ignores case and surrounding space`() {
        assertFalse(VideoConditioning.supportsStartFrame("  wan2.x-t2v-14b  "))
        assertTrue(VideoConditioning.supportsEndFrame(" WAN2.2-TI2V-5B "))
    }
}
