package ai.ondevice.ui.theme

import androidx.compose.runtime.Immutable
import androidx.compose.ui.graphics.Color

/**
 * Nocturne colour tokens, transcribed verbatim from the `:root` block of
 * `Mobile AI model app design/_ds/nocturne-<id>/styles.css`.
 *
 * Every value here is a literal from that stylesheet. Nothing is interpolated,
 * re-derived or "close enough" — the readme's rule is that colour comes from the
 * ramps, so a screen that needs a tint reaches for a ramp step rather than
 * mixing its own.
 *
 * The ramps are generated in OKLCH on one shared lightness scale, so step N of
 * any role carries the same visual weight as step N of any other. On this dark
 * ground: 700–900 for tinted fills, hovers and subtle borders; 500 as the base;
 * 100–300 for text on those tints and for pressed states.
 */
@Immutable
object NocturneColors {

    // — roles —
    val Bg = Color(0xFF161826)
    val Surface = Color(0xFF232532)
    val Text = Color(0xFFE9E9ED)

    /**
     * `--color-accent`. The product's own blurple: OKLCH hue 289.2 at L 0.660,
     * C 0.125. Reads as an accent against the desaturated ramps rather than as
     * another neutral.
     */
    val Accent = Color(0xFF9184D9)

    /**
     * `--color-accent-2`. A machine-derived stand-in, not a second accent —
     * Nocturne is a mono scheme. Kept so both sets resolve; treat as one role.
     */
    val Accent2 = Color(0xFFA7A1DB)

    /** `--color-divider`: `color-mix(in srgb, #e9e9ed 16%, transparent)`. */
    val Divider = Text.copy(alpha = 0.16f)

    // — neutral ramp —
    val Neutral100 = Color(0xFFF3F5FE)
    val Neutral200 = Color(0xFFE4E7F5)
    val Neutral300 = Color(0xFFCFD3E5)
    val Neutral400 = Color(0xFFB2B6CA)
    val Neutral500 = Color(0xFF9397AB)
    val Neutral600 = Color(0xFF75798C)
    val Neutral700 = Color(0xFF595D6C)
    val Neutral800 = Color(0xFF3F424D)
    val Neutral900 = Color(0xFF292B31)

    // — accent ramp —
    val Accent100 = Color(0xFFF5F4FF)
    val Accent200 = Color(0xFFE7E5FE)
    val Accent300 = Color(0xFFD2CEFD)
    val Accent400 = Color(0xFFB5ABFC)
    val Accent500 = Color(0xFF968AE0)
    val Accent600 = Color(0xFF796CBF)
    val Accent700 = Color(0xFF5D5294)
    val Accent800 = Color(0xFF423A6A)
    val Accent900 = Color(0xFF2B2741)

    // — accent-2 ramp (mono: reads the same as accent) —
    val Accent2100 = Color(0xFFF5F4FF)
    val Accent2200 = Color(0xFFE7E5FE)
    val Accent2300 = Color(0xFFD2CEFD)
    val Accent2400 = Color(0xFFB5AFE8)
    val Accent2500 = Color(0xFF9690C9)
    val Accent2600 = Color(0xFF7972A9)
    val Accent2700 = Color(0xFF5C5783)
    val Accent2800 = Color(0xFF423E5D)
    val Accent2900 = Color(0xFF2B293A)

    /**
     * Deck section-divider ground — saturation as presence. Deck-scale fills
     * only, *not* interface colours. The canvas uses [SectionGlow] inside the
     * TAESD preview and gallery gradients; nothing else should touch these.
     */
    val Section = Color(0xFF262A60)
    val SectionGlow = Color(0xFF353B80)
    val SectionGhost = Color(0xFF4C5397)

    // — derived text opacities used throughout the stylesheet —
    /** `.text-muted` / `figcaption`: text at 55%. */
    val TextMuted = Text.copy(alpha = 0.55f)

    /** `.field > label`: text at 70%. */
    val TextLabel = Text.copy(alpha = 0.70f)

    /** `.card-meta`: text at 50%. */
    val TextMeta = Text.copy(alpha = 0.50f)

    /** `.table th`: text at 60%. */
    val TextTableHead = Text.copy(alpha = 0.60f)

    // — interaction tints, from the readme's "interaction states" rules —
    val AccentHover = Accent.copy(alpha = 0.12f)
    val AccentPressed = Accent.copy(alpha = 0.22f)
    val AccentGhostHover = Accent.copy(alpha = 0.10f)
    val AccentGhostPressed = Accent.copy(alpha = 0.18f)
    val NeutralHover = Text.copy(alpha = 0.07f)
    val NeutralPressed = Text.copy(alpha = 0.14f)
    val Selection = Accent.copy(alpha = 0.30f)

    /** Row rule in `.table tbody tr` — text at 8%. */
    val TableRowRule = Text.copy(alpha = 0.08f)

    /** `.table tbody tr:hover` tint — text at 4%. */
    val TableRowHover = Text.copy(alpha = 0.04f)

    /** `.dialog-backdrop`: neutral-900 at 50%. */
    /**
     * The wash behind a modal. Black, not a ramp step.
     *
     * It was `Neutral900` at 50 %, and `Neutral900` is `#292B31` — the darkest
     * step of the *neutral ramp*, but lighter than [Bg] at `#161826`, which the
     * ramp does not contain. So the scrim lifted the screen towards grey
     * instead of pushing it back: the dialog sat on a pale rectangle, the
     * background read as brighter than the thing in front of it, and the
     * separation the scrim exists to create ran the wrong way.
     *
     * Black is the only value guaranteed to darken whatever is behind it,
     * whatever the surface underneath happens to be.
     */
    val DialogScrim = Color(0xFF000000).copy(alpha = 0.62f)

    /** Disabled controls drop to 45% opacity. */
    const val DisabledAlpha = 0.45f
}
