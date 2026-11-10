package ai.ondevice.ui.theme

import androidx.compose.runtime.Immutable
import androidx.compose.ui.graphics.Color

/** Nocturne colour tokens, transcribed verbatim from the `:root` block of `Mobile AI model app design/_ds/nocturne-<id>/styles.css`. */
@Immutable
object NocturneColors {

    // — roles —
    val Bg = Color(0xFF0C0C0C)
    val Surface = Color(0xFF1A1D1A)
    val Text = Color(0xFFDDE3DC)

    /** `--color-accent`. */
    val Accent = Color(0xFF8FBF7F)

    /** `--color-accent-2`. */
    val Accent2 = Color(0xFFA8BEA7)

    /** `--color-divider`: `color-mix(in srgb, #dde3dc 16%, transparent)`. */
    val Divider = Text.copy(alpha = 0.16f)

    // — neutral ramp —
    val Neutral100 = Color(0xFFF2F5F1)
    val Neutral200 = Color(0xFFE2E8E1)
    val Neutral300 = Color(0xFFCCD4CB)
    val Neutral400 = Color(0xFFAFB8AE)
    val Neutral500 = Color(0xFF909A90)
    val Neutral600 = Color(0xFF727C72)
    val Neutral700 = Color(0xFF565F56)
    val Neutral800 = Color(0xFF3D433D)
    val Neutral900 = Color(0xFF282C28)

    // — accent ramp —
    val Accent100 = Color(0xFFF1F7EF)
    val Accent200 = Color(0xFFE0EDDB)
    val Accent300 = Color(0xFFC9E0C0)
    val Accent400 = Color(0xFFA9CF9B)
    val Accent500 = Color(0xFF8FBF7F)
    val Accent600 = Color(0xFF779F69)
    val Accent700 = Color(0xFF5C7C51)
    val Accent800 = Color(0xFF42583B)
    val Accent900 = Color(0xFF2A3627)

    // — accent-2 ramp (mono: reads the same as accent) —
    val Accent2100 = Color(0xFFF1F7EF)
    val Accent2200 = Color(0xFFE0EDDB)
    val Accent2300 = Color(0xFFC9E0C0)
    val Accent2400 = Color(0xFFB4C8B3)
    val Accent2500 = Color(0xFF98AE97)
    val Accent2600 = Color(0xFF7B917A)
    val Accent2700 = Color(0xFF5E705E)
    val Accent2800 = Color(0xFF434F43)
    val Accent2900 = Color(0xFF2B322B)

    /** Deck section-divider ground — saturation as presence. */
    val Section = Color(0xFF233023)
    val SectionGlow = Color(0xFF324332)
    val SectionGhost = Color(0xFF4E624E)

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

    // No `.dialog-backdrop` token.

    /** Disabled controls drop to 45% opacity. */
    const val DisabledAlpha = 0.45f
}
