package ai.ondevice.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.RowScope
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.asPaddingValues
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.navigationBars
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.layout.statusBars
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.foundation.verticalScroll
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import ai.ondevice.ui.theme.NIcons
import ai.ondevice.ui.theme.NocturneColors
import ai.ondevice.ui.theme.NocturneType
import ai.ondevice.ui.theme.Space
import ai.ondevice.ui.theme.ruleAbove

/** The frame every one of the fifteen screens sits in. */
@Composable
fun PhoneScaffold(
    modifier: Modifier = Modifier,
    toolbar: (@Composable () -> Unit)? = null,
    bottomBar: (@Composable () -> Unit)? = null,
    contentPadding: PaddingValues = PaddingValues(
        start = Space.ScreenGutter,
        end = Space.ScreenGutter,
        bottom = Space.ScreenGutter,
    ),
    content: @Composable ColumnScope.() -> Unit,
) {
    Column(
        modifier
            .fillMaxSize()
            .background(NocturneColors.Bg)
            .padding(top = WindowInsets.statusBars.asPaddingValues().calculateTopPadding()),
    ) {
        if (toolbar != null) toolbar()
        Column(
            Modifier
                .weight(1f)
                .fillMaxWidth()
                .padding(contentPadding),
            content = content,
        )
        if (bottomBar != null) bottomBar()
    }
}

/** A pushed screen's toolbar: back chevron, title, optional subtitle, optional trailing action. */
@Composable
fun PushToolbar(
    title: String,
    onBack: () -> Unit,
    modifier: Modifier = Modifier,
    subtitle: String? = null,
    subtitleMono: Boolean = true,
    trailing: (@Composable RowScope.() -> Unit)? = null,
) {
    Row(
        modifier = modifier
            .fillMaxWidth()
            .padding(start = Space.ScreenGutter, end = Space.ScreenGutter, top = 12.dp, bottom = 10.dp),
        horizontalArrangement = Arrangement.spacedBy(10.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Icon(
            NIcons.ChevronLeft,
            contentDescription = "Back",
            tint = NocturneColors.Text,
            modifier = Modifier.size(20.dp).nClickableFlat(onClick = onBack),
        )
        Column(Modifier.weight(1f)) {
            Text(title, style = NocturneType.ScreenTitle, maxLines = 1, overflow = TextOverflow.Ellipsis)
            if (subtitle != null) {
                Text(
                    subtitle,
                    style = if (subtitleMono) NocturneType.MonoXs else NocturneType.Help,
                    color = NocturneColors.TextMuted,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                )
            }
        }
        if (trailing != null) trailing()
    }
}

/** A root destination's header: a title flush left, actions on the right. */
@Composable
fun RootToolbar(
    title: String,
    modifier: Modifier = Modifier,
    subtitle: (@Composable () -> Unit)? = null,
    trailing: (@Composable RowScope.() -> Unit)? = null,
) {
    Row(
        modifier = modifier
            .fillMaxWidth()
            .padding(start = Space.ScreenGutter, end = Space.ScreenGutter, top = 12.dp, bottom = 8.dp),
        horizontalArrangement = Arrangement.spacedBy(10.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Column(Modifier.weight(1f)) {
            Text(
                title,
                style = if (subtitle == null) NocturneType.RootTitle else NocturneType.CardTitle,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
            )
            if (subtitle != null) subtitle()
        }
        if (trailing != null) trailing()
    }
}

/** One toolbar action, so the icon size and hit target are decided once. */
@Composable
fun ToolbarAction(
    icon: ImageVector,
    contentDescription: String,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    tint: Color = NocturneColors.Text,
) {
    Icon(
        icon,
        contentDescription = contentDescription,
        tint = tint,
        modifier = modifier.size(20.dp).nClickableFlat(onClick = onClick),
    )
}

/** A toolbar action that is also a state — one of a set, with one of them on. */
@Composable
fun ToolbarToggle(
    icon: ImageVector,
    contentDescription: String,
    selected: Boolean,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
) {
    Box(
        modifier
            .size(32.dp)
            .clip(RoundedCornerShape(9.dp))
            .background(if (selected) NocturneColors.Accent900 else Color.Transparent)
            .then(
                if (selected) {
                    Modifier.border(1.dp, NocturneColors.Accent700, RoundedCornerShape(9.dp))
                } else {
                    Modifier
                },
            )
            .nClickableFlat(onClick = onClick),
        contentAlignment = Alignment.Center,
    ) {
        Icon(
            icon,
            contentDescription = contentDescription,
            tint = if (selected) NocturneColors.Accent200 else NocturneColors.TextMuted,
            modifier = Modifier.size(18.dp),
        )
    }
}

/** One destination in the bottom bar. */
data class NavDestination(
    val label: String,
    val icon: ImageVector,
    val route: String,
)

/** The five-item bottom bar. */
@Composable
fun NBottomBar(
    destinations: List<NavDestination>,
    currentRoute: String?,
    modifier: Modifier = Modifier,
    onSelect: (NavDestination) -> Unit,
) {
    Row(
        modifier = modifier
            .fillMaxWidth()
            .ruleAbove()
            .background(NocturneColors.Bg)
            .padding(
                start = 6.dp,
                end = 6.dp,
                top = 7.dp,
                bottom = 12.dp + WindowInsets.navigationBars.asPaddingValues().calculateBottomPadding(),
            ),
    ) {
        destinations.forEach { dest ->
            val selected = currentRoute?.startsWith(dest.route) == true
            Column(
                modifier = Modifier
                    .weight(1f)
                    .nClickableFlat { onSelect(dest) }
                    .padding(vertical = 5.dp)
                    .alpha(if (selected) 1f else 0.5f),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.spacedBy(3.dp),
            ) {
                Icon(
                    dest.icon,
                    contentDescription = dest.label,
                    tint = if (selected) NocturneColors.Accent else NocturneColors.Text,
                    modifier = Modifier.size(21.dp),
                )
                Text(
                    dest.label,
                    style = NocturneType.NavLabel,
                    color = if (selected) NocturneColors.Accent else NocturneColors.Text,
                )
            }
        }
    }
}

/** A modal bottom sheet in the canvas' proportions: the ground dimmed, the sheet on `--color-surface` with a 20px top radius and a neutral-700 grabber. */
@Composable
fun NSheetHandle(modifier: Modifier = Modifier) {
    Box(modifier.fillMaxWidth().padding(top = 9.dp, bottom = 3.dp), contentAlignment = Alignment.Center) {
        Box(
            Modifier
                .size(width = 34.dp, height = 4.dp)
                .background(NocturneColors.Neutral700, ai.ondevice.ui.theme.Radius.Sm),
        )
    }
}

/** How much of the screen a quick-settings sheet covers, whatever it holds. */
private const val SHEET_HEIGHT_FRACTION = 0.9f

/** The sheet every quick-settings panel opens in. */
@Composable
fun NBottomSheet(
    title: String,
    onDismiss: () -> Unit,
    modifier: Modifier = Modifier,
    note: String? = null,
    content: @Composable ColumnScope.() -> Unit,
) {
    androidx.compose.ui.window.Dialog(
        onDismissRequest = onDismiss,
        properties = androidx.compose.ui.window.DialogProperties(
            usePlatformDefaultWidth = false,
            // Without this the dialog's window consumes the system-bar insets
            // before anything inside can read them, so the sheet's own
            // navigation-bar padding measured zero and its last control sat
            // under the gesture strip with nowhere left to scroll.
            decorFitsSystemWindows = false,
        ),
    ) {
        Box(
            Modifier
                .fillMaxSize()
                .background(NocturneColors.Neutral900.copy(alpha = 0.55f))
                .nClickableFlat(onClick = onDismiss),
            contentAlignment = Alignment.BottomCenter,
        ) {
            Column(
                modifier
                    .fillMaxWidth()
                    // A fixed fraction of the screen, not "as tall as the content".
                    .fillMaxHeight(SHEET_HEIGHT_FRACTION)
                    .background(
                        NocturneColors.Surface,
                        androidx.compose.foundation.shape.RoundedCornerShape(20.dp, 20.dp, 0.dp, 0.dp),
                    )
                    // Swallowed so a tap inside the sheet does not reach the
                    // scrim behind it and close the thing being tapped.
                    .nClickableFlat { },
            ) {
                NSheetHandle()
                Row(
                    Modifier.fillMaxWidth().padding(start = 18.dp, end = 18.dp, top = 6.dp, bottom = 12.dp),
                    horizontalArrangement = Arrangement.spacedBy(9.dp),
                    verticalAlignment = Alignment.Bottom,
                ) {
                    Text(title, style = NocturneType.SheetTitle, modifier = Modifier.weight(1f))
                    if (note != null) {
                        Text(note, style = NocturneType.Meta, color = NocturneColors.TextMuted)
                    }
                }
                Column(
                    Modifier
                        .weight(1f)
                        .verticalScroll(androidx.compose.foundation.rememberScrollState())
                        .padding(start = 18.dp, end = 18.dp)
                        // The navigation bar sits over the sheet, not beside
                        // it, and the gesture strip over that. Enough room that
                        // the last control in a sheet is a control rather than
                        // something you can see but not reach.
                        .padding(
                            bottom = 24.dp +
                                WindowInsets.navigationBars.asPaddingValues().calculateBottomPadding(),
                        )
                        // Belt as well as braces: a device that reports no
                        // navigation-bar inset at all still leaves a thumb's
                        // width under the last row, clear of the gesture strip.
                        .padding(bottom = 40.dp),
                    content = content,
                )
            }
        }
    }
}

/** The small live-state dot the canvas puts next to the backend readout in the chat header and next to a modified parameter. */
@Composable
fun NDot(
    modifier: Modifier = Modifier,
    color: Color = NocturneColors.Accent,
    size: androidx.compose.ui.unit.Dp = 5.dp,
) {
    Box(modifier.size(size).background(color, androidx.compose.foundation.shape.CircleShape))
}
