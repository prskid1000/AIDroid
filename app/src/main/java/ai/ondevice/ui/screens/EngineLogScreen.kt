package ai.ondevice.ui.screens

import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.rememberScrollState
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.unit.dp
import ai.ondevice.engine.EngineLog
import ai.ondevice.engine.NativeLogTail
import ai.ondevice.ui.components.NButton
import ai.ondevice.ui.components.NButtonStyle
import ai.ondevice.ui.components.NHelp
import ai.ondevice.ui.components.NIconButton
import ai.ondevice.ui.components.NTag
import ai.ondevice.ui.components.NTagStyle
import ai.ondevice.ui.components.PhoneScaffold
import ai.ondevice.ui.components.PushToolbar
import ai.ondevice.ui.components.nClickableFlat
import ai.ondevice.ui.theme.NIcons
import ai.ondevice.ui.theme.NocturneColors
import ai.ondevice.ui.theme.NocturneType
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * What the engines are saying, as they say it.
 *
 * The engines were always this talkative — prompt tokens, cache hits, template
 * origin and a token rate every turn; the diffusion phase and its tile; the
 * shape of an ONNX graph that was refused. All of it went to logcat, which is
 * to say it was available exactly when a laptop was plugged in, and the
 * failures worth reading it for are the ones that happen on a phone in a pocket
 * forty minutes into a clip.
 *
 * Follows the tail while you are at the bottom and stops the moment you scroll
 * up, which is the one behaviour that makes a live log readable rather than
 * something that snatches the line you were reading.
 */
@Composable
fun EngineLogScreen(onBack: () -> Unit) {
    val entries by EngineLog.entries.collectAsStateWithLifecycle()
    val scope = rememberCoroutineScope()

    // Followed only while this screen is open. The native tail is a spawned
    // process reading a pipe, and keeping one alive for the life of the app to
    // service a screen nobody has opened is a cost with no reader.
    DisposableEffect(Unit) {
        NativeLogTail.start(scope)
        onDispose { NativeLogTail.stop() }
    }

    val clipboard = LocalClipboardManager.current
    var filter by rememberSaveable { mutableStateOf<String?>(null) }
    val listState = rememberLazyListState()

    val shown = remember(entries, filter) {
        if (filter == null) entries else entries.filter { it.source.label == filter }
    }

    // Only while already at the bottom. Following unconditionally means the log
    // yanks itself out from under anybody who scrolled back to read something.
    val atBottom by remember {
        androidx.compose.runtime.derivedStateOf {
            val last = listState.layoutInfo.visibleItemsInfo.lastOrNull()?.index ?: 0
            last >= shown.lastIndex - 1
        }
    }
    LaunchedEffect(shown.size) {
        if (atBottom && shown.isNotEmpty()) listState.scrollToItem(shown.lastIndex)
    }

    PhoneScaffold(
        toolbar = {
            PushToolbar(
                title = "Engine log",
                subtitle = "${shown.size} lines · newest last",
                subtitleMono = false,
                onBack = onBack,
                trailing = {
                    NIconButton(
                        NIcons.Copy,
                        "Copy the whole log",
                        onClick = {
                            clipboard.setText(
                                AnnotatedString(
                                    shown.joinToString("\n") {
                                        "${TIME.format(Date(it.at))} ${it.level} ${it.tag}: ${it.message}"
                                    },
                                ),
                            )
                        },
                        size = 32.dp,
                        iconSize = 14.dp,
                    )
                },
            )
        },
        contentPadding = PaddingValues(start = 18.dp, end = 18.dp, top = 4.dp, bottom = 18.dp),
    ) {
        Column {
            Row(
                Modifier
                    .fillMaxWidth()
                    .horizontalScroll(rememberScrollState())
                    .padding(bottom = 10.dp),
                horizontalArrangement = Arrangement.spacedBy(6.dp),
            ) {
                SourceChip("All", filter == null) { filter = null }
                EngineLog.Source.entries.forEach { source ->
                    val count = entries.count { it.source == source }
                    if (count == 0) return@forEach
                    SourceChip("${source.label} $count", filter == source.label) {
                        filter = if (filter == source.label) null else source.label
                    }
                }
            }

            if (shown.isEmpty()) {
                NHelp(
                    "Nothing yet. Load a model, make a picture or answer a message and the " +
                        "engines report here as they work — both this app's own lines and the " +
                        "ones llama.cpp, sd.cpp, whisper.cpp and ONNX Runtime write from C++. " +
                        "Kept for this session only.",
                )
                return@Column
            }

            LazyColumn(
                state = listState,
                modifier = Modifier.fillMaxWidth(),
                verticalArrangement = Arrangement.spacedBy(3.dp),
            ) {
                items(shown) { entry ->
                    Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                        Text(
                            TIME.format(Date(entry.at)),
                            style = NocturneType.Mono2Xs,
                            color = NocturneColors.TextMuted,
                        )
                        Text(
                            entry.level.toString(),
                            style = NocturneType.Mono2Xs,
                            color = levelColour(entry.level),
                        )
                        Column {
                            Text(
                                if (entry.native) "${entry.tag} ·native" else entry.tag,
                                style = NocturneType.Mono2Xs,
                                color = if (entry.native) {
                                    NocturneColors.Accent2300
                                } else {
                                    NocturneColors.Accent300
                                },
                            )
                            Text(entry.message, style = NocturneType.MonoXs)
                        }
                    }
                }
            }

            NButton(
                "Clear",
                onClick = { EngineLog.clear() },
                style = NButtonStyle.Ghost,
                modifier = Modifier.padding(top = 12.dp),
                block = true,
            )
        }
    }
}

@Composable
private fun SourceChip(label: String, selected: Boolean, onClick: () -> Unit) {
    NTag(
        label,
        style = if (selected) NTagStyle.Accent else NTagStyle.Neutral,
        modifier = Modifier.nClickableFlat(onClick = onClick),
    )
}

private fun levelColour(level: Char) = when (level) {
    'E' -> NocturneColors.Accent300
    'W' -> NocturneColors.Neutral300
    else -> NocturneColors.TextMuted
}

private val TIME = SimpleDateFormat("HH:mm:ss.SSS", Locale.UK)
