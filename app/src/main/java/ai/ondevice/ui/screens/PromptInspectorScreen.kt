package ai.ondevice.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import ai.ondevice.core.Fmt
import ai.ondevice.ui.components.NButton
import ai.ondevice.ui.components.NButtonStyle
import ai.ondevice.ui.components.NCard
import ai.ondevice.ui.components.NChipRow
import ai.ondevice.ui.components.NHelp
import ai.ondevice.ui.components.NInput
import ai.ondevice.ui.components.NSeg
import ai.ondevice.ui.components.NTag
import ai.ondevice.ui.components.NTagStyle
import ai.ondevice.ui.components.PhoneScaffold
import ai.ondevice.ui.components.PushToolbar
import ai.ondevice.ui.components.SectionKicker
import ai.ondevice.ui.theme.NocturneColors
import ai.ondevice.ui.theme.NocturneType
import ai.ondevice.ui.theme.Radius
import ai.ondevice.ui.theme.ring
import ai.ondevice.ui.vm.ChatViewModel

/**
 * **S10 — Prompt inspector.**
 *
 * SPEC §4.4 calls this "the single most useful debugging affordance for a local
 * LLM app and almost nothing ships it": the exact final string sent to the
 * tokenizer, its token count, and its token boundaries.
 *
 * Special tokens are painted from the accent ramp and ordinary tokens with an
 * alternating text tint, so boundaries are visible without a second colour —
 * the mono palette doing category work through value, not hue.
 */
@OptIn(ExperimentalLayoutApi::class)
@Composable
fun PromptInspectorScreen(
    onBack: () -> Unit,
    viewModel: ChatViewModel = activityChatViewModel(),
) {
    val state by viewModel.state.collectAsStateWithLifecycle()
    var tab by remember { mutableIntStateOf(0) }
    var showWhitespace by remember { mutableStateOf(false) }
    var addingStop by remember { mutableStateOf(false) }
    var newStop by remember { mutableStateOf("") }
    val clipboard = LocalClipboardManager.current

    LaunchedEffect(Unit) { viewModel.loadPromptInspector() }

    PhoneScaffold(
        toolbar = {
            PushToolbar(
                title = "Prompt inspector",
                subtitle = "What actually reaches the tokenizer",
                subtitleMono = false,
                onBack = onBack,
            )
        },
        contentPadding = PaddingValues(start = 18.dp, end = 18.dp, bottom = 18.dp),
    ) {
        NSeg(
            options = listOf("Rendered", "Tokens", "Template"),
            selectedIndex = tab,
            onSelect = { tab = it },
            modifier = Modifier.fillMaxWidth().padding(bottom = 10.dp),
        )

        val prompt = state.renderedPrompt
        if (prompt == null) {
            NHelp("Load a model and send a message — there is no prompt to inspect yet.")
            return@PhoneScaffold
        }

        Column(Modifier.verticalScroll(rememberScrollState())) {

            FlowRow(
                Modifier.fillMaxWidth().padding(bottom = 12.dp),
                horizontalArrangement = Arrangement.spacedBy(6.dp),
                verticalArrangement = Arrangement.spacedBy(6.dp),
            ) {
                NTag("${Fmt.grouped(prompt.totalTokens)} tokens", style = NTagStyle.Accent)
                NTag("of ${Fmt.grouped(prompt.contextLimit)}", style = NTagStyle.Neutral)
                if (prompt.imageTokens > 0) {
                    NTag("${Fmt.grouped(prompt.imageTokens)} image", style = NTagStyle.Neutral)
                }
                NTag("cache hit ${Fmt.grouped(prompt.cachedTokens)}", style = NTagStyle.Outline)
            }

            when (tab) {
                0 -> PromptBody(
                    buildAnnotatedString {
                        prompt.tokens.forEach { token ->
                            if (token.special) {
                                withStyle(
                                    SpanStyle(
                                        background = NocturneColors.Accent800,
                                        color = NocturneColors.Accent100,
                                    ),
                                ) { append(token.text.marked(showWhitespace)) }
                            } else {
                                append(token.text.marked(showWhitespace))
                            }
                        }
                    },
                )

                1 -> PromptBody(
                    buildAnnotatedString {
                        prompt.tokens.forEachIndexed { index, token ->
                            val style = when {
                                token.special -> SpanStyle(
                                    background = NocturneColors.Accent800,
                                    color = NocturneColors.Accent100,
                                )
                                // Alternating tints make each boundary visible
                                // without introducing a second hue.
                                index % 2 == 0 -> SpanStyle(
                                    background = NocturneColors.Text.copy(alpha = 0.05f),
                                )
                                else -> SpanStyle(background = NocturneColors.Text.copy(alpha = 0.08f))
                            }
                            withStyle(style) { append(token.text.marked(showWhitespace)) }
                        }
                    },
                )

                else -> PromptBody(
                    buildAnnotatedString {
                        append(
                            (prompt.template
                                ?: "No chat template in the GGUF metadata; the runtime default applies.")
                                .marked(showWhitespace),
                        )
                    },
                )
            }

            Row(
                Modifier.fillMaxWidth().padding(top = 10.dp),
                horizontalArrangement = Arrangement.spacedBy(7.dp),
            ) {
                NButton(
                    "Copy",
                    // The exact string, unmarked — a copy that pasted middle
                    // dots for spaces would be useless for the one thing you
                    // copy a prompt for, which is replaying it elsewhere.
                    onClick = {
                        clipboard.setText(
                            androidx.compose.ui.text.AnnotatedString(
                                if (tab == 2) {
                                    prompt.template.orEmpty()
                                } else {
                                    prompt.tokens.joinToString("") { it.text }
                                },
                            ),
                        )
                    },
                    modifier = Modifier.weight(1f),
                )
                NButton(
                    if (showWhitespace) "Hide whitespace" else "Show whitespace",
                    onClick = { showWhitespace = !showWhitespace },
                    style = if (showWhitespace) NButtonStyle.Primary else NButtonStyle.Secondary,
                    modifier = Modifier.weight(1f),
                )
            }

            SectionKicker("Chat template", Modifier.padding(top = 20.dp, bottom = 8.dp))
            NCard(gap = 6.dp) {
                Row(
                    Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(7.dp),
                    verticalAlignment = androidx.compose.ui.Alignment.CenterVertically,
                ) {
                    NTag(prompt.templateSource, style = NTagStyle.Outline)
                    androidx.compose.foundation.layout.Box(Modifier.weight(1f))
                    Text("Override", style = NocturneType.Meta, color = NocturneColors.Accent)
                }
                Text(
                    "Read from gguf.chat_template at load. Nothing about this model is hardcoded in " +
                        "the app.",
                    style = NocturneType.CardBody,
                    color = NocturneColors.Text.copy(alpha = 0.8f),
                )
            }

            SectionKicker("Stop sequences", Modifier.padding(top = 20.dp, bottom = 8.dp))
            NChipRow(
                chips = prompt.stopSequences + state.userStopSequences,
                onRemove = { index ->
                    // Only the user's own are removable; the template's two are
                    // what the model was trained to emit.
                    val userIndex = index - prompt.stopSequences.size
                    if (userIndex >= 0) viewModel.removeStopSequence(userIndex)
                },
                onAdd = { addingStop = true },
            )
            if (addingStop) {
                Row(
                    Modifier.fillMaxWidth().padding(top = 8.dp),
                    horizontalArrangement = Arrangement.spacedBy(7.dp),
                ) {
                    NInput(
                        value = newStop,
                        onValueChange = { newStop = it },
                        placeholder = "</s>",
                        textStyle = NocturneType.MonoCode,
                        modifier = Modifier.weight(1f),
                        minHeight = 42.dp,
                    )
                    NButton(
                        "Add",
                        onClick = {
                            if (newStop.isNotBlank()) viewModel.addStopSequence(newStop)
                            newStop = ""
                            addingStop = false
                        },
                        minHeight = 42.dp,
                    )
                }
            }
            NHelp(
                "${prompt.stopSequences.size} came from the template. Anything you add is yours and " +
                    "is kept per model.",
                Modifier.padding(top = 8.dp),
            )
        }
    }
}

/**
 * SPEC §4.4 — trailing spaces and stray newlines in a chat template are a
 * classic cause of "the model behaves differently to llama-cli", and they are
 * invisible until you make them visible.
 */
private fun String.marked(on: Boolean): String =
    if (!on) this else replace(" ", "·").replace("\t", "→   ").replace("\n", "⏎\n")

@Composable
private fun PromptBody(text: androidx.compose.ui.text.AnnotatedString) {
    Text(
        text,
        style = NocturneType.MonoBody,
        modifier = Modifier
            .fillMaxWidth()
            .background(NocturneColors.Neutral900, Radius.Md)
            .ring(NocturneColors.Divider, Radius.Md)
            .padding(12.dp),
    )
}
