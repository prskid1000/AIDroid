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
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import ai.ondevice.core.Fmt
import ai.ondevice.ui.components.NButton
import ai.ondevice.ui.components.NCard
import ai.ondevice.ui.components.NChipRow
import ai.ondevice.ui.components.NHelp
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
                                ) { append(token.text) }
                            } else {
                                append(token.text)
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
                            withStyle(style) { append(token.text) }
                        }
                    },
                )

                else -> PromptBody(
                    buildAnnotatedString {
                        append(prompt.template ?: "No chat template in the GGUF metadata; the runtime default applies.")
                    },
                )
            }

            Row(
                Modifier.fillMaxWidth().padding(top = 10.dp),
                horizontalArrangement = Arrangement.spacedBy(7.dp),
            ) {
                NButton("Copy", onClick = {}, modifier = Modifier.weight(1f))
                NButton("Show whitespace", onClick = {}, modifier = Modifier.weight(1f))
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
            NChipRow(chips = prompt.stopSequences, onAdd = {})
            NHelp(
                "${prompt.stopSequences.size} came from the template. Anything you add is yours and " +
                    "is kept per model.",
                Modifier.padding(top = 8.dp),
            )
        }
    }
}

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
