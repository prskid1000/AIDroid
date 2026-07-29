package ai.ondevice.ui.screens

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import ai.ondevice.core.SparseParams
import ai.ondevice.core.displayValue
import ai.ondevice.data.db.GeneratedImageEntity
import ai.ondevice.ui.components.NButton
import ai.ondevice.ui.components.NButtonStyle
import ai.ondevice.ui.components.NHelp
import ai.ondevice.ui.components.NIconButton
import ai.ondevice.ui.components.NTable
import ai.ondevice.ui.components.NTableRow
import ai.ondevice.ui.components.PhoneScaffold
import ai.ondevice.ui.components.PushToolbar
import ai.ondevice.ui.components.SectionKicker
import ai.ondevice.ui.components.nClickableFlat
import ai.ondevice.ui.theme.NIcons
import ai.ondevice.ui.theme.NocturneColors
import ai.ondevice.ui.theme.NocturneType
import ai.ondevice.ui.theme.Radius
import ai.ondevice.ui.theme.ring
import ai.ondevice.ui.vm.GalleryViewModel

/**
 * **S13 — Gallery.**
 *
 * SPEC §5.4: every image carries its full generation parameters in the PNG's
 * own metadata, so the file is reproducible on its own — and "reuse
 * parameters" repopulates the whole form from it.
 *
 * The closing note is a real claim, not a flourish: the files sit in an
 * external app directory, browsable in any file manager (§13).
 */
@Composable
fun GalleryScreen(
    onBack: () -> Unit,
    viewModel: GalleryViewModel = hiltViewModel(),
    imageViewModel: ai.ondevice.ui.vm.ImageViewModel = activityImageViewModel(),
) {
    val images by viewModel.images.collectAsStateWithLifecycle()
    val selected by viewModel.selected.collectAsStateWithLifecycle()
    val focused = selected ?: images.firstOrNull()
    val context = LocalContext.current

    PhoneScaffold(
        toolbar = {
            PushToolbar("Gallery", onBack) {
                Text(images.size.toString(), style = NocturneType.Input, color = NocturneColors.TextMuted)
            }
        },
        contentPadding = PaddingValues(start = 18.dp, end = 18.dp, top = 4.dp, bottom = 18.dp),
    ) {
        if (images.isEmpty()) {
            NHelp(
                "Nothing generated yet. Every image is written with its full parameter set embedded in " +
                    "the PNG, so any result can be reproduced exactly — including by another app.",
            )
            return@PhoneScaffold
        }

        Column(Modifier.verticalScroll(rememberScrollState())) {

            focused?.let { image ->
                Box(
                    Modifier
                        .fillMaxWidth()
                        .aspectRatio(1f)
                        .clip(Radius.Md)
                        .background(NocturneColors.Neutral900)
                        .ring(NocturneColors.Divider, Radius.Md),
                    contentAlignment = Alignment.Center,
                ) {
                    GeneratedField(image.seed)
                    Text(
                        "${image.prompt.take(28)} · ${image.width}×${image.height}",
                        style = NocturneType.MonoXs,
                        color = NocturneColors.Accent100.copy(alpha = 0.8f),
                    )
                }

                Row(
                    Modifier.fillMaxWidth().padding(top = 8.dp),
                    horizontalArrangement = Arrangement.spacedBy(6.dp),
                ) {
                    NButton(
                        "Reuse parameters",
                        onClick = {
                            imageViewModel.reuseParameters(image)
                            onBack()
                        },
                        style = NButtonStyle.Primary,
                        modifier = Modifier.weight(1f),
                    )
                    NIconButton(
                        NIcons.Share,
                        "Share",
                        onClick = { shareImage(context, image) },
                        size = 46.dp,
                    )
                    NIconButton(NIcons.Trash, "Delete", onClick = { viewModel.delete(image) }, size = 46.dp)
                }

                SectionKicker("Embedded in the PNG", Modifier.padding(top = 18.dp, bottom = 8.dp))
                val params = SparseParams.parse(image.paramsJson)
                NTable {
                    val rows = buildList {
                        add("model" to (image.modelId?.substringAfterLast('/') ?: "—"))
                        add("seed" to image.seed.toString())
                        params.keys.filterNot { it == "seed" || it == "prompt" }.sorted().forEach { key ->
                            add(key to (params[key]?.displayValue() ?: "—"))
                        }
                    }
                    rows.forEach { (key, value) ->
                        NTableRow {
                            Text(
                                key,
                                style = NocturneType.Row,
                                color = NocturneColors.TextMuted,
                                modifier = Modifier.weight(0.44f),
                            )
                            Text(value, style = NocturneType.MonoValue, modifier = Modifier.weight(0.56f))
                        }
                    }
                }
            }

            SectionKicker("All · ${images.size}", Modifier.padding(top = 20.dp, bottom = 8.dp))
            LazyVerticalGrid(
                columns = GridCells.Fixed(3),
                modifier = Modifier.fillMaxWidth().height(240.dp),
                horizontalArrangement = Arrangement.spacedBy(6.dp),
                verticalArrangement = Arrangement.spacedBy(6.dp),
            ) {
                items(images, key = { it.id }) { image ->
                    Box(
                        Modifier
                            .aspectRatio(1f)
                            .clip(Radius.Sm)
                            .background(NocturneColors.Neutral900)
                            .nClickableFlat { viewModel.select(image) },
                    ) {
                        GeneratedField(image.seed)
                    }
                }
            }

            NHelp(
                "Files sit in a normal folder you can open in any file manager. Nothing is in a " +
                    "private store.",
                Modifier.padding(top = 12.dp),
            )
        }
    }
}

/**
 * Share the artifact itself when the file exists, and its parameter set when it
 * does not.
 *
 * The claim this screen makes is that the file is reproducible on its own, so
 * the share always carries the full parameter set as text alongside the image —
 * which is what makes "another app can reproduce it" true rather than a
 * flourish. The FileProvider grant is scoped to the receiving app and read-only.
 */
private fun shareImage(context: android.content.Context, image: GeneratedImageEntity) {
    val file = java.io.File(image.path)
    val params = SparseParams.parse(image.paramsJson)
    val text = buildString {
        appendLine(image.prompt)
        image.negativePrompt?.let { appendLine("negative: $it") }
        appendLine("seed: ${image.seed} · ${image.width}×${image.height}")
        image.modelId?.let { appendLine("model: $it") }
        params.keys.sorted().forEach { key ->
            if (key != "prompt" && key != "negative_prompt" && key != "seed") {
                appendLine("$key: ${params[key]?.displayValue() ?: "—"}")
            }
        }
    }

    val intent = android.content.Intent(android.content.Intent.ACTION_SEND).apply {
        putExtra(android.content.Intent.EXTRA_TEXT, text)
        if (file.exists()) {
            val uri = androidx.core.content.FileProvider.getUriForFile(
                context,
                "${context.packageName}.fileprovider",
                file,
            )
            type = "image/png"
            putExtra(android.content.Intent.EXTRA_STREAM, uri)
            addFlags(android.content.Intent.FLAG_GRANT_READ_URI_PERMISSION)
        } else {
            type = "text/plain"
        }
    }
    context.startActivity(android.content.Intent.createChooser(intent, "Share image"))
}

/**
 * A deterministic gradient standing in for the generated bitmap until sd.cpp is
 * wired in. Derived from the seed, so a given image always looks the same —
 * which keeps the gallery honest about identity even before pixels exist.
 */
@Composable
private fun GeneratedField(seed: Long) {
    val rng = kotlin.random.Random(seed)
    val cx = 0.25f + rng.nextFloat() * 0.5f
    val cy = 0.2f + rng.nextFloat() * 0.55f
    val palette = listOf(
        NocturneColors.Accent600,
        NocturneColors.Accent700,
        NocturneColors.Accent500,
        NocturneColors.SectionGlow,
        NocturneColors.SectionGhost,
        NocturneColors.Neutral700,
    )
    val tint = palette[(seed % palette.size).toInt().coerceAtLeast(0)]

    Canvas(Modifier.fillMaxSize()) {
        drawRect(NocturneColors.Neutral900)
        drawRect(
            Brush.radialGradient(
                colors = listOf(tint, Color.Transparent),
                center = Offset(size.width * cx, size.height * cy),
                radius = size.maxDimension * 0.6f,
            ),
        )
    }
}
