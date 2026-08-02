package ai.ondevice.params

import android.content.Context
import ai.ondevice.core.SparseParams
import ai.ondevice.core.Tier
import ai.ondevice.data.db.OnDeviceDatabase
import ai.ondevice.data.db.ParamManifestEntity
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonElement

/** Loads the manifest and answers "what should this screen render?". */
class ParamRepository(
    private val context: Context,
    private val db: OnDeviceDatabase,
) {
    private val json = Json {
        ignoreUnknownKeys = true
        isLenient = true
        coerceInputValues = true
    }

    @Volatile
    private var cached: ParamManifest? = null

    suspend fun manifest(): ParamManifest = cached ?: withContext(Dispatchers.IO) {
        val bundled = loadBundled()
        val stored = db.manifests().newest()
        // Only a signature-verified OTA manifest may win.
        val chosen = when {
            stored != null && stored.signatureOk && stored.version > bundled.manifestVersion ->
                runCatching { json.decodeFromString(ParamManifest.serializer(), stored.json) }.getOrDefault(bundled)
            else -> bundled
        }
        cached = chosen
        chosen
    }

    private fun loadBundled(): ParamManifest = runCatching {
        context.assets.open(ASSET).bufferedReader().use { it.readText() }
            .let { json.decodeFromString(ParamManifest.serializer(), it) }
    }.getOrElse { ParamManifest(manifestVersion = 0) }

    suspend fun bundledVersion(): Int = withContext(Dispatchers.IO) { loadBundled().manifestVersion }

    /** The parameters a runtime *has*, as opposed to the ones a document claims for it. */
    fun specsFor(manifest: ParamManifest, runtimeId: String): List<ParamSpec> {
        val described = manifest.paramsFor(runtimeId)
        val declared = EngineParams.capabilities(runtimeId)?.takeIf { it.isNotEmpty() } ?: return described

        val kept = described
            .filter { it.key in declared }
            .map { spec ->
                // The reload flag and the default belong to the runtime.
                val capability = declared.getValue(spec.key)
                if (capability.appliedBy == ParamCapability.Applier.RUNTIME) {
                    spec.copy(
                        requiresReload = capability.requiresReload,
                        default = capability.default ?: spec.default,
                    )
                } else {
                    spec
                }
            }

        val describedKeys = described.mapTo(mutableSetOf()) { it.key }
        val undescribed = declared.values
            .filter { it.key !in describedKeys }
            .sortedBy { it.key }
            .map { capability ->
                ParamSpec(
                    key = capability.key,
                    group = UNDESCRIBED_GROUP,
                    type = ParamType.TEXT,
                    tier = Tier.EXPERT,
                    label = capability.key,
                    help = "This build's runtime accepts this parameter and the manifest has no " +
                        "description for it. The value is passed through as typed.",
                    requiresReload = capability.requiresReload,
                )
            }

        return kept + undescribed
    }

    suspend fun storedManifest(): ParamManifestEntity? = db.manifests().newest()

    /**
     * The keys a runtime offers for one architecture — the same `appliesTo`
     * gate the parameter screen applies, asked as a question instead of
     * rendered as a list.
     *
     * Anything choosing a file for a role needs this: a T5-XXL in the library
     * is a legitimate encoder for FLUX and nonsense for SDXL, and the manifest
     * is where that already knows.
     */
    suspend fun applicableKeys(
        runtimeId: String,
        modality: String?,
        architecture: String?,
    ): Set<String> = specsFor(manifest(), runtimeId)
        .filter { appliesToMatches(it, modality, architecture) }
        .map { it.key }
        .toSet()

    /**
     * The parameter set for a screen, each row carrying why it cannot be
     * edited — or null, when it can.
     *
     * Gated rows used to be dropped, and the screen said so in aggregate: "11
     * hidden by dependsOn or build gate". That is a count of things you cannot
     * see, cannot name and cannot act on. A parameter that does not apply to
     * the loaded model is worth showing precisely *because* it does not: it
     * says what this model does not have, and what would bring it back.
     *
     * The search box still filters, because that is the user's own filter
     * rather than the app's.
     */
    fun visible(
        specs: List<ParamSpec>,
        values: SparseParams,
        loadedBuildTag: String?,
        modality: String?,
        architecture: String?,
        query: String = "",
    ): VisibleParams {
        val afterSearch = if (query.isBlank()) {
            specs
        } else {
            val q = query.trim().lowercase()
            specs.filter {
                it.key.contains(q, true) || it.label.contains(q, true) || it.help.contains(q, true)
            }
        }

        val rows = afterSearch.map { spec ->
            GatedParam(
                spec = spec,
                disabledBecause = appliesToRefusal(spec, modality, architecture)
                    ?: buildRefusal(spec, loadedBuildTag)
                    ?: dependencyRefusal(spec, values),
            )
        }

        return VisibleParams(
            rows = rows,
            shownCount = rows.count { it.disabledBecause == null },
            disabledCount = rows.count { it.disabledBecause != null },
            totalCount = specs.size,
        )
    }

    private fun appliesToMatches(spec: ParamSpec, modality: String?, architecture: String?): Boolean =
        appliesToRefusal(spec, modality, architecture) == null

    private fun appliesToRefusal(
        spec: ParamSpec,
        modality: String?,
        architecture: String?,
    ): String? {
        val applies = spec.appliesTo ?: return null
        applies.modality?.let { kinds ->
            if (modality != null && modality !in kinds) {
                return "Only for ${kinds.joinToString(", ")} models. This one is $modality."
            }
        }
        applies.arch?.let { gates ->
            if (architecture == null) return@let
            if (gates.any { archKey(it) == archKey(architecture) }) return@let
            // The manifest's list is a floor, not a ceiling.
            //
            // Two lists knew which encoders an architecture reads, and they
            // disagreed. `DiffusionFamily` is derived from the branch of
            // sd.cpp's conditioner dispatch that actually runs; the manifest's
            // `appliesTo.arch` is typed by hand and stopped at FLUX.2. So every
            // architecture added upstream since — Qwen-Image, Z-Image, Anima,
            // Ovis, Ernie, Lens and the rest, all of which read their prompt
            // through a language model — had that slot refused, and the file
            // they cannot run without could not be attached.
            //
            // Asking the derived table first means a new architecture works the
            // day its family is known, and the manifest never has to be edited
            // in step with it.
            if (readsEncoder(spec.key, architecture)) return@let
            return "Only for ${gates.joinToString(", ")}. This model is $architecture."
        }
        return null
    }

    /** Whether this architecture conditions through the encoder [key] names. */
    private fun readsEncoder(key: String, architecture: String): Boolean {
        val family = ai.ondevice.core.DiffusionFamily.forName(architecture) ?: return false
        return key in family.encoders || key in family.optionalEncoders
    }

    /** A parameter for a newer build waits until the runtime catches up, then works on its own — no app update, no user action. */
    private fun buildRefusal(spec: ParamSpec, loadedBuildTag: String?): String? {
        if (loadedBuildTag == null) return null
        val current = buildOrdinal(loadedBuildTag) ?: return null
        spec.sinceBuild?.let { since ->
            buildOrdinal(since)?.let {
                if (current < it) return "Needs runtime build $since or newer; this one is $loadedBuildTag."
            }
        }
        spec.untilBuild?.let { until ->
            buildOrdinal(until)?.let {
                if (current >= it) return "Removed from the runtime in build $until; this one is $loadedBuildTag."
            }
        }
        return null
    }

    /** `b6482` → 6482. Non-`b` tags (whisper's `v1.7.6`) fall back to a version sort. */
    private fun buildOrdinal(tag: String): Long? {
        val digits = Regex("""\d+""").findAll(tag).map { it.value }.toList()
        if (digits.isEmpty()) return null
        return if (tag.startsWith("b")) {
            digits.first().toLongOrNull()
        } else {
            digits.take(3).fold(0L) { acc, part -> acc * 1000 + (part.toLongOrNull() ?: 0) }
        }
    }

    private fun dependencyRefusal(spec: ParamSpec, values: SparseParams): String? {
        val dep = spec.dependsOn ?: return null
        val label = labelFor(dep.key)
        val current: JsonElement = values[dep.key]
            ?: defaultFor(spec, dep.key)
            ?: return "Enabled once $label is set."
        dep.equals?.let {
            return if (sameValue(current, it)) null else "Enabled when $label is ${renderJson(it)}."
        }
        dep.notEquals?.let {
            val forbidden = renderJson(it)
            return if (!sameValue(current, it)) {
                null
            } else if (forbidden.isBlank()) {
                "Enabled once $label is set."
            } else {
                "Enabled when $label is anything but $forbidden."
            }
        }
        return null
    }

    /** The name a parameter is shown under, for a sentence about it. */
    private fun labelFor(key: String): String = cached
        ?.runtimes
        ?.values
        ?.firstNotNullOfOrNull { rt -> rt.params.firstOrNull { it.key == key }?.label }
        ?: key

    private fun defaultFor(spec: ParamSpec, key: String): JsonElement? = cached
        ?.runtimes
        ?.values
        ?.firstNotNullOfOrNull { rt -> rt.params.firstOrNull { it.key == key }?.default }

    private fun sameValue(a: JsonElement, b: JsonElement): Boolean =
        renderJson(a).equals(renderJson(b), ignoreCase = true)

    /** Group the visible set into the sections the parameter screen renders. */
    fun grouped(specs: List<ParamSpec>): List<ParamGroup> {
        val order = LinkedHashMap<String, MutableList<ParamSpec>>()
        specs.forEach { order.getOrPut(it.group) { mutableListOf() }.add(it) }
        return order.map { (name, items) -> ParamGroup(name, items) }
    }

    companion object {
        const val ASSET = "params-manifest.json"

        /**
         * One spelling, for two vocabularies that name the same thing.
         *
         * An architecture reaches this app from either of two places, and they
         * do not agree. A GGUF's `general.architecture` says `sd3`, and the
         * manifest's gates are written to match it. stable-diffusion.cpp,
         * which is the only thing that can name a checkpoint carrying no
         * metadata, prints its own label: `SDXL`, `Flux.2 klein`, `SD3.x`.
         * Both end up in the same column, and comparing them literally meant a
         * model the runtime had named lost every parameter gated on its
         * architecture — the IP-Adapter, the ControlNet, the CLIP-vision
         * encoder and the three text encoders all disappeared from a model
         * that takes all six, because the row said `SDXL` and the gate said
         * `sdxl`.
         *
         * Case and separators carry no meaning in either vocabulary, so both
         * sides are stripped to letters and digits. The trailing `x` goes with
         * them: `SD1.x`, `SD2.x` and `SD3.x` are how the runtime writes "any
         * point release of", and the family is what the gate is asking about.
         */
        internal fun archKey(name: String): String {
            val squashed = name.lowercase().filter { it.isLetterOrDigit() }
            val versionWildcard = squashed.length > 1 &&
                squashed.last() == 'x' &&
                squashed[squashed.lastIndex - 1].isDigit()
            return if (versionWildcard) squashed.dropLast(1) else squashed
        }

        /** Where a runtime-reported key with no manifest row is filed. */
        const val UNDESCRIBED_GROUP = "not described yet"
    }
}

/** One parameter as the screen should render it, and why it is not editable. */
data class GatedParam(
    val spec: ParamSpec,
    /** Null when it can be edited; otherwise what would have to change first. */
    val disabledBecause: String?,
)

data class VisibleParams(
    val rows: List<GatedParam>,
    val shownCount: Int,
    val disabledCount: Int,
    val totalCount: Int,
)

data class ParamGroup(val name: String, val specs: List<ParamSpec>)
