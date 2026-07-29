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

/**
 * Loads the manifest and answers "what should this screen render?".
 *
 * SPEC §16.5 — the app uses `max(bundled, downloaded)` by version. The bundled
 * copy in `assets/` guarantees the app works offline with no updates applied;
 * an OTA manifest can correct metadata, retier, relabel and *reveal* parameters
 * the installed `.so` already supports, but it cannot add capability the native
 * lib lacks. The `sinceBuild` gate makes that automatic and invisible.
 */
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
        // Only a signature-verified OTA manifest may win. A failed signature
        // means we keep what we have (§16.5) — it is never a reason to render
        // nothing.
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

    suspend fun storedManifest(): ParamManifestEntity? = db.manifests().newest()

    /**
     * The visible parameter set for a screen.
     *
     * Filtering happens in this order, and each step is a spec requirement:
     *  1. `appliesTo` — modality and architecture restrictions (§16.1).
     *  2. `sinceBuild` / `untilBuild` — gated against the *actually loaded*
     *     runtime build tag, queried from the native lib at init (§16.5).
     *  3. `dependsOn` — conditional visibility against current values.
     *  4. tier — visibility only, collapsed entirely by "show all" (§9).
     *  5. search.
     */
    fun visible(
        specs: List<ParamSpec>,
        values: SparseParams,
        tier: Tier?,
        showAll: Boolean,
        loadedBuildTag: String?,
        modality: String?,
        architecture: String?,
        query: String = "",
    ): VisibleParams {
        val applicable = specs.filter { spec ->
            appliesToMatches(spec, modality, architecture) && buildInRange(spec, loadedBuildTag)
        }
        val afterDependencies = applicable.filter { dependencySatisfied(it, values) }
        val hiddenByGate = applicable.size - afterDependencies.size +
            (specs.size - applicable.size)

        val afterTier = if (showAll || tier == null) {
            afterDependencies
        } else {
            afterDependencies.filter { it.tier.ordinal <= tier.ordinal }
        }

        val afterSearch = if (query.isBlank()) {
            afterTier
        } else {
            val q = query.trim().lowercase()
            afterTier.filter {
                it.key.contains(q, true) || it.label.contains(q, true) || it.help.contains(q, true)
            }
        }

        return VisibleParams(
            specs = afterSearch,
            shownCount = afterSearch.size,
            hiddenCount = hiddenByGate,
            totalCount = specs.size,
        )
    }

    private fun appliesToMatches(spec: ParamSpec, modality: String?, architecture: String?): Boolean {
        val applies = spec.appliesTo ?: return true
        applies.modality?.let { if (modality != null && modality !in it) return false }
        applies.arch?.let { if (architecture != null && architecture !in it) return false }
        return true
    }

    /**
     * A parameter for a newer build simply stays hidden until the runtime
     * catches up, then appears on its own — no app update, no user action.
     */
    private fun buildInRange(spec: ParamSpec, loadedBuildTag: String?): Boolean {
        if (loadedBuildTag == null) return true
        val current = buildOrdinal(loadedBuildTag) ?: return true
        spec.sinceBuild?.let { since -> buildOrdinal(since)?.let { if (current < it) return false } }
        spec.untilBuild?.let { until -> buildOrdinal(until)?.let { if (current >= it) return false } }
        return true
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

    private fun dependencySatisfied(spec: ParamSpec, values: SparseParams): Boolean {
        val dep = spec.dependsOn ?: return true
        val current: JsonElement = values[dep.key] ?: defaultFor(spec, dep.key) ?: return false
        dep.equals?.let { return sameValue(current, it) }
        dep.notEquals?.let { return !sameValue(current, it) }
        return true
    }

    private fun defaultFor(spec: ParamSpec, key: String): JsonElement? = cached
        ?.runtimes
        ?.values
        ?.firstNotNullOfOrNull { rt -> rt.params.firstOrNull { it.key == key }?.default }

    private fun sameValue(a: JsonElement, b: JsonElement): Boolean =
        renderJson(a).equals(renderJson(b), ignoreCase = true)

    /**
     * Group the visible set into the sections the parameter screen renders.
     * Order follows the manifest's own group ordering so a new group lands in a
     * sensible place without a code change.
     */
    fun grouped(specs: List<ParamSpec>): List<ParamGroup> {
        val order = LinkedHashMap<String, MutableList<ParamSpec>>()
        specs.forEach { order.getOrPut(it.group) { mutableListOf() }.add(it) }
        return order.map { (name, items) -> ParamGroup(name, items) }
    }

    companion object {
        const val ASSET = "params-manifest.json"
    }
}

data class VisibleParams(
    val specs: List<ParamSpec>,
    val shownCount: Int,
    val hiddenCount: Int,
    val totalCount: Int,
)

data class ParamGroup(val name: String, val specs: List<ParamSpec>)
