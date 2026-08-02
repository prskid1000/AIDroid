package ai.ondevice.core

/**
 * Names for files, short enough to read and long enough to tell apart.
 *
 * A filename is the obvious label and is not always a distinguishing one. The
 * diffusers layout calls every component's weights the same thing —
 * `vae/diffusion_pytorch_model.safetensors`,
 * `controlnet/diffusion_pytorch_model.safetensors` — so a list of files
 * labelled by filename shows the same row twice and asks you to pick one.
 *
 * Rather than keep a list of filenames considered too generic, which would be
 * wrong the moment a repo invents another, this asks the only question that
 * matters: is this label unique *here*? A name grows one folder at a time until
 * it is, so it is as short as it can be and no shorter, and a list of files
 * that were never ambiguous is left exactly as it was.
 */
object FileLabels {

    /**
     * A label per path, each the shortest trailing run of segments that no
     * other path in [paths] shares.
     *
     * Paths that are genuinely identical keep the full name — there is nothing
     * left to distinguish, and inventing a difference would be a lie about the
     * files rather than a fact about them.
     */
    fun distinguish(paths: List<String>): Map<String, String> {
        val depths = paths.associateWith { 1 }.toMutableMap()

        // Grow every label that collides, then look again: growing one may
        // resolve a collision or expose a new one.
        var growing = true
        while (growing) {
            growing = false
            val byLabel = paths.groupBy { tail(it, depths.getValue(it)) }
            byLabel.values.filter { it.size > 1 }.forEach { clashing ->
                clashing.forEach { path ->
                    val depth = depths.getValue(path)
                    if (depth < segments(path).size) {
                        depths[path] = depth + 1
                        growing = true
                    }
                }
            }
        }
        return paths.associateWith { tail(it, depths.getValue(it)) }
    }

    /** One path against a fixed set of others, for callers holding one at a time. */
    fun distinguish(path: String, among: List<String>): String =
        distinguish((listOf(path) + among).distinct())[path] ?: path.substringAfterLast('/')

    private fun segments(path: String) = path.split('/', '\\').filter { it.isNotEmpty() }

    private fun tail(path: String, depth: Int): String =
        segments(path).takeLast(depth).joinToString("/")
}

/**
 * Names for things chosen from a list, qualified only as far as they need to be.
 *
 * Several rows can share a display name honestly: a repo that ships CLIP-L,
 * CLIP-G and a T5 gives all three the repo's name, and the app stores them as
 * three models. Left alone, the list shows one name three times — and where the
 * control hands back the label rather than the row, as a dropdown does, picking
 * the second gives you the first.
 */
object Labels {

    /**
     * A name, the details that could tell it from a namesake, and the ones
     * that belong to it whether or not anything else is called the same.
     *
     * @param always shown even when the name stands alone. For a repo that
     *   holds many models rather than one — `ggerganov/whisper.cpp` is every
     *   whisper size — the name is the repo and the variant is the identity,
     *   so a unique label reading "whisper.cpp" names the runtime and not the
     *   thing being picked.
     */
    data class Item(
        val name: String,
        val qualifiers: List<String?>,
        val always: List<String?> = emptyList(),
    )

    /**
     * One label per item, in order, unique wherever the inputs allow it.
     *
     * A name that already stands alone is returned untouched, so the common
     * case reads as it always did. A name that clashes takes on its first
     * qualifier, then its second, and so on only while it is still ambiguous.
     */
    fun unique(items: List<Item>): List<String> {
        val depth = IntArray(items.size)

        var growing = true
        while (growing) {
            growing = false
            val labels = items.indices.map { label(items[it], depth[it]) }
            labels.withIndex()
                .groupBy({ it.value }, { it.index })
                .values
                .filter { it.size > 1 }
                .forEach { clashing ->
                    clashing.forEach { i ->
                        val available = items[i].qualifiers.count { !it.isNullOrBlank() }
                        if (depth[i] < available) {
                            depth[i]++
                            growing = true
                        }
                    }
                }
        }
        return items.indices.map { label(items[it], depth[it]) }
    }

    private fun label(item: Item, depth: Int): String {
        val always = item.always.filterNot { it.isNullOrBlank() }
        // Anything in `always` is part of the name for collision purposes too:
        // two rows differing only by something already on screen are not
        // ambiguous, and adding a further qualifier to separate them would be
        // adding one nobody needed.
        val taken = item.qualifiers
            .filterNot { it.isNullOrBlank() || it in always }
            .take(depth)
        return (listOf(item.name) + always + taken).joinToString(" · ")
    }
}
