package ai.ondevice.core

import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put

/**
 * What was attached to a message, beside what was typed.
 *
 * Images have always lived in `messages.imagePathsJson` as `{"images":[…]}`.
 * Documents used to live nowhere: their text was pasted into the message
 * content, which meant a 300-line file was a 300-line chat bubble, and the
 * only way to see the question was to scroll past the answer to it.
 *
 * They are stored beside the images now, in the same column — a sparse JSON
 * object takes a second key without a migration, and these two things are the
 * same thing at different resolutions.
 */
data class MessageAttachments(
    val images: List<String> = emptyList(),
    val documents: List<Document> = emptyList(),
) {
    data class Document(val name: String, val path: String, val text: String)

    val isEmpty: Boolean get() = images.isEmpty() && documents.isEmpty()

    /** The message as the model sees it: every document, then what was typed. */
    fun promptText(typed: String): String {
        if (documents.isEmpty()) return typed
        return buildString {
            documents.forEach { document ->
                appendLine("--- ${document.name} ---")
                appendLine(document.text)
                appendLine("--- end of ${document.name} ---")
                appendLine()
            }
            append(typed)
        }.trim()
    }

    fun toJsonString(): String = buildJsonObject {
        put("images", buildJsonArray { images.forEach { add(JsonPrimitive(it)) } })
        put(
            "documents",
            buildJsonArray {
                documents.forEach { document ->
                    add(
                        buildJsonObject {
                            put("name", document.name)
                            put("path", document.path)
                            put("text", document.text)
                        },
                    )
                }
            },
        )
    }.toString()

    companion object {
        fun of(json: String?): MessageAttachments {
            val params = SparseParams.parse(json)
            return MessageAttachments(
                images = params.stringList("images").orEmpty(),
                documents = (params["documents"] as? JsonArray).orEmpty().mapNotNull { element ->
                    val row = element as? JsonObject ?: return@mapNotNull null
                    Document(
                        name = row.text("name") ?: return@mapNotNull null,
                        path = row.text("path").orEmpty(),
                        text = row.text("text").orEmpty(),
                    )
                },
            )
        }

        private fun JsonArray?.orEmpty(): List<kotlinx.serialization.json.JsonElement> =
            this ?: emptyList()

        private fun JsonObject.text(key: String): String? =
            (this[key] as? JsonPrimitive)?.takeIf { it.isString }?.content
    }
}
