package ai.ondevice.data

import ai.ondevice.core.BackendId
import ai.ondevice.core.MessageRole
import ai.ondevice.data.db.ConversationEntity
import ai.ondevice.data.db.MessageEntity
import ai.ondevice.data.db.OnDeviceDatabase
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import java.io.File
import java.io.InputStream
import java.util.UUID
import java.util.zip.ZipEntry
import java.util.zip.ZipInputStream
import java.util.zip.ZipOutputStream

/**
 * Conversations in and out of the app, whole.
 *
 * SPEC §13's position is that nothing lives in a private opaque store, and a
 * conversation you cannot get out of the app is exactly that. Two formats,
 * because they answer different questions:
 *
 *  - **`.zip`** — lossless. Every message with its full generation parameter
 *    set, its measured tok/s, its backend, its reasoning block and its
 *    attachments as real files. This round-trips: what you export is what you
 *    get back, including the numbers that make a reply reproducible.
 *  - **`.md`** — readable. What you paste into a bug report.
 *
 * The import side is deliberately forgiving about *unknown* fields and strict
 * about *missing* ones. An archive written by a later build carries keys this
 * one has never seen; §11 says keep them, so the parameter blobs come across as
 * opaque strings rather than being parsed into a schema that would drop them.
 */
class ConversationArchive(
    private val db: OnDeviceDatabase,
    private val storage: ModelStorage,
) {

    private val json = Json {
        prettyPrint = true
        ignoreUnknownKeys = true
        encodeDefaults = true
    }

    // — export —

    /**
     * Write a lossless archive. [conversationIds] empty means the whole library.
     */
    suspend fun exportArchive(
        conversationIds: List<String> = emptyList(),
        destination: File,
    ): File = withContext(Dispatchers.IO) {
        val conversations = db.conversations().getAll()
            .filter { conversationIds.isEmpty() || it.id in conversationIds }

        val document = ArchiveDocument(
            formatVersion = FORMAT_VERSION,
            exportedAt = System.currentTimeMillis(),
            application = "ai.ondevice",
            conversations = conversations.map { conversation ->
                ArchivedConversation(
                    id = conversation.id,
                    title = conversation.title,
                    modelId = conversation.modelId,
                    personaId = conversation.personaId,
                    systemPrompt = conversation.systemPrompt,
                    presetId = conversation.presetId,
                    createdAt = conversation.createdAt,
                    updatedAt = conversation.updatedAt,
                    messages = db.messages().getFor(conversation.id).map { it.toArchived() },
                )
            },
            personas = db.personas().getAll().map {
                ArchivedPersona(it.id, it.name, it.systemPrompt, it.defaultModelId, it.defaultVoice, it.memoryNotes)
            },
            presets = db.presets().getAll().map {
                ArchivedPreset(it.id, it.modality.name, it.name, it.paramsJson, it.isBuiltIn)
            },
        )

        destination.parentFile?.mkdirs()
        ZipOutputStream(destination.outputStream().buffered()).use { zip ->
            zip.putNextEntry(ZipEntry(MANIFEST_ENTRY))
            zip.write(json.encodeToString(ArchiveDocument.serializer(), document).toByteArray())
            zip.closeEntry()

            // Attachments travel with the archive. A reference to a path on a
            // handset you no longer own is not an export.
            document.conversations
                .flatMap { it.messages }
                .flatMap { it.attachments }
                .distinctBy { it.path }
                .forEach { attachment ->
                    val file = File(attachment.path)
                    if (!file.exists()) return@forEach
                    zip.putNextEntry(ZipEntry("$ATTACHMENT_DIR/${attachment.archiveName}"))
                    file.inputStream().use { it.copyTo(zip) }
                    zip.closeEntry()
                }
        }
        destination
    }

    /** The readable form. One conversation, as Markdown. */
    suspend fun exportMarkdown(conversationId: String, destination: File): File =
        withContext(Dispatchers.IO) {
            val conversation = db.conversations().get(conversationId)
                ?: error("No conversation $conversationId")
            val messages = db.messages().getFor(conversationId)

            destination.parentFile?.mkdirs()
            destination.writeText(
                buildString {
                    appendLine("# ${conversation.title}")
                    appendLine()
                    conversation.modelId?.let { appendLine("*Model:* `$it`") }
                    appendLine("*Exported:* ${java.util.Date()}")
                    appendLine()
                    conversation.systemPrompt?.takeIf { it.isNotBlank() }?.let {
                        appendLine("> **System**")
                        it.lines().forEach { line -> appendLine("> $line") }
                        appendLine()
                    }
                    messages.forEach { message ->
                        appendLine("## ${message.role.name.lowercase().replaceFirstChar(Char::uppercase)}")
                        appendLine()
                        // The reasoning block is folded, not dropped — it is
                        // often the interesting part of a local model's reply.
                        message.thinking?.takeIf { it.isNotBlank() }?.let { thinking ->
                            appendLine("<details><summary>Reasoning" +
                                (message.thinkingTokens?.let { " · $it tokens" } ?: "") +
                                "</summary>")
                            appendLine()
                            appendLine(thinking)
                            appendLine()
                            appendLine("</details>")
                            appendLine()
                        }
                        appendLine(message.content)
                        appendLine()
                        val meta = buildList {
                            message.tokensPerSecond?.let { add(String.format("%.1f tok/s", it)) }
                            message.backend?.let { add(it.name.lowercase()) }
                            message.tokenCount?.let { add("$it tokens") }
                        }
                        if (meta.isNotEmpty()) {
                            appendLine("<sub>${meta.joinToString(" · ")}</sub>")
                            appendLine()
                        }
                    }
                },
            )
            destination
        }

    // — import —

    /**
     * Read an archive back. Conversations keep their identity where they can:
     * an id that already exists is imported under a fresh one rather than
     * overwriting what is already here, because a silent overwrite of a
     * conversation is not recoverable.
     */
    suspend fun importArchive(input: InputStream): ImportReport = withContext(Dispatchers.IO) {
        var document: ArchiveDocument? = null
        val attachments = mutableMapOf<String, File>()

        ZipInputStream(input.buffered()).use { zip ->
            while (true) {
                val entry = zip.nextEntry ?: break
                when {
                    entry.name == MANIFEST_ENTRY -> {
                        document = json.decodeFromString(
                            ArchiveDocument.serializer(),
                            zip.readBytes().decodeToString(),
                        )
                    }
                    entry.name.startsWith("$ATTACHMENT_DIR/") && !entry.isDirectory -> {
                        val name = entry.name.substringAfterLast('/')
                        val target = File(storage.attachmentsDir(), name)
                        target.outputStream().use { zip.copyTo(it) }
                        attachments[name] = target
                    }
                }
                zip.closeEntry()
            }
        }

        val archive = document
            ?: return@withContext ImportReport(
                error = "That file has no $MANIFEST_ENTRY — it is not a conversation archive.",
            )
        if (archive.formatVersion > FORMAT_VERSION) {
            return@withContext ImportReport(
                error = "The archive is format v${archive.formatVersion} and this build reads " +
                    "v$FORMAT_VERSION. Update the app rather than importing it partially.",
            )
        }

        var conversationCount = 0
        var messageCount = 0

        archive.personas.forEach { persona ->
            if (db.personas().get(persona.id) == null) {
                db.personas().upsert(
                    ai.ondevice.data.db.PersonaEntity(
                        id = persona.id,
                        name = persona.name,
                        avatarPath = null,
                        systemPrompt = persona.systemPrompt,
                        defaultModelId = persona.defaultModelId,
                        defaultPresetId = null,
                        defaultVoice = persona.defaultVoice,
                        memoryNotes = persona.memoryNotes,
                    ),
                )
            }
        }

        archive.presets.forEach { preset ->
            if (db.presets().get(preset.id) == null) {
                db.presets().upsert(
                    ai.ondevice.data.db.PresetEntity(
                        id = preset.id,
                        modality = runCatching { ai.ondevice.core.Modality.valueOf(preset.modality) }
                            .getOrDefault(ai.ondevice.core.Modality.TEXT),
                        name = preset.name,
                        paramsJson = preset.paramsJson,
                        isBuiltIn = false,
                    ),
                )
            }
        }

        archive.conversations.forEach { archived ->
            val collides = db.conversations().get(archived.id) != null
            val conversationId = if (collides) UUID.randomUUID().toString() else archived.id

            db.conversations().upsert(
                ConversationEntity(
                    id = conversationId,
                    title = if (collides) "${archived.title} (imported)" else archived.title,
                    modelId = archived.modelId,
                    personaId = archived.personaId,
                    systemPrompt = archived.systemPrompt,
                    presetId = archived.presetId,
                    createdAt = archived.createdAt,
                    updatedAt = archived.updatedAt,
                ),
            )
            conversationCount++

            archived.messages.forEach { message ->
                val paths = message.attachments.mapNotNull { attachments[it.archiveName]?.absolutePath }
                db.messages().upsert(
                    MessageEntity(
                        id = if (collides) UUID.randomUUID().toString() else message.id,
                        conversationId = conversationId,
                        role = runCatching { MessageRole.valueOf(message.role) }
                            .getOrDefault(MessageRole.USER),
                        content = message.content,
                        thinking = message.thinking,
                        thinkingMillis = message.thinkingMillis,
                        thinkingTokens = message.thinkingTokens,
                        imagePathsJson = ai.ondevice.core.SparseParams
                            .of("images" to paths).toJsonString(),
                        toolCallsJson = message.toolCallsJson,
                        tokenCount = message.tokenCount,
                        imageTokenCount = message.imageTokenCount,
                        // Kept as an opaque string: an archive written under a
                        // newer engine carries keys this build has never seen,
                        // and §11 says an unknown key survives.
                        generationParamsJson = message.generationParamsJson,
                        tokensPerSecond = message.tokensPerSecond,
                        backend = message.backend?.let {
                            runCatching { BackendId.valueOf(it) }.getOrNull()
                        },
                        createdAt = message.createdAt,
                        parentMessageId = null,
                    ),
                )
                messageCount++
            }
        }

        ImportReport(
            conversations = conversationCount,
            messages = messageCount,
            attachments = attachments.size,
        )
    }

    private fun MessageEntity.toArchived(): ArchivedMessage {
        val paths = ai.ondevice.core.SparseParams.parse(imagePathsJson)
            .stringList("images").orEmpty()
        return ArchivedMessage(
            id = id,
            role = role.name,
            content = content,
            thinking = thinking,
            thinkingMillis = thinkingMillis,
            thinkingTokens = thinkingTokens,
            toolCallsJson = toolCallsJson,
            tokenCount = tokenCount,
            imageTokenCount = imageTokenCount,
            generationParamsJson = generationParamsJson,
            tokensPerSecond = tokensPerSecond,
            backend = backend?.name,
            createdAt = createdAt,
            attachments = paths.map { path ->
                ArchivedAttachment(
                    path = path,
                    // Prefixed with the message id so two conversations that
                    // both attached "photo.jpg" do not collide in the zip.
                    archiveName = "${id.take(8)}-${File(path).name}",
                )
            },
        )
    }

    companion object {
        const val FORMAT_VERSION = 1
        const val MANIFEST_ENTRY = "conversations.json"
        const val ATTACHMENT_DIR = "attachments"
        const val MIME = "application/zip"
    }
}

data class ImportReport(
    val conversations: Int = 0,
    val messages: Int = 0,
    val attachments: Int = 0,
    val error: String? = null,
) {
    val ok: Boolean get() = error == null
}

// — the on-disk shape —

@Serializable
data class ArchiveDocument(
    @SerialName("format_version") val formatVersion: Int,
    @SerialName("exported_at") val exportedAt: Long,
    val application: String,
    val conversations: List<ArchivedConversation> = emptyList(),
    val personas: List<ArchivedPersona> = emptyList(),
    val presets: List<ArchivedPreset> = emptyList(),
)

@Serializable
data class ArchivedConversation(
    val id: String,
    val title: String,
    val modelId: String? = null,
    val personaId: String? = null,
    val systemPrompt: String? = null,
    val presetId: String? = null,
    val createdAt: Long,
    val updatedAt: Long,
    val messages: List<ArchivedMessage> = emptyList(),
)

@Serializable
data class ArchivedMessage(
    val id: String,
    val role: String,
    val content: String,
    val thinking: String? = null,
    val thinkingMillis: Long? = null,
    val thinkingTokens: Int? = null,
    val toolCallsJson: String? = null,
    val tokenCount: Int? = null,
    val imageTokenCount: Int? = null,
    val generationParamsJson: String = "{}",
    val tokensPerSecond: Float? = null,
    val backend: String? = null,
    val createdAt: Long,
    val attachments: List<ArchivedAttachment> = emptyList(),
)

@Serializable
data class ArchivedAttachment(
    val path: String,
    @SerialName("archive_name") val archiveName: String,
)

@Serializable
data class ArchivedPersona(
    val id: String,
    val name: String,
    val systemPrompt: String,
    val defaultModelId: String? = null,
    val defaultVoice: String? = null,
    val memoryNotes: String? = null,
)

@Serializable
data class ArchivedPreset(
    val id: String,
    val modality: String,
    val name: String,
    val paramsJson: String,
    val isBuiltIn: Boolean = false,
)
