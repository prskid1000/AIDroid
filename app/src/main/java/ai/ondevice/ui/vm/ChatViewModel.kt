package ai.ondevice.ui.vm

import android.content.Context
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import ai.ondevice.core.MessageRole
import ai.ondevice.core.Modality
import ai.ondevice.core.SparseParams
import ai.ondevice.data.db.ConversationEntity
import ai.ondevice.data.db.MessageEntity
import ai.ondevice.data.db.ModelEntity
import ai.ondevice.data.db.OnDeviceDatabase
import ai.ondevice.data.prefs.AppPrefs
import ai.ondevice.engine.EngineMessage
import ai.ondevice.engine.EngineManager
import ai.ondevice.engine.GenerateRequest
import ai.ondevice.engine.GenerationEvent
import ai.ondevice.engine.InferenceService
import ai.ondevice.engine.RenderedPrompt
import ai.ondevice.engine.record
import dagger.hilt.android.lifecycle.HiltViewModel
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.Job
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.withContext
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import java.util.UUID
import javax.inject.Inject

/** S6/S7/S10 — the chat loop. */
@HiltViewModel
class ChatViewModel @Inject constructor(
    @ApplicationContext private val context: Context,
    private val db: OnDeviceDatabase,
    private val engines: EngineManager,
    private val prefs: AppPrefs,
    private val toolProviders: ai.ondevice.tools.ToolProviderFactory,
    private val attachments: ai.ondevice.data.AttachmentStore,
    private val archive: ai.ondevice.data.ConversationArchive,
    private val storage: ai.ondevice.data.ModelStorage,
    private val recorder: ai.ondevice.engine.ResourceRecorder,
) : ViewModel() {

    private val _state = MutableStateFlow(ChatState())
    val state: StateFlow<ChatState> = _state.asStateFlow()

    private var generationJob: Job? = null

    init {
        viewModelScope.launch { restore() }

        // The model list has to be live.
        viewModelScope.launch {
            db.models().observeInstalled().collect { models ->
                _state.value = _state.value.copy(
                    availableModels = models.filter {
                        it.modality == Modality.TEXT || it.modality == Modality.VISION
                    },
                    // Re-read the selected row too, not just the list.
                    model = _state.value.model?.let { current ->
                        models.firstOrNull { it.id == current.id } ?: current
                    },
                )
            }
        }

        // A chat model still arriving is not a chat model missing, and the
        // empty screen told you to add one you were already adding.
        viewModelScope.launch {
            db.models().observeInstalling().collect { jobs ->
                _state.value = _state.value.copy(
                    installing = jobs.filter {
                        it.modality == Modality.TEXT || it.modality == Modality.VISION
                    },
                )
            }
        }

        // Likewise the engine's own view of what is loaded, so the sheet says
        // "loaded" only when something actually is.
        viewModelScope.launch {
            engines.state.collect { engine ->
                _state.value = _state.value.copy(
                    loadedModelId = engine.loaded?.modelId,
                    loadingModel = engine.loading,
                    chatTemplate = engine.loaded?.chatTemplate,
                    templateSource = engine.loaded?.templateSource ?: "gguf.chat_template",
                    loadedTemplateKwargsJson = engine.loaded?.templateKwargsJson ?: "{}",
                )
            }
        }
    }

    private suspend fun restore() {
        val conversation = prefs.lastConversationId.first()?.let { db.conversations().get(it) }
            ?: db.conversations().mostRecent()
            ?: newConversation()

        // A vision model is a text model that can also see.
        val chatModels = db.models().getInstalled().filter {
            it.modality == Modality.TEXT || it.modality == Modality.VISION
        }
        val model = conversation.modelId?.let { db.models().get(it) }
            ?: chatModels.firstOrNull()

        _state.value = _state.value.copy(
            conversation = conversation,
            model = model,
            messages = db.messages().getFor(conversation.id),
            systemPrompt = conversation.systemPrompt ?: "",
            availableModels = chatModels,
        )
    }

    private suspend fun newConversation(): ConversationEntity {
        val conversation = ConversationEntity(
            id = UUID.randomUUID().toString(),
            title = "New conversation",
            modelId = null,
            personaId = null,
            systemPrompt = null,
            presetId = null,
            createdAt = System.currentTimeMillis(),
            updatedAt = System.currentTimeMillis(),
        )
        db.conversations().upsert(conversation)
        prefs.setLastConversationId(conversation.id)
        return conversation
    }

    /** Start a fresh conversation. */
    fun startNewConversation() {
        if (_state.value.generating) stop()
        // Already on an empty one.
        if (_state.value.messages.isEmpty() && _state.value.conversation != null) return
        viewModelScope.launch {
            (engines.llama as? ai.ondevice.engine.LlamaEngine)?.clearCache()
            val conversation = newConversation()
            _state.value = _state.value.copy(
                conversation = conversation,
                messages = emptyList(),
                streaming = null,
                input = "",
                pendingAttachments = emptyList(),
                contextUsed = 0,
                cachedTokens = 0,
                error = null,
                errorSuggestion = null,
                lastExport = null,
                importSummary = null,
            )
        }
    }

    /** Switch to a thread the user picked out of the library. */
    fun openConversation(id: String) {
        if (_state.value.conversation?.id == id) return
        if (_state.value.generating) stop()
        viewModelScope.launch {
            val conversation = db.conversations().get(id) ?: return@launch
            (engines.llama as? ai.ondevice.engine.LlamaEngine)?.clearCache()
            prefs.setLastConversationId(conversation.id)
            _state.value = _state.value.copy(
                conversation = conversation,
                streaming = null,
                input = "",
                pendingAttachments = emptyList(),
                contextUsed = 0,
                cachedTokens = 0,
                error = null,
                errorSuggestion = null,
                lastExport = null,
                importSummary = null,
                systemPrompt = conversation.systemPrompt ?: "",
                model = conversation.modelId?.let { db.models().get(it) } ?: _state.value.model,
                messages = db.messages().getFor(conversation.id),
            )
        }
    }

    fun deleteConversation(id: String) {
        viewModelScope.launch {
            db.messages().clearFor(id)
            db.conversations().deleteById(id)
            if (_state.value.conversation?.id == id) {
                (engines.llama as? ai.ondevice.engine.LlamaEngine)?.clearCache()
                val next = db.conversations().mostRecent() ?: newConversation()
                prefs.setLastConversationId(next.id)
                _state.value = _state.value.copy(
                    conversation = next,
                    messages = db.messages().getFor(next.id),
                    streaming = null,
                )
            }
        }
    }

    fun onInputChange(value: String) {
        _state.value = _state.value.copy(input = value)
    }

    /** Attach a file the user picked, and decide what it *is* before deciding what to do with it. */
    fun attach(uri: android.net.Uri) {
        viewModelScope.launch {
            val attachment = attachments.copyIn(uri) ?: run {
                _state.value = _state.value.copy(
                    error = "That file could not be read.",
                    errorSuggestion = "Pick it from a different app, or copy it into local storage first.",
                )
                return@launch
            }

            val pending = when (attachment.kind) {
                ai.ondevice.data.AttachmentKind.IMAGE -> {
                    val model = _state.value.model
                    // Two separate ways this fails, and they used to be one check.
                    val missing = if (model?.modality != Modality.VISION) {
                        ai.ondevice.core.MissingComponent(
                            what = "${model?.label ?: "This model"} cannot see images",
                            because = "it is a text model, and images reach one only through a " +
                                "separate vision projector",
                            state = ai.ondevice.core.MissingComponent.State.NOT_INSTALLED,
                        )
                    } else {
                        ai.ondevice.core.ComponentCheck.forChatImage(
                            ai.ondevice.core.SparseParams.parse(model.companionPathsJson).keys
                                .associateWith { "" },
                        )
                    }
                    if (missing != null) {
                        // Attaching it anyway and letting the model ignore it
                        // silently would be the worst of both worlds.
                        _state.value = _state.value.copy(
                            error = "${missing.what} — ${missing.because}.",
                            errorSuggestion = "Add a model with a projector companion, " +
                                "or describe the image in text.",
                        )
                        return@launch
                    }
                    PendingAttachment(
                        path = attachment.path,
                        name = attachment.displayName,
                        kind = attachment.kind,
                        tokenCost = IMAGE_TOKEN_COST,
                    )
                }

                ai.ondevice.data.AttachmentKind.DOCUMENT -> {
                    val extraction = attachments.extractText(attachment)
                    if (extraction.text.isBlank()) {
                        _state.value = _state.value.copy(
                            error = extraction.error ?: "Nothing readable in ${attachment.displayName}.",
                            errorSuggestion = null,
                        )
                        return@launch
                    }
                    PendingAttachment(
                        path = attachment.path,
                        name = attachment.displayName,
                        kind = attachment.kind,
                        text = extraction.text,
                        // Counted by the real tokenizer when a model is loaded,
                        // so the figure above the composer is the true cost.
                        tokenCost = engines.llama?.tokenCount(extraction.text)
                            ?: (extraction.text.length / 4),
                        note = extraction.error,
                    )
                }

                // Not offered by the picker — see `pickAttachment` — but a file manager can return anything, so the refusal stays.
                ai.ondevice.data.AttachmentKind.AUDIO -> {
                    _state.value = _state.value.copy(
                        error = "Audio has to be transcribed before it can enter a prompt.",
                        errorSuggestion = "Voice → File transcribes it, and the transcript can be " +
                            "pasted or attached here.",
                    )
                    return@launch
                }
            }

            _state.value = _state.value.copy(
                pendingAttachments = _state.value.pendingAttachments + pending,
                error = null,
            )
        }
    }

    fun removeAttachment(path: String) {
        _state.value = _state.value.copy(
            pendingAttachments = _state.value.pendingAttachments.filterNot { it.path == path },
        )
    }

    /**
     * The parameter set actually in force: the model's own overrides, then
     * whatever the sheet has changed this session. There used to be a preset
     * layer under both, applied invisibly after its picker was removed.
     */
    private fun effectiveParams(): SparseParams =
        SparseParams.parse(_state.value.model?.paramOverridesJson)
            .overlaidWith(_state.value.liveOverrides)

    fun send() {
        val typed = _state.value.input.trim()
        val conversation = _state.value.conversation ?: return
        val model = _state.value.model ?: run {
            val arriving = _state.value.installing.firstOrNull()
            _state.value = _state.value.copy(
                error = if (arriving != null) {
                    "${arriving.displayName} is still downloading — ${(arriving.fraction * 100).toInt()}%."
                } else {
                    "Pick a model first — Models → Add."
                },
            )
            return
        }
        val pending = _state.value.pendingAttachments
        if (typed.isEmpty() && pending.isEmpty()) return

        // What was typed and what was attached are stored apart. The model is
        // handed both (toEngineMessage puts them back together); the bubble
        // shows only the first.
        val attachments = ai.ondevice.core.MessageAttachments(
            images = pending.filter { it.kind == ai.ondevice.data.AttachmentKind.IMAGE }.map { it.path },
            documents = pending
                .filter { it.kind == ai.ondevice.data.AttachmentKind.DOCUMENT }
                .map {
                    ai.ondevice.core.MessageAttachments.Document(
                        name = it.name,
                        path = it.path,
                        text = it.text,
                    )
                },
        )
        val images = attachments.images

        _state.value = _state.value.copy(input = "", pendingAttachments = emptyList(), error = null)

        generationJob = viewModelScope.launch {
            val params = effectiveParams()

            val userMessage = MessageEntity(
                id = UUID.randomUUID().toString(),
                conversationId = conversation.id,
                role = MessageRole.USER,
                content = typed,
                thinking = null,
                thinkingMillis = null,
                thinkingTokens = null,
                imagePathsJson = attachments.toJsonString(),
                toolCallsJson = null,
                tokenCount = null,
                imageTokenCount = images.size * IMAGE_TOKEN_COST,
                generationParamsJson = params.toJsonString(),
                tokensPerSecond = null,
                createdAt = System.currentTimeMillis(),
                parentMessageId = null,
            )
            db.messages().upsert(userMessage)
            refreshMessages()

            if (!engines.state.value.loaded.let { it?.modelId == model.id }) {
                _state.value = _state.value.copy(loadingModel = true)
                val loadResult = engines.load(model)
                _state.value = _state.value.copy(loadingModel = false)
                if (loadResult.isFailure) {
                    _state.value = _state.value.copy(
                        error = engines.state.value.error?.message ?: "Model load failed.",
                        errorSuggestion = engines.state.value.error?.suggestion,
                    )
                    return@launch
                }
            }

            val engine = engines.llama ?: return@launch
            engine.applyParams(params)

            InferenceService.holdWakeLock(context)
            try {
                runTurn(conversation, params, images, parentId = userMessage.id)
            } finally {
                InferenceService.releaseWakeLock(context)
                withContext(NonCancellable) {
                    db.conversations().touch(conversation.id, System.currentTimeMillis())
                    _state.value = _state.value.copy(generating = false, streaming = null)
                    refreshMessages()
                }
            }
        }
    }

    /** One assistant turn, and the tool loop around it. */
    private suspend fun runTurn(
        conversation: ConversationEntity,
        params: SparseParams,
        images: List<String>,
        parentId: String?,
    ) {
        val engine = engines.llama ?: return
        val registry = if (prefs.toolsEnabled.first()) {
            toolProviders.registry(
                builtInEnabled = ai.ondevice.tools.BuiltInToolProvider.ID in
                    prefs.enabledToolProviders.first(),
            )
        } else {
            null
        }
        val tools = registry?.specs().orEmpty()

        var round = 0
        var lastParent = parentId

        while (true) {
            val assistantId = UUID.randomUUID().toString()
            val content = StringBuilder()
            val thinking = StringBuilder()
            var thinkingMillis: Long? = null
            var thinkingTokens: Int? = null
            var tps = 0f
            var promptTokens = 0
            val toolCalls = mutableListOf<ai.ondevice.engine.ToolCallRequest>()

            _state.value = _state.value.copy(
                generating = true,
                streaming = StreamingMessage(id = assistantId),
            )

            // Per round, not per turn.
            val recording = recorder.start(viewModelScope)
            val liveJob = viewModelScope.launch {
                recording.live.collect { trace ->
                    _state.value = _state.value.copy(liveTrace = trace)
                }
            }
            val startedAt = System.currentTimeMillis()

            try {
                engine.generate(
                    GenerateRequest(
                        messages = db.messages().getFor(conversation.id).map { it.toEngineMessage() },
                        params = params,
                        systemPrompt = _state.value.systemPrompt.takeIf { it.isNotBlank() },
                        imagePaths = images,
                        tools = tools,
                    ),
                ).collect { event ->
                    when (event) {
                        is GenerationEvent.PromptProcessed -> {
                            promptTokens = event.promptTokens
                            _state.value = _state.value.copy(
                                contextUsed = event.promptTokens,
                                cachedTokens = event.cachedTokens,
                                streaming = _state.value.streaming?.copy(
                                    promptTokens = event.promptTokens,
                                ),
                            )
                        }
                        is GenerationEvent.ThinkingDelta -> {
                            thinking.append(event.text)
                            _state.value = _state.value.copy(
                                streaming = _state.value.streaming?.copy(thinking = thinking.toString()),
                            )
                        }
                        is GenerationEvent.ThinkingDone -> {
                            thinkingMillis = event.elapsedMillis
                            thinkingTokens = event.totalTokens
                            _state.value = _state.value.copy(
                                streaming = _state.value.streaming?.copy(
                                    thinkingMillis = event.elapsedMillis,
                                    thinkingTokens = event.totalTokens,
                                    thinkingComplete = true,
                                ),
                            )
                        }
                        is GenerationEvent.Token -> {
                            content.append(event.text)
                            _state.value = _state.value.copy(
                                streaming = _state.value.streaming?.copy(content = content.toString()),
                            )
                        }
                        is GenerationEvent.ToolCall -> {
                            toolCalls += ai.ondevice.engine.ToolCallRequest(
                                name = event.name,
                                argumentsJson = event.argumentsJson,
                                id = event.id.ifBlank { UUID.randomUUID().toString().take(8) },
                            )
                        }
                        is GenerationEvent.Stats -> {
                            tps = event.tokensPerSecond
                            _state.value = _state.value.copy(
                                tokensPerSecond = event.tokensPerSecond,
                                contextUsed = event.contextUsed,
                            )
                        }
                        is GenerationEvent.Done -> Unit
                        is GenerationEvent.Failed -> _state.value = _state.value.copy(
                            error = event.message,
                            errorSuggestion = event.suggestion,
                        )
                    }
                }
            } finally {
                liveJob.cancel()
                val trace = recording.stop()

                withContext(NonCancellable) {
                    _state.value = _state.value.copy(liveTrace = null)
                    if (content.isNotEmpty() || thinking.isNotEmpty() || toolCalls.isNotEmpty()) {
                        db.messages().upsert(
                            MessageEntity(
                                id = assistantId,
                                conversationId = conversation.id,
                                role = MessageRole.ASSISTANT,
                                content = content.toString(),
                                thinking = thinking.toString().takeIf { it.isNotBlank() },
                                thinkingMillis = thinkingMillis,
                                thinkingTokens = thinkingTokens,
                                imagePathsJson = "{}",
                                toolCallsJson = toolCalls.takeIf { it.isNotEmpty() }?.let(::encodeToolCalls),
                                tokenCount = promptTokens,
                                imageTokenCount = null,
                                generationParamsJson = params.toJsonString(),
                                tokensPerSecond = tps,
                                createdAt = System.currentTimeMillis(),
                                parentMessageId = lastParent,
                            ),
                        )
                        db.predictionRuns().record(
                            kind = ai.ondevice.core.PredictionKind.CHAT,
                            artifactId = assistantId,
                            modelId = _state.value.model?.id,
                            startedAt = startedAt,
                            trace = trace,
                            stats = SparseParams.of("tokens_per_second" to tps),
                        )
                        lastParent = assistantId
                    }
                    refreshMessages()
                }
            }

            if (toolCalls.isEmpty() || registry == null) return

            round++
            if (round > MAX_TOOL_ROUNDS) {
                db.messages().upsert(
                    toolMessage(
                        conversation.id,
                        name = "system",
                        callId = "",
                        text = "Stopped after $MAX_TOOL_ROUNDS rounds of tool calls. " +
                            "Ask again if the model should keep going.",
                        parentId = lastParent,
                    ),
                )
                refreshMessages()
                return
            }

            // Run them in order.
            for (call in toolCalls) {
                _state.value = _state.value.copy(runningTool = call.name)
                val result = registry.call(call.name, call.argumentsJson)
                db.messages().upsert(
                    toolMessage(
                        conversation.id,
                        name = call.name,
                        callId = call.id,
                        text = result.text,
                        parentId = lastParent,
                        isError = result.isError,
                    ),
                )
                refreshMessages()
            }
            _state.value = _state.value.copy(runningTool = null)
        }
    }

    private fun toolMessage(
        conversationId: String,
        name: String,
        callId: String,
        text: String,
        parentId: String?,
        isError: Boolean = false,
    ) = MessageEntity(
        id = UUID.randomUUID().toString(),
        conversationId = conversationId,
        role = MessageRole.TOOL_RESULT,
        content = text,
        thinking = null,
        thinkingMillis = null,
        thinkingTokens = null,
        imagePathsJson = "{}",
        toolCallsJson = SparseParams.of(
            "tool_name" to name,
            "tool_call_id" to callId,
            "is_error" to isError,
        ).toJsonString(),
        tokenCount = null,
        imageTokenCount = null,
        generationParamsJson = "{}",
        tokensPerSecond = null,
        createdAt = System.currentTimeMillis(),
        parentMessageId = parentId,
    )

    private fun encodeToolCalls(calls: List<ai.ondevice.engine.ToolCallRequest>): String =
        kotlinx.serialization.json.buildJsonArray {
            calls.forEach { call ->
                add(
                    kotlinx.serialization.json.buildJsonObject {
                        put("name", kotlinx.serialization.json.JsonPrimitive(call.name))
                        put("arguments", kotlinx.serialization.json.JsonPrimitive(call.argumentsJson))
                        put("id", kotlinx.serialization.json.JsonPrimitive(call.id))
                    },
                )
            }
        }.toString()

    // — export and import (SPEC §13) —

    /** Write this conversation out and hand the file back for sharing. */
    fun export(format: ExportFormat, onReady: (java.io.File) -> Unit) {
        val conversation = _state.value.conversation ?: run {
            _state.value = _state.value.copy(error = "There is no conversation to export.")
            return
        }
        viewModelScope.launch {
            val slug = conversation.title
                .lowercase()
                .replace(Regex("[^a-z0-9]+"), "-")
                .trim('-')
                .ifBlank { "conversation" }
            val result = runCatching {
                when (format) {
                    ExportFormat.ARCHIVE -> archive.exportArchive(
                        conversationIds = listOf(conversation.id),
                        destination = java.io.File(storage.exportsDir(), "$slug.zip"),
                    )
                    ExportFormat.MARKDOWN -> archive.exportMarkdown(
                        conversationId = conversation.id,
                        destination = java.io.File(storage.exportsDir(), "$slug.md"),
                    )
                }
            }
            result.fold(
                onSuccess = { file ->
                    _state.value = _state.value.copy(lastExport = file.absolutePath, error = null)
                    onReady(file)
                },
                onFailure = {
                    _state.value = _state.value.copy(
                        error = "The export failed: ${it.message}",
                        errorSuggestion = "Check there is free space in ${storage.exportsDir().name}.",
                    )
                },
            )
        }
    }

    /** Export the whole library, not just the open conversation. */
    fun exportEverything(onReady: (java.io.File) -> Unit) {
        viewModelScope.launch {
            runCatching {
                archive.exportArchive(destination = java.io.File(storage.exportsDir(), "conversations.zip"))
            }.fold(
                onSuccess = { file ->
                    _state.value = _state.value.copy(lastExport = file.absolutePath, error = null)
                    onReady(file)
                },
                onFailure = {
                    _state.value = _state.value.copy(error = "The export failed: ${it.message}")
                },
            )
        }
    }

    fun import(uri: android.net.Uri) {
        viewModelScope.launch {
            val report = runCatching {
                context.contentResolver.openInputStream(uri)?.use { archive.importArchive(it) }
                    ?: ai.ondevice.data.ImportReport(error = "That file could not be opened.")
            }.getOrElse { ai.ondevice.data.ImportReport(error = "That file could not be read: ${it.message}") }

            if (report.ok) {
                _state.value = _state.value.copy(
                    error = null,
                    importSummary = "Imported ${report.conversations} conversation(s), " +
                        "${report.messages} message(s), ${report.attachments} attachment(s).",
                )
                restore()
            } else {
                _state.value = _state.value.copy(
                    error = report.error,
                    errorSuggestion = "Pick a .zip written by this app's export.",
                )
            }
        }
    }

    fun dismissImportSummary() {
        _state.value = _state.value.copy(importSummary = null)
    }

    /** Cancellation unwinds the engine Flow, whose teardown frees native memory. */
    fun stop() {
        // The native side first, and from this thread. Cancelling the job only
        // stops the coroutine at its next suspension point, and there is no
        // suspension point inside a JNI call — so a Stop pressed while a long
        // prompt or an image is going in would otherwise sit unnoticed until
        // the work it was meant to stop had finished.
        (engines.llama as? ai.ondevice.engine.LlamaEngine)?.cancel()
        generationJob?.cancel()
        generationJob = null
    }

    fun regenerate(message: MessageEntity) {
        viewModelScope.launch {
            db.messages().deleteById(message.id)
            refreshMessages()
            val previousUser = _state.value.messages.lastOrNull { it.role == MessageRole.USER }
            if (previousUser != null) {
                _state.value = _state.value.copy(input = previousUser.content)
                db.messages().deleteById(previousUser.id)
                refreshMessages()
                send()
            }
        }
    }

    fun setModel(model: ModelEntity) {
        viewModelScope.launch {
            val conversation = _state.value.conversation ?: return@launch
            db.conversations().upsert(conversation.copy(modelId = model.id, updatedAt = System.currentTimeMillis()))
            _state.value = _state.value.copy(model = model, conversation = db.conversations().get(conversation.id))
            engines.load(model)
        }
    }

    /**
     * The chat template, overridden per model and applied by reloading it.
     *
     * A template is not a live setting: llama.cpp builds its parser and its
     * stop sequences from it once, in `common_chat_templates_init` at load, so
     * setting it on a resident model would change nothing until the next one.
     * Pass null to go back to the one in the GGUF.
     */
    fun setChatTemplate(template: String?) {
        viewModelScope.launch {
            val model = _state.value.model ?: return@launch
            val trimmed = template?.takeIf { it.isNotBlank() }
            val overrides = SparseParams.parse(model.paramOverridesJson).let {
                if (trimmed == null) it.without("chat_template") else it.with("chat_template", trimmed)
            }
            db.models().setParamOverrides(model.id, overrides.toJsonString())
            val updated = db.models().get(model.id) ?: return@launch
            _state.value = _state.value.copy(model = updated)
            engines.unload()
            engines.load(updated)
        }
    }

    /** `--chat-template-kwargs`, verbatim. Anything but a JSON object is ignored by the runtime. */
    fun setTemplateKwargs(json: String) {
        setLiveParam("chat_template_kwargs", json)
    }

    fun setSystemPrompt(value: String) {
        _state.value = _state.value.copy(systemPrompt = value)
        viewModelScope.launch {
            val conversation = _state.value.conversation ?: return@launch
            db.conversations().upsert(conversation.copy(systemPrompt = value, updatedAt = System.currentTimeMillis()))
        }
    }

    /** Live parameter edits from the Basic tier on the settings sheet. */
    fun setLiveParam(key: String, value: Any?) {
        val current = _state.value.liveOverrides
        _state.value = _state.value.copy(
            liveOverrides = if (value == null) current.without(key) else current.overlaidWith(SparseParams.of(key to value)),
        )
        viewModelScope.launch { engines.llama?.applyParams(_state.value.liveOverrides) }
    }

    fun toggleThinking(messageId: String) {
        val open = _state.value.expandedThinking
        _state.value = _state.value.copy(
            expandedThinking = if (messageId in open) open - messageId else open + messageId,
        )
    }

    /** S10 — the exact string that reaches the tokenizer. */
    fun loadPromptInspector() {
        viewModelScope.launch {
            val engine = engines.llama ?: return@launch
            val conversation = _state.value.conversation ?: return@launch
            val rendered = engine.renderPrompt(
                GenerateRequest(
                    messages = db.messages().getFor(conversation.id).map { it.toEngineMessage() },
                    params = effectiveParams(),
                    systemPrompt = _state.value.systemPrompt.takeIf { it.isNotBlank() },
                ),
            )
            _state.value = _state.value.copy(renderedPrompt = rendered)
        }
    }

    /** SPEC §4.4 — "anything you add is yours and is kept per model". */
    fun addStopSequence(value: String) {
        val updated = _state.value.userStopSequences + value
        applyStopSequences(updated)
    }

    fun removeStopSequence(index: Int) {
        val updated = _state.value.userStopSequences.toMutableList()
        if (index !in updated.indices) return
        updated.removeAt(index)
        applyStopSequences(updated)
    }

    private fun applyStopSequences(values: List<String>) {
        _state.value = _state.value.copy(userStopSequences = values)
        setLiveParam("stop", values)
        val modelId = _state.value.model?.id ?: return
        viewModelScope.launch {
            val stored = SparseParams.parse(db.models().get(modelId)?.paramOverridesJson)
            db.models().setParamOverrides(
                modelId,
                stored.overlaidWith(SparseParams.of("stop" to values)).toJsonString(),
            )
        }
    }

    fun dismissError() {
        _state.value = _state.value.copy(error = null, errorSuggestion = null)
    }

    private suspend fun refreshMessages() {
        val conversation = _state.value.conversation ?: return
        val messages = db.messages().getFor(conversation.id)
        val traces = messages
            .filter { it.role == MessageRole.ASSISTANT }
            .mapNotNull { message ->
                db.predictionRuns().getFor(message.id)
                    .firstNotNullOfOrNull { ai.ondevice.engine.ResourceTrace.parse(it.traceJson) }
                    ?.let { message.id to it }
            }
            .toMap()
        _state.value = _state.value.copy(messages = messages, traces = traces)
    }

    private fun MessageEntity.toEngineMessage(): EngineMessage {
        val meta = SparseParams.parse(toolCallsJson)
        return EngineMessage(
            // The wire roles the chat templates know are user/assistant/system/ tool.
            role = when (role) {
                MessageRole.USER -> "user"
                MessageRole.SYSTEM -> "system"
                MessageRole.TOOL_RESULT -> "tool"
                MessageRole.ASSISTANT, MessageRole.TOOL_CALL -> "assistant"
            },
            // The documents go back in front of what was typed. They are kept
            // out of `content` so the bubble shows the question rather than
            // the file — see MessageAttachments.
            content = ai.ondevice.core.MessageAttachments.of(imagePathsJson).promptText(content),
            imagePaths = SparseParams.parse(imagePathsJson).stringList("images").orEmpty(),
            toolCalls = if (role == MessageRole.ASSISTANT) decodeToolCalls(toolCallsJson) else emptyList(),
            toolCallId = meta.string("tool_call_id")?.takeIf { role == MessageRole.TOOL_RESULT },
            toolName = meta.string("tool_name")?.takeIf { role == MessageRole.TOOL_RESULT },
        )
    }

    private fun decodeToolCalls(raw: String?): List<ai.ondevice.engine.ToolCallRequest> {
        if (raw.isNullOrBlank()) return emptyList()
        return runCatching {
            kotlinx.serialization.json.Json.parseToJsonElement(raw)
                .let { it as? kotlinx.serialization.json.JsonArray }
                ?.map { element ->
                    val obj = element as kotlinx.serialization.json.JsonObject
                    ai.ondevice.engine.ToolCallRequest(
                        name = (obj["name"] as? kotlinx.serialization.json.JsonPrimitive)?.content.orEmpty(),
                        argumentsJson = (obj["arguments"] as? kotlinx.serialization.json.JsonPrimitive)?.content.orEmpty(),
                        id = (obj["id"] as? kotlinx.serialization.json.JsonPrimitive)?.content.orEmpty(),
                    )
                }
                .orEmpty()
        }.getOrDefault(emptyList())
    }

    private companion object {
        const val IMAGE_TOKEN_COST = 1456

        /** A model stuck on a failing tool would otherwise call it forever. */
        const val MAX_TOOL_ROUNDS = 5
    }
}

data class ChatState(
    val conversation: ConversationEntity? = null,
    val model: ModelEntity? = null,
    val availableModels: List<ModelEntity> = emptyList(),
    /** Text and vision downloads still running, so "none" reads as "not yet". */
    val installing: List<ai.ondevice.data.db.InstallingModel> = emptyList(),
    val messages: List<MessageEntity> = emptyList(),
    val streaming: StreamingMessage? = null,
    val input: String = "",
    val pendingAttachments: List<PendingAttachment> = emptyList(),
    val generating: Boolean = false,
    val loadingModel: Boolean = false,
    val tokensPerSecond: Float = 0f,
    val contextUsed: Int = 0,
    val cachedTokens: Int = 0,
    val systemPrompt: String = "",
    val liveOverrides: SparseParams = SparseParams.EMPTY,
    val expandedThinking: Set<String> = emptySet(),
    val renderedPrompt: RenderedPrompt? = null,
    /** Added by hand on S10, kept per model — the template's own are separate. */
    val userStopSequences: List<String> = emptyList(),
    /** Non-null while a tool call is actually running, for the chat spinner. */
    val runningTool: String? = null,
    /** Where the last export landed, so the screen can name the file. */
    val lastExport: String? = null,
    val importSummary: String? = null,
    /** What the engine says is resident — not what the conversation prefers. */
    val loadedModelId: String? = null,
    /** The template the loaded model is actually rendering with. */
    val chatTemplate: String? = null,
    val templateSource: String = "gguf.chat_template",
    /** What the runtime held when the model loaded; [templateKwargsJson] is the live answer. */
    val loadedTemplateKwargsJson: String = "{}",
    val error: String? = null,
    val errorSuggestion: String? = null,
    /** Sampled while the current turn runs; null when nothing is generating. */
    val liveTrace: ai.ondevice.engine.ResourceTrace? = null,
    /** What each assistant message cost, by message id. */
    val traces: Map<String, ai.ondevice.engine.ResourceTrace> = emptyMap(),
) {
    /** The context the model is *loaded at*, which is the per-model `n_ctx` override when there is one — not the architecture's theoretical maximum. */
    val contextLimit: Int
        get() = SparseParams.parse(model?.paramOverridesJson).int("n_ctx")
            ?: model?.contextLength
            ?: 8192

    /** The template arguments in force: what has been set this session, else what the model loaded with. */
    val templateKwargsJson: String
        get() = liveOverrides.string("chat_template_kwargs") ?: loadedTemplateKwargsJson

    /** Whether an image can reach this model, which decides what the file picker offers. */
    val acceptsImages: Boolean
        get() = model?.modality == Modality.VISION &&
            ai.ondevice.core.ComponentCheck.forChatImage(
                SparseParams.parse(model.companionPathsJson).keys.associateWith { "" },
            ) == null
}

/** A file the user attached but has not sent yet. */
data class PendingAttachment(
    val path: String,
    val name: String,
    val kind: ai.ondevice.data.AttachmentKind,
    val text: String = "",
    val tokenCost: Int = 0,
    /** e.g. "only the first 512 kB was read". */
    val note: String? = null,
)

data class StreamingMessage(
    val id: String,
    val content: String = "",
    val thinking: String = "",
    val thinkingMillis: Long? = null,
    val thinkingTokens: Int? = null,
    val thinkingComplete: Boolean = false,
    /**
     * How many tokens went in, once they have. Null while the prompt is still
     * being read — which on a phone is a minute of work with nothing on screen,
     * and is what "it froze" turned out to mean.
     */
    val promptTokens: Int? = null,
)

/** The two shapes a conversation can leave the app in. */
enum class ExportFormat(val label: String, val mime: String) {
    ARCHIVE("Archive (.zip)", "application/zip"),
    MARKDOWN("Markdown (.md)", "text/markdown"),
}
