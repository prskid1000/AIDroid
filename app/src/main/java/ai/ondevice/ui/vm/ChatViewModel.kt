package ai.ondevice.ui.vm

import android.content.Context
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import ai.ondevice.core.BackendId
import ai.ondevice.core.MessageRole
import ai.ondevice.core.Modality
import ai.ondevice.core.SparseParams
import ai.ondevice.data.db.ConversationEntity
import ai.ondevice.data.db.MessageEntity
import ai.ondevice.data.db.ModelEntity
import ai.ondevice.data.db.OnDeviceDatabase
import ai.ondevice.data.db.PersonaEntity
import ai.ondevice.data.db.PresetEntity
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

/**
 * S6/S7/S10 — the chat loop.
 *
 * Two spec obligations live here rather than in the UI:
 *  - **The full parameter set is stored with every message** (SPEC §11,
 *    Appendix A #6). Cheap at write time, impossible to reconstruct later.
 *  - **Generation is cancellable at every stage** and cancelling frees native
 *    memory — the job is cancelled, which unwinds the engine's Flow through its
 *    `onCompletion` teardown.
 */
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

        // The model list has to be live. Taking it once at construction meant a
        // model that finished downloading while the chat screen existed never
        // appeared in the picker — and since the picker only rendered when
        // there was more than one model, the *first* model you ever installed
        // was unreachable until the app was restarted.
        viewModelScope.launch {
            db.models().observeInstalled().collect { models ->
                _state.value = _state.value.copy(
                    availableModels = models.filter {
                        it.modality == Modality.TEXT || it.modality == Modality.VISION
                    },
                    // Re-read the selected row too, not just the list. The
                    // parameters screen edits this same row, and holding a
                    // snapshot taken at restore meant the context readout kept
                    // reporting the old n_ctx after it had been changed —
                    // "8192 ctx" on a model since set to 2048.
                    model = _state.value.model?.let { current ->
                        models.firstOrNull { it.id == current.id } ?: current
                    },
                )
            }
        }

        // Settings → Backend is a global preference, so it has to be observed
        // rather than read once: changing it while the chat is open used to
        // leave the readout describing the old choice.
        viewModelScope.launch {
            prefs.backendMode.collect { mode ->
                _state.value = _state.value.copy(backendPreference = mode)
            }
        }

        // Likewise the engine's own view of what is loaded, so the sheet says
        // "loaded" only when something actually is.
        viewModelScope.launch {
            engines.state.collect { engine ->
                _state.value = _state.value.copy(
                    loadedModelId = engine.loaded?.modelId,
                    loadingModel = engine.loading,
                    // The backend the engine actually resolved, which is the
                    // only trustworthy answer: it is the end of a four-step
                    // fallback — per-model override, then the global setting,
                    // then the measured winner, then OpenCL — and none of those
                    // steps is visible from the model row alone.
                    loadedBackend = engine.backend,
                )
            }
        }
    }

    private suspend fun restore() {
        val conversation = prefs.lastConversationId.first()?.let { db.conversations().get(it) }
            ?: db.conversations().mostRecent()
            ?: newConversation()

        val model = conversation.modelId?.let { db.models().get(it) }
            ?: db.models().observeInstalledByModality(Modality.TEXT).first().firstOrNull()

        val presets = db.presets().observeFor(Modality.TEXT).first()
        val personas = db.personas().observeAll().first()

        _state.value = _state.value.copy(
            conversation = conversation,
            model = model,
            messages = db.messages().getFor(conversation.id),
            presets = presets,
            personas = personas,
            selectedPresetId = conversation.presetId ?: model?.defaultPresetId ?: presets.firstOrNull()?.id,
            selectedPersonaId = conversation.personaId,
            systemPrompt = conversation.systemPrompt
                ?: personas.firstOrNull { it.id == conversation.personaId }?.systemPrompt
                ?: "",
            availableModels = db.models().observeInstalledByModality(Modality.TEXT).first(),
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

    /**
     * Start a fresh conversation.
     *
     * A *new* one, not a cleared one. Wiping the current thread's messages in
     * place would destroy work the user might want back and would leave the
     * export they were about to take pointing at an empty conversation; a new
     * row costs nothing and leaves the old thread whole in the library. The KV
     * cache is dropped because it holds the previous thread's tokens, and
     * carrying those into a conversation the user thinks is empty is the kind
     * of invisible context that produces baffling replies.
     */
    fun startNewConversation() {
        if (_state.value.generating) stop()
        // Already on an empty one. Writing a second row would leave the first
        // behind for good — nothing ever deletes it — so pressing this a few
        // times filled the library with threads that had never held anything.
        if (_state.value.messages.isEmpty() && _state.value.conversation != null) return
        viewModelScope.launch {
            // Dropping the KV is llama-specific, so it is asked for by type
            // rather than added to the engine interface — no other runtime has
            // a conversation-shaped cache to drop.
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

    /**
     * Switch to a thread the user picked out of the library.
     *
     * The KV goes with it for the same reason [startNewConversation] drops it:
     * the cache holds the *previous* thread's tokens, and llama.cpp would happily
     * treat them as a shared prefix of a conversation that never contained them.
     */
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
                selectedPresetId = conversation.presetId ?: _state.value.selectedPresetId,
                selectedPersonaId = conversation.personaId,
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

    /**
     * Attach a file the user picked, and decide what it *is* before deciding
     * what to do with it.
     *
     * The three kinds go three different ways, and conflating them is how these
     * features usually break: an image is passed to the vision projector as an
     * image; a document is read and its text enters the prompt as text, priced
     * in tokens the user can see before sending; audio needs transcription,
     * which needs a runtime that may not be installed, so it says so rather
     * than attaching something the model will never receive.
     */
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
                    // Two separate ways this fails, and they used to be one
                    // check. A model can be classified as vision and still have
                    // arrived without its projector file — the classification
                    // reads the repo, the companion is a download that can be
                    // skipped — so the second question has to be asked of the
                    // files rather than of the label.
                    val missing = if (model?.modality != Modality.VISION) {
                        ai.ondevice.core.MissingComponent(
                            what = "${model?.displayName ?: "This model"} cannot see images",
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

                ai.ondevice.data.AttachmentKind.AUDIO -> {
                    _state.value = _state.value.copy(
                        error = "Audio has to be transcribed before it can enter a prompt, and the " +
                            "whisper.cpp runtime is not installed.",
                        errorSuggestion = "Settings → Runtimes installs it. Voice → File transcribes there.",
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

    /** The parameter set actually in force: preset, then per-model overrides. */
    private suspend fun effectiveParams(): SparseParams {
        val presetJson = _state.value.selectedPresetId?.let { db.presets().get(it)?.paramsJson }
        val preset = SparseParams.parse(presetJson)
        val modelOverrides = SparseParams.parse(_state.value.model?.paramOverridesJson)
        return preset.overlaidWith(modelOverrides).overlaidWith(_state.value.liveOverrides)
    }

    fun send() {
        val typed = _state.value.input.trim()
        val conversation = _state.value.conversation ?: return
        val model = _state.value.model ?: run {
            _state.value = _state.value.copy(error = "Pick a model first — Models → Add.")
            return
        }
        val pending = _state.value.pendingAttachments
        if (typed.isEmpty() && pending.isEmpty()) return

        val images = pending.filter { it.kind == ai.ondevice.data.AttachmentKind.IMAGE }.map { it.path }

        // A document's text becomes part of the message, fenced and named, so
        // the conversation records what the model actually read — not a path
        // that means nothing once the file moves.
        val documents = pending.filter { it.kind == ai.ondevice.data.AttachmentKind.DOCUMENT }
        val text = buildString {
            documents.forEach { document ->
                appendLine("--- ${document.name} ---")
                appendLine(document.text)
                appendLine("--- end of ${document.name} ---")
                appendLine()
            }
            append(typed)
        }.trim()

        _state.value = _state.value.copy(input = "", pendingAttachments = emptyList(), error = null)

        generationJob = viewModelScope.launch {
            val params = effectiveParams()

            val userMessage = MessageEntity(
                id = UUID.randomUUID().toString(),
                conversationId = conversation.id,
                role = MessageRole.USER,
                content = text,
                thinking = null,
                thinkingMillis = null,
                thinkingTokens = null,
                imagePathsJson = SparseParams.of("images" to images).toJsonString(),
                toolCallsJson = null,
                tokenCount = null,
                imageTokenCount = images.size * IMAGE_TOKEN_COST,
                generationParamsJson = params.toJsonString(),
                tokensPerSecond = null,
                backend = null,
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
                // Stop is a cancellation, and a cancelled coroutine cannot
                // suspend: the first suspending call in this block throws
                // CancellationException again and everything after it is
                // skipped. That is why Stop appeared not to work — `touch`
                // threw, so `generating` was never cleared and the UI kept
                // showing a running turn over a native loop that had already
                // finished. Teardown has to be uncancellable to run at all.
                InferenceService.releaseWakeLock(context)
                withContext(NonCancellable) {
                    db.conversations().touch(conversation.id, System.currentTimeMillis())
                    _state.value = _state.value.copy(generating = false, streaming = null)
                    refreshMessages()
                }
            }
        }
    }

    /**
     * One assistant turn, and the tool loop around it.
     *
     * A model that asks for a tool has not finished its turn: it has to see the
     * result and speak again. So this generates, and if the reply is a tool
     * call, runs it, writes the result into the conversation as a `tool`
     * message, and generates once more — up to [MAX_TOOL_ROUNDS], because a
     * model that loops on a failing tool would otherwise run the battery flat.
     * The cap is surfaced in the conversation rather than silently applied.
     */
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
            var backend: BackendId? = null
            var promptTokens = 0
            val toolCalls = mutableListOf<ai.ondevice.engine.ToolCallRequest>()

            _state.value = _state.value.copy(
                generating = true,
                streaming = StreamingMessage(id = assistantId),
            )

            // Per round, not per turn. A reply that calls a tool generates
            // again after the result comes back, and those are separate pieces
            // of work with separate costs — averaging them into one trace would
            // hide the fact that the second round starts from a warm cache.
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
                            backend = event.backend
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
                // Both of these have to happen before the NonCancellable block,
                // and neither may suspend: Stop cancels this coroutine, so a
                // suspending stop here would never run and the sampler would
                // outlive the run it was describing.
                liveJob.cancel()
                val trace = recording.stop()

                // Persist whatever was generated, including on cancellation — a
                // half-finished reply is still the user's, and it carries the
                // parameters it was produced under. NonCancellable is what makes
                // that true rather than aspirational: Stop cancels this
                // coroutine, and without it the upsert below is itself a
                // suspending call on a cancelled job, so the partial reply was
                // thrown away every single time.
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
                                backend = backend,
                                createdAt = System.currentTimeMillis(),
                                parentMessageId = lastParent,
                            ),
                        )
                        // Keyed to the message, so a stopped generation keeps
                        // its trace for the same reason it keeps its partial
                        // reply: what it cost to get that far is still true.
                        db.predictionRuns().record(
                            kind = ai.ondevice.core.PredictionKind.CHAT,
                            artifactId = assistantId,
                            modelId = _state.value.model?.id,
                            backend = backend,
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

            // Run them in order. Sequential rather than parallel on purpose: a
            // later call's arguments routinely depend on an earlier result, and
            // the model wrote them expecting to be read in order.
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
        backend = null,
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

    /**
     * Write this conversation out and hand the file back for sharing.
     *
     * Two formats because they answer different questions: the `.zip`
     * round-trips losslessly, and the `.md` is what you paste into a bug
     * report. Offering only the readable one would make the app the only place
     * a conversation can fully exist, which §13 rules out.
     */
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

    fun setPreset(presetId: String) {
        viewModelScope.launch {
            val conversation = _state.value.conversation ?: return@launch
            db.conversations().upsert(conversation.copy(presetId = presetId, updatedAt = System.currentTimeMillis()))
            _state.value = _state.value.copy(selectedPresetId = presetId)
        }
    }

    fun setPersona(persona: PersonaEntity) {
        viewModelScope.launch {
            val conversation = _state.value.conversation ?: return@launch
            db.conversations().upsert(
                conversation.copy(
                    personaId = persona.id,
                    systemPrompt = persona.systemPrompt,
                    updatedAt = System.currentTimeMillis(),
                ),
            )
            _state.value = _state.value.copy(
                selectedPersonaId = persona.id,
                systemPrompt = persona.systemPrompt,
            )
        }
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

    /**
     * SPEC §4.4 — "anything you add is yours and is kept per model". The
     * template's own stops come back from the engine every render, so the user's
     * live in `stop`, the manifest's own key, and are merged for display rather
     * than rewritten into the template.
     */
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
        // Loaded alongside the messages rather than observed: runs only ever
        // appear as a consequence of a message being written, and this function
        // already runs on every one of those.
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
            // The wire roles the chat templates know are user/assistant/system/
            // tool. Our storage enum is finer-grained than that, so the mapping
            // is explicit rather than a lowercased name that happens to match.
            role = when (role) {
                MessageRole.USER -> "user"
                MessageRole.SYSTEM -> "system"
                MessageRole.TOOL_RESULT -> "tool"
                MessageRole.ASSISTANT, MessageRole.TOOL_CALL -> "assistant"
            },
            content = content,
            imagePaths = SparseParams.parse(imagePathsJson).stringList("images").orEmpty(),
            // An assistant message that asked for tools has to go back to the
            // template *as* tool calls, not as text — the template renders each
            // family's syntax, and re-feeding our rendering of it would teach
            // the model a format it does not use.
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

        /**
         * A model stuck on a failing tool would otherwise call it forever. The
         * cap is deliberately visible in the conversation when it is reached.
         */
        const val MAX_TOOL_ROUNDS = 5
    }
}

data class ChatState(
    val conversation: ConversationEntity? = null,
    val model: ModelEntity? = null,
    val availableModels: List<ModelEntity> = emptyList(),
    val messages: List<MessageEntity> = emptyList(),
    val streaming: StreamingMessage? = null,
    val input: String = "",
    val pendingAttachments: List<PendingAttachment> = emptyList(),
    val generating: Boolean = false,
    val loadingModel: Boolean = false,
    val tokensPerSecond: Float = 0f,
    val contextUsed: Int = 0,
    val cachedTokens: Int = 0,
    val presets: List<PresetEntity> = emptyList(),
    val personas: List<PersonaEntity> = emptyList(),
    val selectedPresetId: String? = null,
    val selectedPersonaId: String? = null,
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

    /**
     * The backend actually in use, from the engine. Null until something is
     * loaded, which the toolbar must render as "not loaded" rather than
     * guessing — it used to print the literal string "OpenCL" in that case,
     * which meant the readout was wrong on any device that chose otherwise and
     * on every device before the first load.
     */
    val loadedBackend: ai.ondevice.core.BackendId? = null,
    /** The global preference, for describing what *would* be used. */
    val backendPreference: String = ai.ondevice.data.prefs.AppPrefs.BACKEND_AUTO,
    val error: String? = null,
    val errorSuggestion: String? = null,
    /** Sampled while the current turn runs; null when nothing is generating. */
    val liveTrace: ai.ondevice.engine.ResourceTrace? = null,
    /** What each assistant message cost, by message id. */
    val traces: Map<String, ai.ondevice.engine.ResourceTrace> = emptyMap(),
) {
    /**
     * The context the model is *loaded at*, which is the per-model `n_ctx`
     * override when there is one — not the architecture's theoretical maximum.
     * Showing 262 144 when the KV cache was sized for 8 192 would misreport how
     * much room the conversation actually has.
     */
    val contextLimit: Int
        get() = SparseParams.parse(model?.paramOverridesJson).int("n_ctx")
            ?: model?.contextLength
            ?: 8192
    val presetName: String? get() = presets.firstOrNull { it.id == selectedPresetId }?.name
}

/**
 * A file the user attached but has not sent yet.
 *
 * It carries its own token cost because the composer states the price *before*
 * the send — SPEC §4.5: an image or a document that quietly consumes half the
 * context is the kind of surprise this app is supposed to prevent.
 */
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
)

/**
 * The two shapes a conversation can leave the app in.
 *
 * [ARCHIVE] round-trips — every generation parameter, measured tok/s, backend
 * and attachment comes back on import. [MARKDOWN] does not, and is not meant
 * to: it is for reading.
 */
enum class ExportFormat(val label: String, val mime: String) {
    ARCHIVE("Archive (.zip)", "application/zip"),
    MARKDOWN("Markdown (.md)", "text/markdown"),
}
