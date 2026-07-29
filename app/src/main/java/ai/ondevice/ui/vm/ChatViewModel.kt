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
import dagger.hilt.android.lifecycle.HiltViewModel
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.Job
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
) : ViewModel() {

    private val _state = MutableStateFlow(ChatState())
    val state: StateFlow<ChatState> = _state.asStateFlow()

    private var generationJob: Job? = null

    init {
        viewModelScope.launch { restore() }
    }

    private suspend fun restore() {
        val conversation = prefs.lastConversationId.first()?.let { db.conversations().get(it) }
            ?: db.conversations().mostRecent()
            ?: newConversation()

        val model = conversation.modelId?.let { db.models().get(it) }
            ?: db.models().observeByModality(Modality.TEXT).first().firstOrNull()

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
            availableModels = db.models().observeByModality(Modality.TEXT).first(),
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

    fun onInputChange(value: String) {
        _state.value = _state.value.copy(input = value)
    }

    fun attachImage(path: String) {
        _state.value = _state.value.copy(pendingImages = _state.value.pendingImages + path)
    }

    fun removeImage(path: String) {
        _state.value = _state.value.copy(pendingImages = _state.value.pendingImages - path)
    }

    /** The parameter set actually in force: preset, then per-model overrides. */
    private suspend fun effectiveParams(): SparseParams {
        val presetJson = _state.value.selectedPresetId?.let { db.presets().get(it)?.paramsJson }
        val preset = SparseParams.parse(presetJson)
        val modelOverrides = SparseParams.parse(_state.value.model?.paramOverridesJson)
        return preset.overlaidWith(modelOverrides).overlaidWith(_state.value.liveOverrides)
    }

    fun send() {
        val text = _state.value.input.trim()
        val conversation = _state.value.conversation ?: return
        val model = _state.value.model ?: run {
            _state.value = _state.value.copy(error = "Pick a model first — Models → Add.")
            return
        }
        if (text.isEmpty() && _state.value.pendingImages.isEmpty()) return

        val images = _state.value.pendingImages
        _state.value = _state.value.copy(input = "", pendingImages = emptyList(), error = null)

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

            val assistantId = UUID.randomUUID().toString()
            val startedAt = System.currentTimeMillis()
            var content = StringBuilder()
            var thinking = StringBuilder()
            var thinkingMillis: Long? = null
            var thinkingTokens: Int? = null
            var tps = 0f
            var backend: BackendId? = null
            var promptTokens = 0

            _state.value = _state.value.copy(
                generating = true,
                streaming = StreamingMessage(id = assistantId),
            )

            try {
                engine.generate(
                    GenerateRequest(
                        messages = db.messages().getFor(conversation.id).map { it.toEngineMessage() },
                        params = params,
                        systemPrompt = _state.value.systemPrompt.takeIf { it.isNotBlank() },
                        imagePaths = images,
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
                InferenceService.releaseWakeLock(context)

                // Persist whatever was generated, including on cancellation —
                // a half-finished reply is still the user's, and it carries the
                // parameters it was produced under.
                if (content.isNotEmpty() || thinking.isNotEmpty()) {
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
                            toolCallsJson = null,
                            tokenCount = promptTokens,
                            imageTokenCount = null,
                            generationParamsJson = params.toJsonString(),
                            tokensPerSecond = tps,
                            backend = backend,
                            createdAt = System.currentTimeMillis(),
                            parentMessageId = userMessage.id,
                        ),
                    )
                }
                db.conversations().touch(conversation.id, System.currentTimeMillis())
                _state.value = _state.value.copy(generating = false, streaming = null)
                refreshMessages()
            }
        }
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

    fun dismissError() {
        _state.value = _state.value.copy(error = null, errorSuggestion = null)
    }

    private suspend fun refreshMessages() {
        val conversation = _state.value.conversation ?: return
        _state.value = _state.value.copy(messages = db.messages().getFor(conversation.id))
    }

    private fun MessageEntity.toEngineMessage() = EngineMessage(
        role = role.name.lowercase(),
        content = content,
        imagePaths = SparseParams.parse(imagePathsJson).stringList("images").orEmpty(),
    )

    private companion object {
        const val IMAGE_TOKEN_COST = 1456
    }
}

data class ChatState(
    val conversation: ConversationEntity? = null,
    val model: ModelEntity? = null,
    val availableModels: List<ModelEntity> = emptyList(),
    val messages: List<MessageEntity> = emptyList(),
    val streaming: StreamingMessage? = null,
    val input: String = "",
    val pendingImages: List<String> = emptyList(),
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
    val error: String? = null,
    val errorSuggestion: String? = null,
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

data class StreamingMessage(
    val id: String,
    val content: String = "",
    val thinking: String = "",
    val thinkingMillis: Long? = null,
    val thinkingTokens: Int? = null,
    val thinkingComplete: Boolean = false,
)
