package ai.ondevice.di

import android.content.Context
import androidx.room.Room
import ai.ondevice.data.ModelStorage
import ai.ondevice.data.db.OnDeviceDatabase
import ai.ondevice.data.download.Downloader
import ai.ondevice.data.hf.DeviceCapabilities
import ai.ondevice.data.hf.HfApi
import ai.ondevice.data.hf.ModelResolver
import ai.ondevice.data.prefs.AppPrefs
import ai.ondevice.data.secure.TokenStore
import ai.ondevice.engine.Benchmarker
import ai.ondevice.engine.EngineManager
import ai.ondevice.engine.RuntimeRegistry
import ai.ondevice.params.ParamRepository
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.components.SingletonComponent
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import okhttp3.OkHttpClient
import java.util.concurrent.TimeUnit
import javax.inject.Qualifier
import javax.inject.Singleton

@Qualifier
@Retention(AnnotationRetention.BINARY)
annotation class ApplicationScope

@Module
@InstallIn(SingletonComponent::class)
object AppModule {

    @Provides
    @Singleton
    @ApplicationScope
    fun provideApplicationScope(): CoroutineScope = CoroutineScope(SupervisorJob() + Dispatchers.Default)

    @Provides
    @Singleton
    fun provideOkHttp(): OkHttpClient = OkHttpClient.Builder()
        // Multi-gigabyte transfers over a mobile link: generous read timeouts,
        // and retry-on-failure left on because the downloader also resumes.
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(120, TimeUnit.SECONDS)
        .writeTimeout(60, TimeUnit.SECONDS)
        .retryOnConnectionFailure(true)
        .followRedirects(true)
        .build()

    @Provides
    @Singleton
    fun provideDatabase(@ApplicationContext context: Context): OnDeviceDatabase =
        Room.databaseBuilder(context, OnDeviceDatabase::class.java, OnDeviceDatabase.NAME)
            // A real migration rather than a destructive one: the conversations
            // and the per-model parameter overrides in this database are the
            // user's work, and dropping them to add a table is not a trade the
            // app gets to make on their behalf.
            .addMigrations(OnDeviceDatabase.MIGRATION_1_2)
            .build()

    @Provides @Singleton fun provideModelDao(db: OnDeviceDatabase) = db.models()
    @Provides @Singleton fun provideConversationDao(db: OnDeviceDatabase) = db.conversations()
    @Provides @Singleton fun provideMessageDao(db: OnDeviceDatabase) = db.messages()

    @Provides
    @Singleton
    fun providePrefs(@ApplicationContext context: Context) = AppPrefs(context)

    @Provides
    @Singleton
    fun provideTokenStore(@ApplicationContext context: Context) = TokenStore(context)

    @Provides
    @Singleton
    fun provideCapabilities(@ApplicationContext context: Context) = DeviceCapabilities(context)

    @Provides
    @Singleton
    fun provideRegistry(@ApplicationContext context: Context) = RuntimeRegistry(context)

    @Provides
    @Singleton
    fun provideHfApi(client: OkHttpClient, tokens: TokenStore) = HfApi(client, tokens)

    @Provides
    @Singleton
    fun provideResolver(api: HfApi, registry: RuntimeRegistry) = ModelResolver(api, registry)

    @Provides
    @Singleton
    fun provideStorage(@ApplicationContext context: Context, db: OnDeviceDatabase) = ModelStorage(context, db)

    @Provides
    @Singleton
    fun provideParamRepository(@ApplicationContext context: Context, db: OnDeviceDatabase) =
        ParamRepository(context, db)

    @Provides
    @Singleton
    fun provideDownloader(
        @ApplicationContext context: Context,
        client: OkHttpClient,
        db: OnDeviceDatabase,
        prefs: AppPrefs,
        tokens: TokenStore,
        @ApplicationScope scope: CoroutineScope,
    ) = Downloader(context, client, db, prefs, tokens, scope)

    @Provides
    @Singleton
    fun provideEngineManager(
        @ApplicationContext context: Context,
        registry: RuntimeRegistry,
        db: OnDeviceDatabase,
        prefs: AppPrefs,
        capabilities: DeviceCapabilities,
        @ApplicationScope scope: CoroutineScope,
    ) = EngineManager(context, registry, db, prefs, capabilities, scope)

    @Provides
    @Singleton
    fun provideBenchmarker(registry: RuntimeRegistry, db: OnDeviceDatabase) = Benchmarker(registry, db)

    @Provides
    @Singleton
    fun provideToolProviders(db: OnDeviceDatabase, capabilities: DeviceCapabilities) =
        ai.ondevice.tools.ToolProviderFactory(db, capabilities)

    @Provides
    @Singleton
    fun provideConversationArchive(db: OnDeviceDatabase, storage: ModelStorage) =
        ai.ondevice.data.ConversationArchive(db, storage)

    @Provides
    @Singleton
    fun provideAttachmentStore(@ApplicationContext context: Context, storage: ModelStorage) =
        ai.ondevice.data.AttachmentStore(context, storage)

    @Provides
    @Singleton
    fun providePhonemizer(@ApplicationContext context: Context) =
        ai.ondevice.speech.Phonemizer(context)

    @Provides
    @Singleton
    fun provideKokoroEngine(phonemizer: ai.ondevice.speech.Phonemizer) =
        ai.ondevice.speech.KokoroEngine(phonemizer)

    @Provides
    @Singleton
    fun provideOmniVoiceEngine() = ai.ondevice.speech.OmniVoiceEngine()

    @Provides
    @Singleton
    fun provideSpeechSynthesizer(
        @ApplicationContext context: Context,
        kokoro: ai.ondevice.speech.KokoroEngine,
        omniVoice: ai.ondevice.speech.OmniVoiceEngine,
    ) = ai.ondevice.speech.SpeechSynthesizer(context, kokoro, omniVoice)

    @Provides
    @Singleton
    fun provideTranscriber(@ApplicationContext context: Context) =
        ai.ondevice.engine.Transcriber(context)

    @Provides
    @Singleton
    fun provideDiffusionEngine(@ApplicationContext context: Context) =
        ai.ondevice.engine.DiffusionEngine(context)
}
