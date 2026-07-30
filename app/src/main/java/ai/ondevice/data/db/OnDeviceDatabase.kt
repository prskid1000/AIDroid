package ai.ondevice.data.db

import androidx.room.Database
import androidx.room.RoomDatabase
import androidx.room.TypeConverter
import androidx.room.TypeConverters
import ai.ondevice.core.AttachmentRole
import ai.ondevice.core.BackendId
import ai.ondevice.core.DownloadState
import ai.ondevice.core.MessageRole
import ai.ondevice.core.Modality
import ai.ondevice.core.ModelFormat
import ai.ondevice.core.RuntimeState

/**
 * Enums are stored by name, not ordinal. An ordinal column silently
 * reinterprets itself the moment someone inserts a value into the middle of an
 * enum; the name survives that.
 */
class Converters {
    @TypeConverter fun modalityTo(v: Modality?): String? = v?.name
    @TypeConverter fun modalityFrom(v: String?): Modality? = v?.let { runCatching { Modality.valueOf(it) }.getOrNull() }

    @TypeConverter fun formatTo(v: ModelFormat?): String? = v?.name
    @TypeConverter fun formatFrom(v: String?): ModelFormat? = v?.let { runCatching { ModelFormat.valueOf(it) }.getOrNull() }

    @TypeConverter fun backendTo(v: BackendId?): String? = v?.name
    @TypeConverter fun backendFrom(v: String?): BackendId? = v?.let { runCatching { BackendId.valueOf(it) }.getOrNull() }

    @TypeConverter fun roleTo(v: MessageRole?): String? = v?.name
    @TypeConverter fun roleFrom(v: String?): MessageRole? = v?.let { runCatching { MessageRole.valueOf(it) }.getOrNull() }

    @TypeConverter fun dlStateTo(v: DownloadState?): String? = v?.name
    @TypeConverter fun dlStateFrom(v: String?): DownloadState? = v?.let { runCatching { DownloadState.valueOf(it) }.getOrNull() }

    @TypeConverter fun rtStateTo(v: RuntimeState?): String? = v?.name
    @TypeConverter fun rtStateFrom(v: String?): RuntimeState? = v?.let { runCatching { RuntimeState.valueOf(it) }.getOrNull() }

    @TypeConverter fun attachmentRoleTo(v: AttachmentRole?): String? = v?.name
    @TypeConverter fun attachmentRoleFrom(v: String?): AttachmentRole? =
        v?.let { runCatching { AttachmentRole.valueOf(it) }.getOrNull() }
}

@Database(
    entities = [
        ModelEntity::class,
        BenchmarkEntity::class,
        PresetEntity::class,
        PersonaEntity::class,
        ConversationEntity::class,
        MessageEntity::class,
        GeneratedImageEntity::class,
        TranscriptEntity::class,
        DownloadJobEntity::class,
        RuntimeBundleEntity::class,
        ParamManifestEntity::class,
        McpServerEntity::class,
    ],
    version = 1,
    exportSchema = true,
)
@TypeConverters(Converters::class)
abstract class OnDeviceDatabase : RoomDatabase() {
    abstract fun models(): ModelDao
    abstract fun benchmarks(): BenchmarkDao
    abstract fun presets(): PresetDao
    abstract fun personas(): PersonaDao
    abstract fun conversations(): ConversationDao
    abstract fun messages(): MessageDao
    abstract fun images(): GeneratedImageDao
    abstract fun transcripts(): TranscriptDao
    abstract fun downloads(): DownloadDao
    abstract fun runtimes(): RuntimeDao
    abstract fun manifests(): ParamManifestDao
    abstract fun mcpServers(): McpServerDao

    /**
     * One version, no migrations.
     *
     * The schema had reached v4 through three migrations, each written to carry
     * an installed base that does not exist: this app has never shipped, and the
     * only databases in the world are on development devices that are wiped
     * whenever the model set changes anyway. Three migration scripts, three
     * exported schema files and a backfill whose correctness nobody could check
     * against real data — all of it maintenance for a population of zero.
     *
     * So the schema is stated once, at v1, and the builder falls back to
     * recreating the database when it finds an older one. When the app does
     * ship, this becomes v1 for real and the next change is v2 with a migration
     * that has someone's data to protect. Until then a migration would be
     * ceremony, and the ceremony was starting to hide things — `completedAt`'s
     * backfill deliberately reproduced the bug it was replacing, which is right
     * for a real upgrade and pointless when there is nothing to upgrade.
     */
    companion object {
        const val NAME = "ondevice.db"
    }
}
