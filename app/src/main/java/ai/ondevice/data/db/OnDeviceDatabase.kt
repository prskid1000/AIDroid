package ai.ondevice.data.db

import androidx.room.Database
import androidx.room.RoomDatabase
import androidx.room.TypeConverter
import androidx.room.TypeConverters
import androidx.room.migration.Migration
import androidx.sqlite.db.SupportSQLiteDatabase
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
    version = 3,
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

    companion object {
        const val NAME = "ondevice.db"

        /** v2 adds the MCP server list. Nothing existing changes shape. */
        val MIGRATION_1_2 = object : Migration(1, 2) {
            override fun migrate(db: SupportSQLiteDatabase) {
                db.execSQL(
                    """
                    CREATE TABLE IF NOT EXISTS `mcp_servers` (
                        `id` TEXT NOT NULL,
                        `name` TEXT NOT NULL,
                        `url` TEXT NOT NULL,
                        `authHeader` TEXT,
                        `enabled` INTEGER NOT NULL,
                        `lastToolsJson` TEXT,
                        `lastCheckedAt` INTEGER,
                        `lastError` TEXT,
                        `createdAt` INTEGER NOT NULL,
                        PRIMARY KEY(`id`)
                    )
                    """.trimIndent(),
                )
            }
        }

        /**
         * v3 records the add-on role the user chose, rather than re-deriving it
         * from the file path on every read.
         *
         * Rows installed before this version have no answer to migrate, and
         * leaving them NULL would read as "base model" — putting every already
         * installed ControlNet and upscaler back in the base-model picker, the
         * exact fault this column exists to end. So they are backfilled once,
         * here, with the path classifier that used to run on every read.
         *
         * A one-time backfill of rows that predate the field is a different
         * thing from inferring the answer forever: it runs once, its result is
         * visible and correctable, and nothing downstream re-derives anything.
         * Where the classifier has no opinion the row stays NULL, which for a
         * checkpoint is the right answer anyway.
         */
        val MIGRATION_2_3 = object : Migration(2, 3) {
            override fun migrate(db: SupportSQLiteDatabase) {
                db.execSQL("ALTER TABLE `models` ADD COLUMN `attachmentRole` TEXT DEFAULT NULL")
                db.query("SELECT `id`, `localPath` FROM `models`").use { cursor ->
                    val updates = mutableListOf<Pair<String, String>>()
                    while (cursor.moveToNext()) {
                        val id = cursor.getString(0) ?: continue
                        val path = cursor.getString(1) ?: continue
                        AttachmentRole.classify(path)?.let { updates += id to it.name }
                    }
                    updates.forEach { (id, role) ->
                        db.execSQL(
                            "UPDATE `models` SET `attachmentRole` = ? WHERE `id` = ?",
                            arrayOf<Any>(role, id),
                        )
                    }
                }
            }
        }
    }
}
