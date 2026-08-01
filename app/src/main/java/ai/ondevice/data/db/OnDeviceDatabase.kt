package ai.ondevice.data.db

import androidx.room.Database
import androidx.room.RoomDatabase
import androidx.room.TypeConverter
import androidx.room.TypeConverters
import ai.ondevice.core.AttachmentRole
import ai.ondevice.core.DownloadState
import ai.ondevice.core.MessageRole
import ai.ondevice.core.Modality
import ai.ondevice.core.ModelFormat
import ai.ondevice.core.PredictionKind
import ai.ondevice.core.RuntimeState

/** Enums are stored by name, not ordinal. */
class Converters {
    @TypeConverter fun modalityTo(v: Modality?): String? = v?.name
    @TypeConverter fun modalityFrom(v: String?): Modality? = v?.let { runCatching { Modality.valueOf(it) }.getOrNull() }

    @TypeConverter fun formatTo(v: ModelFormat?): String? = v?.name
    @TypeConverter fun formatFrom(v: String?): ModelFormat? = v?.let { runCatching { ModelFormat.valueOf(it) }.getOrNull() }

    @TypeConverter fun roleTo(v: MessageRole?): String? = v?.name
    @TypeConverter fun roleFrom(v: String?): MessageRole? = v?.let { runCatching { MessageRole.valueOf(it) }.getOrNull() }

    @TypeConverter fun dlStateTo(v: DownloadState?): String? = v?.name
    @TypeConverter fun dlStateFrom(v: String?): DownloadState? = v?.let { runCatching { DownloadState.valueOf(it) }.getOrNull() }

    @TypeConverter fun rtStateTo(v: RuntimeState?): String? = v?.name
    @TypeConverter fun rtStateFrom(v: String?): RuntimeState? = v?.let { runCatching { RuntimeState.valueOf(it) }.getOrNull() }

    @TypeConverter fun predictionKindTo(v: PredictionKind?): String? = v?.name
    @TypeConverter fun predictionKindFrom(v: String?): PredictionKind? =
        v?.let { runCatching { PredictionKind.valueOf(it) }.getOrNull() }

    @TypeConverter fun attachmentRoleTo(v: AttachmentRole?): String? = v?.name
    @TypeConverter fun attachmentRoleFrom(v: String?): AttachmentRole? =
        v?.let { runCatching { AttachmentRole.valueOf(it) }.getOrNull() }
}

@Database(
    entities = [
        ModelEntity::class,
        ConversationEntity::class,
        MessageEntity::class,
        GeneratedImageEntity::class,
        TranscriptEntity::class,
        SynthesisEntity::class,
        DownloadJobEntity::class,
        RuntimeBundleEntity::class,
        ParamManifestEntity::class,
        McpServerEntity::class,
        PredictionRunEntity::class,
    ],
    version = DATABASE_VERSION,
    exportSchema = true,
)
@TypeConverters(Converters::class)
abstract class OnDeviceDatabase : RoomDatabase() {
    abstract fun models(): ModelDao
    abstract fun conversations(): ConversationDao
    abstract fun messages(): MessageDao
    abstract fun images(): GeneratedImageDao
    abstract fun transcripts(): TranscriptDao
    abstract fun syntheses(): SynthesisDao
    abstract fun downloads(): DownloadDao
    abstract fun runtimes(): RuntimeDao
    abstract fun manifests(): ParamManifestDao
    abstract fun mcpServers(): McpServerDao
    abstract fun predictionRuns(): PredictionRunDao

    /** Versions, with a guard so a bump stays deliberate. */
    companion object {
        const val NAME = "ondevice.db"

        /** One [Migration] per version step, in order, from 1 to [DATABASE_VERSION]. */
        val MIGRATIONS: Array<androidx.room.migration.Migration> =
            arrayOf(MIGRATION_1_2, MIGRATION_2_3, MIGRATION_3_4, MIGRATION_4_5, MIGRATION_5_6)
    }
}

/** v2 — `syntheses`. */
private val MIGRATION_1_2 = object : androidx.room.migration.Migration(1, 2) {
    override fun migrate(db: androidx.sqlite.db.SupportSQLiteDatabase) {
        db.execSQL(
            """
            CREATE TABLE IF NOT EXISTS `syntheses` (
                `id` TEXT NOT NULL,
                `path` TEXT NOT NULL,
                `text` TEXT NOT NULL,
                `engineId` TEXT NOT NULL,
                `modelId` TEXT,
                `voice` TEXT,
                `paramsJson` TEXT NOT NULL,
                `durationMillis` INTEGER NOT NULL,
                `sampleRate` INTEGER NOT NULL,
                `createdAt` INTEGER NOT NULL,
                PRIMARY KEY(`id`)
            )
            """.trimIndent(),
        )
        db.execSQL("CREATE INDEX IF NOT EXISTS `index_syntheses_createdAt` ON `syntheses` (`createdAt`)")
    }
}

/** v3 — which tools on an MCP server the user has switched off. */
private val MIGRATION_2_3 = object : androidx.room.migration.Migration(2, 3) {
    override fun migrate(db: androidx.sqlite.db.SupportSQLiteDatabase) {
        db.execSQL("ALTER TABLE `mcp_servers` ADD COLUMN `disabledToolsJson` TEXT NOT NULL DEFAULT '[]'")
    }
}

/** v4 — what each prediction cost. */
private val MIGRATION_3_4 = object : androidx.room.migration.Migration(3, 4) {
    override fun migrate(db: androidx.sqlite.db.SupportSQLiteDatabase) {
        db.execSQL(
            """
            CREATE TABLE IF NOT EXISTS `prediction_runs` (
                `id` TEXT NOT NULL,
                `kind` TEXT NOT NULL,
                `artifactId` TEXT NOT NULL,
                `modelId` TEXT,
                `backend` TEXT,
                `startedAt` INTEGER NOT NULL,
                `elapsedMillis` INTEGER NOT NULL,
                `peakCpuPercent` INTEGER NOT NULL,
                `meanCpuPercent` INTEGER NOT NULL,
                `peakRssBytes` INTEGER NOT NULL,
                `traceJson` TEXT NOT NULL,
                `statsJson` TEXT NOT NULL,
                PRIMARY KEY(`id`)
            )
            """.trimIndent(),
        )
        db.execSQL(
            "CREATE INDEX IF NOT EXISTS `index_prediction_runs_artifactId` " +
                "ON `prediction_runs` (`artifactId`)",
        )
        db.execSQL(
            "CREATE INDEX IF NOT EXISTS `index_prediction_runs_startedAt` " +
                "ON `prediction_runs` (`startedAt`)",
        )
    }
}

/** Drop `benchmarks`. */
private val MIGRATION_4_5 = object : androidx.room.migration.Migration(4, 5) {
    override fun migrate(db: androidx.sqlite.db.SupportSQLiteDatabase) {
        db.execSQL("DROP TABLE IF EXISTS `benchmarks`")
    }
}

/**
 * v6 — three things that described features nobody could reach.
 *
 * `runtime_bundles` loses the six columns that carried an install/update/
 * rollback story: the `.so` files are inside the APK, Android's W^X enforcement
 * refuses to load a native library from writable storage, and so an engine
 * update is an app update. Nothing ever wrote them. The table is recreated
 * rather than altered because ALTER TABLE DROP COLUMN needs SQLite 3.35 and so
 * Android 14, and minSdk here is 31; nothing is carried across because every
 * row is seeded from `runtimes.json` and an empty table seeds itself again.
 *
 * `presets` and `personas` go entirely. The preset picker was removed from chat
 * settings but its parameters kept being applied as the base layer under the
 * model's own overrides, so "Precise" went on quietly lowering the temperature
 * of a conversation nobody had set it on. Personas were never read into a
 * prompt at all.
 *
 * `conversations.presetId`, `conversations.personaId` and
 * `models.defaultPresetId` stay as inert columns: dropping those means
 * recreating the tables that hold every conversation and the whole model
 * library, which is not a trade worth making for three unread TEXT columns.
 */
private val MIGRATION_5_6 = object : androidx.room.migration.Migration(5, 6) {
    override fun migrate(db: androidx.sqlite.db.SupportSQLiteDatabase) {
        db.execSQL("DROP TABLE IF EXISTS `presets`")
        db.execSQL("DROP TABLE IF EXISTS `personas`")
        db.execSQL("DROP TABLE IF EXISTS `runtime_bundles`")
        db.execSQL(
            """
            CREATE TABLE IF NOT EXISTS `runtime_bundles` (
                `engine` TEXT NOT NULL,
                `buildTag` TEXT,
                `upstreamCommit` TEXT,
                `jniContract` INTEGER NOT NULL,
                `installedAt` INTEGER,
                `sizeBytes` INTEGER NOT NULL,
                `state` TEXT NOT NULL,
                `architectureCount` INTEGER NOT NULL,
                `backendsJson` TEXT NOT NULL,
                PRIMARY KEY(`engine`)
            )
            """.trimIndent(),
        )
    }
}

internal const val DATABASE_VERSION = 6
