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
import ai.ondevice.core.PredictionKind
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
        PresetEntity::class,
        PersonaEntity::class,
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
    abstract fun presets(): PresetDao
    abstract fun personas(): PersonaDao
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

    /**
     * Versions, with a guard so a bump stays deliberate.
     *
     * The schema had reached v4 through three migrations, each written to carry
     * an installed base that does not exist: this app has never shipped, and the
     * only databases in the world are on development devices that are wiped
     * whenever the model set changes anyway. Three migration scripts, three
     * exported schema files and a backfill whose correctness nobody could check
     * against real data — all of it maintenance for a population of zero. So the
     * schema was restated once, at v1, and grows from there.
     *
     * The danger was never v1. It was v2: a destructive fallback that is correct
     * today reads exactly the same on the day someone's conversations are in
     * here, and nothing would have failed to warn about it. The fallback is now
     * debug-only, so a version bump without a matching entry in [MIGRATIONS]
     * throws on a release build instead of quietly emptying the database.
     * `OnDeviceDatabaseMigrationTest` fails sooner still, at compile-and-test
     * time, which is where this should be caught.
     */
    companion object {
        const val NAME = "ondevice.db"

        /**
         * One [Migration] per version step, in order, from 1 to
         * [DATABASE_VERSION]. Empty while the schema has never changed.
         *
         * Adding a migration is not optional once the app has shipped: Room only
         * consults this list, and anything it cannot find a path for is a
         * refusal on release builds.
         */
        val MIGRATIONS: Array<androidx.room.migration.Migration> =
            arrayOf(MIGRATION_1_2, MIGRATION_2_3, MIGRATION_3_4, MIGRATION_4_5)
    }
}

/**
 * v2 — `syntheses`.
 *
 * Speak already wrote its WAV to disk; nothing recorded that it had. A new table
 * only, so there is no data to carry and nothing to get wrong: the migration
 * creates it and the index Room expects, and every existing row in every other
 * table is untouched.
 */
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

/**
 * v3 — which tools on an MCP server the user has switched off.
 *
 * A column with a default, so existing rows land on "nothing disabled", which
 * is what they meant before the column existed. `lastToolsJson` changes meaning
 * in the same release — from a comma-joined list of names to a JSON array of
 * name and description — but its *type* does not, and the reader accepts both,
 * so there is nothing to rewrite here.
 */
private val MIGRATION_2_3 = object : androidx.room.migration.Migration(2, 3) {
    override fun migrate(db: androidx.sqlite.db.SupportSQLiteDatabase) {
        db.execSQL("ALTER TABLE `mcp_servers` ADD COLUMN `disabledToolsJson` TEXT NOT NULL DEFAULT '[]'")
    }
}

/**
 * v4 — what each prediction cost.
 *
 * A new table only, so nothing existing is touched and there is no backfill to
 * get wrong: runs recorded from this version on have traces, and everything
 * generated before simply has none, which is the truth. The two indices match
 * what Room derives from the entity's `@Index` declarations, and they have to,
 * or Room's own schema validation rejects the migrated database on open.
 */
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

/**
 * Drop `benchmarks`.
 *
 * The table stored one row per model per backend, and llama.cpp registers one
 * backend in this build — so every row it ever held was a measurement of CPU
 * against itself, and the selection it fed then picked the only candidate. The
 * rows are not worth carrying and there is nothing to migrate them into.
 *
 * `DROP TABLE IF EXISTS` rather than `DROP TABLE`, because a database created at
 * v4 by a build that had already been rebuilt without the entity has no such
 * table, and a migration that fails on a database it was meant to repair is
 * worse than the schema drift it fixes.
 */
private val MIGRATION_4_5 = object : androidx.room.migration.Migration(4, 5) {
    override fun migrate(db: androidx.sqlite.db.SupportSQLiteDatabase) {
        db.execSQL("DROP TABLE IF EXISTS `benchmarks`")
    }
}

/**
 * Named rather than written into the annotation so a test can compare it with
 * [OnDeviceDatabase.MIGRATIONS] — an annotation argument is not readable at
 * runtime, and the pair only means anything when they are checked against each
 * other.
 */
internal const val DATABASE_VERSION = 5
