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
        GeneratedClipEntity::class,
        TranscriptEntity::class,
        SynthesisEntity::class,
        DownloadJobEntity::class,
        ParamManifestEntity::class,
        McpServerEntity::class,
        PredictionRunEntity::class,
        WorkflowEntity::class,
        WorkflowRunEntity::class,
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
    abstract fun clips(): GeneratedClipDao
    abstract fun transcripts(): TranscriptDao
    abstract fun syntheses(): SynthesisDao
    abstract fun downloads(): DownloadDao
    abstract fun manifests(): ParamManifestDao
    abstract fun mcpServers(): McpServerDao
    abstract fun predictionRuns(): PredictionRunDao

    abstract fun workflows(): WorkflowDao

    /** Versions, with a guard so a bump stays deliberate. */
    companion object {
        const val NAME = "ondevice.db"

        /** One [Migration] per version step, in order, from 1 to [DATABASE_VERSION]. */
        val MIGRATIONS: Array<androidx.room.migration.Migration> =
            arrayOf(
                MIGRATION_1_2, MIGRATION_2_3, MIGRATION_3_4, MIGRATION_4_5,
                MIGRATION_5_6, MIGRATION_6_7, MIGRATION_7_8, MIGRATION_8_9,
                MIGRATION_9_10, MIGRATION_10_11, MIGRATION_11_12, MIGRATION_12_13,
                MIGRATION_13_14,
            )
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

/**
 * Clear the architectures that were never architectures.
 *
 * Kokoro's espeak language list lived in the same manifest field as every other
 * runtime's architecture list, so `en`, `es`, `fr`, `hi`, `it` and `pt` were all
 * things the resolver believed a model could be built on. When a GGUF header
 * carries no architecture the resolver infers one from the repo's tags, and an
 * ordinary Hugging Face language tag then matched: FLUX.2 Klein and Real-ESRGAN
 * were both stored as architecture `en`.
 *
 * None of the six is a real architecture in llama.cpp's 138 or sd.cpp's 46, so
 * clearing them is unambiguous. Null means "not known yet", which is what it
 * always meant, and the next resolve fills it in properly.
 */
private val MIGRATION_6_7 = object : androidx.room.migration.Migration(6, 7) {
    override fun migrate(db: androidx.sqlite.db.SupportSQLiteDatabase) {
        db.execSQL(
            "UPDATE `models` SET `architecture` = NULL " +
                "WHERE `architecture` IN ('en', 'es', 'fr', 'hi', 'it', 'pt')",
        )
    }
}

private val MIGRATION_7_8 = object : androidx.room.migration.Migration(7, 8) {
    override fun migrate(db: androidx.sqlite.db.SupportSQLiteDatabase) {
        // Null until somebody types one; nothing to backfill, because the name
        // this replaces is still there to fall back to.
        db.execSQL("ALTER TABLE `models` ADD COLUMN `customLabel` TEXT")
    }
}

private val MIGRATION_8_9 = object : androidx.room.migration.Migration(8, 9) {
    override fun migrate(db: androidx.sqlite.db.SupportSQLiteDatabase) {
        // Null everywhere, and deliberately so: only a load can answer it, and
        // no load has happened yet under a build that records the answer.
        db.execSQL("ALTER TABLE `models` ADD COLUMN `selfContained` INTEGER")
    }
}

/** v10 — `generated_clips`, the video counterpart of `generated_images`. */
private val MIGRATION_9_10 = object : androidx.room.migration.Migration(9, 10) {
    override fun migrate(db: androidx.sqlite.db.SupportSQLiteDatabase) {
        db.execSQL(
            """
            CREATE TABLE IF NOT EXISTS `generated_clips` (
                `id` TEXT NOT NULL,
                `directory` TEXT NOT NULL,
                `frameCount` INTEGER NOT NULL,
                `prompt` TEXT NOT NULL,
                `negativePrompt` TEXT,
                `paramsJson` TEXT NOT NULL,
                `modelId` TEXT,
                `seed` INTEGER NOT NULL,
                `width` INTEGER NOT NULL,
                `height` INTEGER NOT NULL,
                `fps` INTEGER NOT NULL,
                `audioPath` TEXT,
                `createdAt` INTEGER NOT NULL,
                PRIMARY KEY(`id`)
            )
            """.trimIndent(),
        )
        db.execSQL(
            "CREATE INDEX IF NOT EXISTS `index_generated_clips_createdAt` ON `generated_clips` (`createdAt`)",
        )
    }
}

internal const val DATABASE_VERSION = 14

/**
 * v11 — OAuth on `mcp_servers`.
 *
 * Columns only, and every one nullable: a server added before this release is
 * still a working server with a static header, and it must stay one. Nothing
 * here is a secret; the tokens go to TokenStore.
 */
private val MIGRATION_10_11 = object : androidx.room.migration.Migration(10, 11) {
    override fun migrate(db: androidx.sqlite.db.SupportSQLiteDatabase) {
        listOf(
            "oauthIssuer", "oauthAuthorizeEndpoint", "oauthTokenEndpoint",
            "oauthRegistrationEndpoint", "oauthClientId", "oauthClientSecret", "oauthScope",
        ).forEach { column ->
            db.execSQL("ALTER TABLE `mcp_servers` ADD COLUMN `$column` TEXT")
        }
    }
}

/**
 * v12 — `runtime_bundles` goes.
 *
 * The table mirrored the generated runtime manifest at first launch and nothing
 * ever wrote to it again: the engines are compiled into the APK, so an engine
 * update is an app update. The screen that read it said as much and offered
 * nothing to press. The one live question — can this build load a diffusion
 * model — is now asked of the library itself, which is the only thing that
 * actually knows.
 */
private val MIGRATION_11_12 = object : androidx.room.migration.Migration(11, 12) {
    override fun migrate(db: androidx.sqlite.db.SupportSQLiteDatabase) {
        db.execSQL("DROP TABLE IF EXISTS `runtime_bundles`")
    }
}

/**
 * v13 — `parameterCount` on `models`.
 *
 * Nullable, and left null for every row already there. It is what tells two
 * files apart when nothing else does: the two T5-XXLs share a name, an
 * architecture and a slot, and differ by the size of a vocabulary. A row
 * installed before this column existed keeps a null, and the readers treat
 * that as "not known" rather than "does not match" — so nothing that worked
 * yesterday starts refusing today.
 */
private val MIGRATION_12_13 = object : androidx.room.migration.Migration(12, 13) {
    override fun migrate(db: androidx.sqlite.db.SupportSQLiteDatabase) {
        db.execSQL("ALTER TABLE `models` ADD COLUMN `parameterCount` INTEGER")
    }
}

/**
 * Workflows, and the runs they produced.
 *
 * Additive only: two new tables and their indices, nothing existing touched.
 * A migration that rewrites a table the user's whole history lives in is a
 * risk worth taking only when there is no other way, and here there is.
 */
private val MIGRATION_13_14 = object : androidx.room.migration.Migration(13, 14) {
    override fun migrate(db: androidx.sqlite.db.SupportSQLiteDatabase) {
        db.execSQL(
            """
            CREATE TABLE IF NOT EXISTS `workflows` (
                `id` TEXT NOT NULL,
                `name` TEXT NOT NULL,
                `notes` TEXT NOT NULL,
                `graphJson` TEXT NOT NULL,
                `createdAt` INTEGER NOT NULL,
                `updatedAt` INTEGER NOT NULL,
                `lastRunAt` INTEGER,
                PRIMARY KEY(`id`)
            )
            """.trimIndent(),
        )
        db.execSQL("CREATE INDEX IF NOT EXISTS `index_workflows_updatedAt` ON `workflows` (`updatedAt`)")
        db.execSQL(
            """
            CREATE TABLE IF NOT EXISTS `workflow_runs` (
                `id` TEXT NOT NULL,
                `workflowId` TEXT NOT NULL,
                `graphJson` TEXT NOT NULL,
                `state` TEXT NOT NULL,
                `startedAt` INTEGER NOT NULL,
                `finishedAt` INTEGER,
                `error` TEXT,
                `errorHint` TEXT,
                `nodeStatesJson` TEXT NOT NULL,
                PRIMARY KEY(`id`)
            )
            """.trimIndent(),
        )
        db.execSQL("CREATE INDEX IF NOT EXISTS `index_workflow_runs_workflowId` ON `workflow_runs` (`workflowId`)")
        db.execSQL("CREATE INDEX IF NOT EXISTS `index_workflow_runs_startedAt` ON `workflow_runs` (`startedAt`)")
    }
}
