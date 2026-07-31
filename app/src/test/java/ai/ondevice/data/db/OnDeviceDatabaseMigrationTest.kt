package ai.ondevice.data.db

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * That the declared schema version and the migration list agree.
 *
 * The failure this exists for is quiet. Bumping [DATABASE_VERSION] to add a
 * column, without adding the matching [OnDeviceDatabase.MIGRATIONS] entry, is a
 * one-line change that compiles, passes every other test, and works perfectly on
 * the developer's device — because debug builds recreate the database. It is
 * only wrong on someone else's phone, where it either wipes their conversations
 * or refuses to start, and by then the change has shipped.
 *
 * So the check happens here, where a version bump and a missing migration cannot
 * be separated.
 */
class OnDeviceDatabaseMigrationTest {

    private val migrations = OnDeviceDatabase.MIGRATIONS

    @Test
    fun `there is a migration for every version step`() {
        assertEquals(
            "Version $DATABASE_VERSION needs ${DATABASE_VERSION - 1} migrations to reach it " +
                "from 1, and ${migrations.size} are declared. Add the missing Migration to " +
                "OnDeviceDatabase.MIGRATIONS, and commit the exported schema JSON alongside it.",
            DATABASE_VERSION - 1,
            migrations.size,
        )
    }

    @Test
    fun `the migrations form one unbroken chain`() {
        var version = 1
        migrations.sortedBy { it.startVersion }.forEach { migration ->
            assertEquals(
                "Migration ${migration.startVersion} to ${migration.endVersion} leaves a gap; " +
                    "nothing upgrades a database sitting at version $version.",
                version,
                migration.startVersion,
            )
            assertTrue(
                "Migration ${migration.startVersion} to ${migration.endVersion} does not move " +
                    "forwards.",
                migration.endVersion > migration.startVersion,
            )
            version = migration.endVersion
        }
        assertEquals(
            "The migration chain stops at $version but the schema declares $DATABASE_VERSION.",
            DATABASE_VERSION,
            version,
        )
    }

    /**
     * That schema export is actually configured and producing files.
     *
     * Narrower than it first looks, and the limit is worth stating: Room writes
     * the JSON during the build that precedes this test, so the file is always
     * on disk by the time the assertion runs. Bumping the version and *not
     * committing* the new JSON therefore passes here — verified by bumping the
     * version and watching the other two tests fail while this one did not.
     *
     * What it does catch is `room.schemaLocation` being dropped or repointed, at
     * which point no schema is exported at all and every future migration has
     * nothing to be diffed against. That is worth a test; catching the uncommitted
     * file is a job for review, not for this.
     */
    @Test
    fun `schema export is configured`() {
        val exported = java.io.File("schemas/ai.ondevice.data.db.OnDeviceDatabase")
        // Gradle runs unit tests with the module directory as the working
        // directory. If that ever stops being true, say so rather than passing
        // by accident — a guard that silently disables itself is worse than none.
        assertTrue(
            "Cannot find ${exported.absolutePath}. Room's schemaLocation is set in " +
                "build.gradle.kts; if this test moved, fix the path rather than deleting it.",
            exported.isDirectory,
        )
        assertTrue(
            "No $DATABASE_VERSION.json in ${exported.absolutePath}. Build once to export it, " +
                "then commit it — it is the only record of what this version looked like.",
            java.io.File(exported, "$DATABASE_VERSION.json").isFile,
        )
    }
}
