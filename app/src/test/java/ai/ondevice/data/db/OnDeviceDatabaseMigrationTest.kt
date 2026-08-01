package ai.ondevice.data.db

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/** That the declared schema version and the migration list agree. */
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

    /** That schema export is actually configured and producing files. */
    @Test
    fun `schema export is configured`() {
        val exported = java.io.File("schemas/ai.ondevice.data.db.OnDeviceDatabase")
        // Gradle runs unit tests with the module directory as the working directory.
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
