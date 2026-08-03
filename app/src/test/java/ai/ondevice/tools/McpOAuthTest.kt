package ai.ondevice.tools

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The two OAuth rules a token's usefulness actually turns on.
 *
 * The audience is compared literally by the server, so a trailing slash is a
 * rejected token rather than a near miss; and the discovery pointer decides
 * which authorization server is asked at all. Both are string handling, both
 * are easy to get subtly wrong, and neither shows up until a real server says
 * no for a reason it does not explain.
 */
class McpOAuthTest {

    @Test
    fun `the resource is the server url without query or fragment`() {
        assertEquals(
            "https://mcp.example.com/v1",
            McpOAuth.canonicalResource("https://mcp.example.com/v1?session=9#frag"),
        )
    }

    @Test
    fun `a trailing slash is not part of the resource`() {
        // RFC 8707 comparison is literal, so these have to agree with whatever
        // the server derives from its own URL — and a stray slash is the
        // difference between a token that works and a 401 with no explanation.
        assertEquals(
            McpOAuth.canonicalResource("https://mcp.example.com/v1"),
            McpOAuth.canonicalResource("https://mcp.example.com/v1/"),
        )
    }

    @Test
    fun `a bare origin has no path`() {
        assertEquals("https://mcp.example.com", McpOAuth.canonicalResource("https://mcp.example.com"))
        assertEquals("https://mcp.example.com", McpOAuth.canonicalResource("https://mcp.example.com/"))
    }

    @Test
    fun `a port stays part of the origin`() {
        assertEquals(
            "http://localhost:3000/mcp",
            McpOAuth.canonicalResource("http://localhost:3000/mcp"),
        )
    }

    @Test
    fun `the challenge decides where the metadata is read from`() {
        val oauth = McpOAuth(McpToolProvider.httpClient())
        val header =
            """Bearer realm="mcp", resource_metadata="https://auth.example.com/.well-known/oauth-protected-resource""""
        assertEquals(
            "https://auth.example.com/.well-known/oauth-protected-resource",
            oauth.resourceMetadataUrl(header, "https://mcp.example.com/v1"),
        )
    }

    @Test
    fun `without a challenge the well-known path is derived from the server`() {
        val oauth = McpOAuth(McpToolProvider.httpClient())
        // Derived from the origin, not from the full URL: the document lives at
        // the root of the host, so keeping "/v1" would ask for a path that does
        // not exist and fall through to treating the server as its own issuer.
        assertEquals(
            "https://mcp.example.com/.well-known/oauth-protected-resource",
            oauth.resourceMetadataUrl(null, "https://mcp.example.com/v1"),
        )
        assertEquals(
            "https://mcp.example.com/.well-known/oauth-protected-resource",
            oauth.resourceMetadataUrl("Bearer realm=\"mcp\"", "https://mcp.example.com/v1"),
        )
    }

    @Test
    fun `expiry is judged a minute early`() {
        val now = 1_000_000L
        // Still a minute and a half of life: usable.
        assertFalse(OAuthTokens("a", null, now + 90_000).isExpired(now))
        // Half a minute left, which is inside the margin — refreshed now rather
        // than failing a request that is already in flight.
        assertTrue(OAuthTokens("a", null, now + 30_000).isExpired(now))
        // No stated expiry means the server did not say, and guessing would
        // throw away a working token.
        assertFalse(OAuthTokens("a", null, null).isExpired(now))
    }
}
