package ai.ondevice.tools

import android.net.Uri
import android.util.Base64
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.add
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.put
import okhttp3.FormBody
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import java.security.MessageDigest
import java.security.SecureRandom

/**
 * Where a server's authorisation actually lives, once it has been discovered.
 *
 * Discovered rather than configured: MCP's whole point here is that you paste a
 * URL and the server says where to send you. Asking someone to find an
 * authorize endpoint by hand would be asking them to do what the 401 already
 * answers.
 */
data class AuthServer(
    val issuer: String,
    val authorizationEndpoint: String,
    val tokenEndpoint: String,
    val registrationEndpoint: String?,
    val scopesSupported: List<String> = emptyList(),
)

/** What a token exchange came back with. */
data class OAuthTokens(
    val accessToken: String,
    val refreshToken: String?,
    /** Absolute, in epoch millis; null when the server did not say. */
    val expiresAt: Long?,
) {
    /**
     * Treated as expired a minute early.
     *
     * A token that expires while the request is in flight fails the call and
     * costs a round trip to find out, so the refresh happens on the near side
     * of the boundary rather than the far one.
     */
    fun isExpired(now: Long = System.currentTimeMillis()): Boolean =
        expiresAt != null && now >= expiresAt - EXPIRY_MARGIN_MILLIS

    private companion object {
        const val EXPIRY_MARGIN_MILLIS = 60_000L
    }
}

/** One authorisation in progress, held between opening the browser and the redirect. */
data class PendingAuthorization(
    val serverId: String,
    val server: AuthServer,
    val clientId: String,
    val clientSecret: String?,
    val codeVerifier: String,
    val state: String,
    val resource: String,
    val authorizeUrl: String,
)

/**
 * OAuth for MCP servers, as the 2025-06-18 revision describes it.
 *
 * Four steps, none of which the user types: an unauthorised request comes back
 * 401 with a pointer to the protected-resource metadata; that names the
 * authorization server; the authorization server's own metadata names its
 * endpoints; and if it accepts dynamic registration this app registers itself
 * on the spot. Only then does a browser open.
 *
 * PKCE is not optional here and there is no client secret to fall back on: a
 * public client on a phone cannot keep one, and the redirect comes back through
 * a custom scheme that another app could in principle claim. The verifier never
 * leaves the device until it is exchanged, so a stolen code is worth nothing on
 * its own.
 */
class McpOAuth(private val http: OkHttpClient) {

    private val json = Json { ignoreUnknownKeys = true; isLenient = true }

    /**
     * The metadata URL a 401 pointed at, if it did.
     *
     * RFC 9728 puts it in the `WWW-Authenticate` header as
     * `resource_metadata="…"`. A server that omits it is not broken — the
     * well-known path is derived from the server URL instead — but taking it
     * when offered is what lets a resource live somewhere other than its own
     * origin.
     */
    fun resourceMetadataUrl(wwwAuthenticate: String?, serverUrl: String): String {
        wwwAuthenticate?.let { header ->
            RESOURCE_METADATA.find(header)?.groupValues?.get(1)?.takeIf { it.isNotBlank() }
                ?.let { return it }
        }
        return "${origin(serverUrl)}/.well-known/oauth-protected-resource"
    }

    /** Discover which authorization server guards [serverUrl], and where its endpoints are. */
    suspend fun discover(serverUrl: String, wwwAuthenticate: String?): AuthServer =
        withContext(Dispatchers.IO) {
            val issuer = authorizationServerFor(serverUrl, wwwAuthenticate)
            metadataFor(issuer)
        }

    private fun authorizationServerFor(serverUrl: String, wwwAuthenticate: String?): String {
        val metadataUrl = resourceMetadataUrl(wwwAuthenticate, serverUrl)
        val body = getJson(metadataUrl)
        if (body != null) {
            body["authorization_servers"]?.jsonArray?.firstOrNull()?.jsonPrimitive?.content
                ?.takeIf { it.isNotBlank() }
                ?.let { return it }
        }
        // No protected-resource document. Plenty of servers are their own
        // authorization server, so the origin is the next thing to try rather
        // than a dead end.
        return origin(serverUrl)
    }

    /**
     * The authorization server's endpoints.
     *
     * Both well-known paths are tried. RFC 8414 inserts its segment after the
     * host and before any path, while OpenID Connect appends its own to the
     * end, and real deployments are split between them — so a client that knows
     * only one of the two fails against half the servers it meets.
     */
    private fun metadataFor(issuer: String): AuthServer {
        val trimmed = issuer.trimEnd('/')
        val path = pathOf(trimmed).trimEnd('/')
        val origin = origin(trimmed)

        val candidates = listOf(
            "$origin/.well-known/oauth-authorization-server$path",
            "$trimmed/.well-known/openid-configuration",
            "$origin/.well-known/openid-configuration$path",
        )

        candidates.forEach { candidate ->
            val body = getJson(candidate) ?: return@forEach
            val authorize = body["authorization_endpoint"]?.jsonPrimitive?.content
            val token = body["token_endpoint"]?.jsonPrimitive?.content
            if (!authorize.isNullOrBlank() && !token.isNullOrBlank()) {
                return AuthServer(
                    issuer = body["issuer"]?.jsonPrimitive?.content ?: trimmed,
                    authorizationEndpoint = authorize,
                    tokenEndpoint = token,
                    registrationEndpoint = body["registration_endpoint"]?.jsonPrimitive?.content,
                    scopesSupported = body["scopes_supported"]?.jsonArray
                        ?.mapNotNull { it.jsonPrimitive.content }.orEmpty(),
                )
            }
        }
        error(
            "$trimmed did not publish OAuth metadata at either well-known path, so there is " +
                "nowhere to send you to sign in.",
        )
    }

    /**
     * Register this app with the authorization server (RFC 7591).
     *
     * There is no other way for an app that ships to strangers to have a client
     * id for a server it has never met. Servers that do not offer it need one
     * issued out of band, and this says so rather than failing at the redirect
     * with something less legible.
     */
    suspend fun register(
        server: AuthServer,
        redirectUri: String,
        /**
         * RFC 7591 §3's initial access token, when the endpoint wants one.
         *
         * Registration is allowed to be protected, and deployments that guard
         * it answer 401 to an anonymous POST — which looks like the app being
         * broken when it is the endpoint being closed. Passing whatever the
         * server row carries in its Authorization field turns that into
         * something the person can act on: paste the token their
         * administrator gave them and registration goes through.
         */
        initialAccessToken: String? = null,
    ): Pair<String, String?> =
        withContext(Dispatchers.IO) {
            val endpoint = server.registrationEndpoint
                ?: error(
                    "${server.issuer} does not offer dynamic client registration, so it needs a " +
                        "client id issued to you. Paste it as an Authorization header instead.",
                )

            val body = buildJsonObject {
                put("client_name", CLIENT_NAME)
                put("redirect_uris", buildJsonArray { add(redirectUri) })
                put("grant_types", buildJsonArray { add("authorization_code"); add("refresh_token") })
                put("response_types", buildJsonArray { add("code") })
                // A phone app cannot keep a secret, and saying so is what makes
                // the server enforce PKCE rather than expect one.
                put("token_endpoint_auth_method", "none")
                put("application_type", "native")
            }

            val response = runCatching {
                post(endpoint, body.toString().toRequestBody(JSON_MEDIA), initialAccessToken)
            }.getOrElse { failure ->
                val unauthorized = failure.message.orEmpty().let {
                    it.contains("401") || it.contains("Unauthorized", ignoreCase = true) ||
                        it.contains("invalid_token") || it.contains("Invalid credentials", ignoreCase = true)
                }
                if (unauthorized && initialAccessToken == null) {
                    error(
                        "$endpoint refused the registration because nothing was sent to " +
                            "authenticate it. This server does not let apps register themselves, " +
                            "so it needs either a client ID issued to you or a registration " +
                            "token — fill in one of the two fields above, then press Authorize.",
                    )
                }
                throw failure
            }
            val clientId = response["client_id"]?.jsonPrimitive?.content
                ?: error("The registration came back with no client_id.")
            clientId to response["client_secret"]?.jsonPrimitive?.content
        }

    /** Everything needed to open a browser, and everything needed to finish afterwards. */
    fun begin(
        serverId: String,
        server: AuthServer,
        clientId: String,
        clientSecret: String?,
        redirectUri: String,
        resource: String,
        scope: String?,
    ): PendingAuthorization {
        val verifier = randomUrlSafe(VERIFIER_BYTES)
        val state = randomUrlSafe(STATE_BYTES)
        val challenge = base64Url(
            MessageDigest.getInstance("SHA-256").digest(verifier.toByteArray(Charsets.US_ASCII)),
        )

        val url = Uri.parse(server.authorizationEndpoint).buildUpon()
            .appendQueryParameter("response_type", "code")
            .appendQueryParameter("client_id", clientId)
            .appendQueryParameter("redirect_uri", redirectUri)
            .appendQueryParameter("code_challenge", challenge)
            .appendQueryParameter("code_challenge_method", "S256")
            .appendQueryParameter("state", state)
            // RFC 8707. Without it a server that guards several resources
            // issues a token good for all of them, which is more than this
            // conversation needs and more than it should be able to lose.
            .appendQueryParameter("resource", resource)
            .apply {
                val wanted = scope?.takeIf { it.isNotBlank() }
                    ?: server.scopesSupported.takeIf { it.isNotEmpty() }?.joinToString(" ")
                wanted?.let { appendQueryParameter("scope", it) }
            }
            .build()
            .toString()

        return PendingAuthorization(
            serverId = serverId,
            server = server,
            clientId = clientId,
            clientSecret = clientSecret,
            codeVerifier = verifier,
            state = state,
            resource = resource,
            authorizeUrl = url,
        )
    }

    /** Trade the code the redirect carried for tokens. */
    suspend fun exchange(
        pending: PendingAuthorization,
        code: String,
        redirectUri: String,
    ): OAuthTokens = withContext(Dispatchers.IO) {
        val form = FormBody.Builder()
            .add("grant_type", "authorization_code")
            .add("code", code)
            .add("redirect_uri", redirectUri)
            .add("client_id", pending.clientId)
            .add("code_verifier", pending.codeVerifier)
            .add("resource", pending.resource)
            .apply { pending.clientSecret?.let { add("client_secret", it) } }
            .build()
        tokensFrom(post(pending.server.tokenEndpoint, form, null))
    }

    /** A new access token from a refresh token, without sending anyone back to a browser. */
    suspend fun refresh(
        server: AuthServer,
        clientId: String,
        clientSecret: String?,
        refreshToken: String,
        resource: String,
    ): OAuthTokens = withContext(Dispatchers.IO) {
        val form = FormBody.Builder()
            .add("grant_type", "refresh_token")
            .add("refresh_token", refreshToken)
            .add("client_id", clientId)
            .add("resource", resource)
            .apply { clientSecret?.let { add("client_secret", it) } }
            .build()
        val tokens = tokensFrom(post(server.tokenEndpoint, form, null))
        // Rotation is optional, and a server that does not rotate sends no
        // refresh_token back. Dropping the old one on that reply would end the
        // session at the next expiry for no reason.
        if (tokens.refreshToken == null) tokens.copy(refreshToken = refreshToken) else tokens
    }

    private fun tokensFrom(body: JsonObject): OAuthTokens {
        val access = body["access_token"]?.jsonPrimitive?.content
            ?: error("The token response carried no access_token.")
        val expiresIn = body["expires_in"]?.jsonPrimitive?.content?.toLongOrNull()
        return OAuthTokens(
            accessToken = access,
            refreshToken = body["refresh_token"]?.jsonPrimitive?.content,
            expiresAt = expiresIn?.let { System.currentTimeMillis() + it * 1000 },
        )
    }

    // — http —

    private fun getJson(url: String): JsonObject? = runCatching {
        val request = Request.Builder().url(url)
            .header("Accept", "application/json")
            .header("MCP-Protocol-Version", McpToolProvider.PROTOCOL_VERSION)
            .build()
        http.newCall(request).execute().use { response ->
            if (!response.isSuccessful) return@use null
            json.parseToJsonElement(response.body?.string().orEmpty()).jsonObject
        }
    }.getOrNull()

    private fun post(url: String, body: okhttp3.RequestBody, bearer: String?): JsonObject {
        val request = Request.Builder().url(url).post(body)
            .header("Accept", "application/json")
            .apply { bearer?.let { header("Authorization", "Bearer $it") } }
            .build()
        http.newCall(request).execute().use { response ->
            val raw = response.body?.string().orEmpty()
            if (!response.isSuccessful) {
                // OAuth error bodies are the useful half of a failure — the
                // code says invalid_grant, invalid_client, and so on — so it is
                // reported rather than the status alone.
                val described = runCatching {
                    val problem = json.parseToJsonElement(raw).jsonObject
                    listOfNotNull(
                        problem["error"]?.jsonPrimitive?.content,
                        problem["error_description"]?.jsonPrimitive?.content,
                    ).joinToString(": ").takeIf { it.isNotBlank() }
                }.getOrNull()
                error(described ?: "HTTP ${response.code} from $url: ${raw.take(200)}")
            }
            return json.parseToJsonElement(raw).jsonObject
        }
    }

    companion object {
        /** What the app calls itself when it registers. */
        const val CLIENT_NAME = "On-device AI"

        /**
         * Where the browser comes back to.
         *
         * A custom scheme rather than an https App Link because an App Link
         * needs a domain this app can publish `assetlinks.json` on, and it has
         * none — there is no server anywhere in this project.
         */
        const val REDIRECT_URI = "ai.ondevice://oauth/callback"

        private val JSON_MEDIA = "application/json".toMediaType()
        private val RESOURCE_METADATA = Regex("""resource_metadata="([^"]+)"""")

        /** RFC 7636 allows 43–128 characters; 64 bytes lands in the middle. */
        private const val VERIFIER_BYTES = 48
        private const val STATE_BYTES = 24

        /**
         * The token's audience: the server URL, with query and fragment gone.
         *
         * RFC 8707 calls this the canonical resource URI and servers compare it
         * literally, so a trailing difference is a rejected token rather than a
         * near miss.
         */
        fun canonicalResource(serverUrl: String): String {
            val path = pathOf(serverUrl).trimEnd('/').takeIf { it.isNotEmpty() }.orEmpty()
            return "${origin(serverUrl)}$path"
        }

        /**
         * Scheme and authority, using [java.net.URI] rather than android.net.
         *
         * Not a style choice: android.net.Uri is a stub in JVM unit tests and
         * every call throws, so the two rules a token's audience depends on
         * would be the two rules with no tests.
         */
        fun origin(url: String): String {
            val uri = java.net.URI(url.trim())
            return "${uri.scheme}://${uri.authority}"
        }

        fun pathOf(url: String): String = java.net.URI(url.trim()).path.orEmpty()

        private fun randomUrlSafe(bytes: Int): String =
            base64Url(ByteArray(bytes).also { SecureRandom().nextBytes(it) })

        private fun base64Url(bytes: ByteArray): String =
            Base64.encodeToString(bytes, Base64.URL_SAFE or Base64.NO_PADDING or Base64.NO_WRAP)
    }
}
