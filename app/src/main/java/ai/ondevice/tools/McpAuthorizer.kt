package ai.ondevice.tools

import android.content.Context
import android.content.Intent
import android.net.Uri
import ai.ondevice.data.db.McpServerEntity
import ai.ondevice.data.db.OnDeviceDatabase
import ai.ondevice.data.secure.TokenStore
import kotlinx.coroutines.async
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withTimeoutOrNull
import okhttp3.OkHttpClient

/**
 * Signing in to an MCP server, start to finish.
 *
 * Holds the two halves that have to agree: the discovery-and-exchange steps in
 * [McpOAuth], and the bearer the client asks for on every request. Keeping them
 * together is what lets a refresh happen underneath a tool call rather than
 * surfacing as a 401 the model has to be told about.
 */
class McpAuthorizer(
    private val context: Context,
    private val db: OnDeviceDatabase,
    private val tokens: TokenStore,
    http: OkHttpClient,
) {

    private val oauth = McpOAuth(http)

    /**
     * One refresh at a time, per server.
     *
     * Two tool calls in the same turn both seeing an expired token would
     * otherwise both refresh, and a server that rotates refresh tokens
     * invalidates the first exchange when the second lands — logging the user
     * out as a direct result of having asked for two things at once.
     */
    private val refreshLocks = mutableMapOf<String, Mutex>()

    private fun lockFor(serverId: String): Mutex = synchronized(refreshLocks) {
        refreshLocks.getOrPut(serverId) { Mutex() }
    }

    /** Whether this server has been signed in to at all. */
    fun isAuthorized(serverId: String): Boolean = tokens.oauthTokens(serverId) != null

    /**
     * A usable access token, refreshed if it has expired, or null.
     *
     * Null means "not signed in" rather than "failed": the caller falls back to
     * whatever static header the server row carries, which is what an
     * unauthenticated or header-authenticated server wants.
     */
    suspend fun bearer(serverId: String): String? {
        val stored = tokens.oauthTokens(serverId) ?: return null
        val (access, refresh, expiresAt) = stored
        if (!OAuthTokens(access, refresh, expiresAt).isExpired()) return access
        if (refresh == null) return access

        return lockFor(serverId).withLock {
            // Re-read inside the lock: whoever held it may have just refreshed,
            // and refreshing again with a rotated token is how a session ends.
            val current = tokens.oauthTokens(serverId) ?: return@withLock null
            if (!OAuthTokens(current.first, current.second, current.third).isExpired()) {
                return@withLock current.first
            }
            val server = db.mcpServers().getAll().firstOrNull { it.id == serverId }
                ?: return@withLock current.first
            val authServer = server.authServer() ?: return@withLock current.first

            runCatching {
                oauth.refresh(
                    server = authServer,
                    clientId = server.oauthClientId ?: return@runCatching null,
                    clientSecret = server.oauthClientSecret,
                    refreshToken = current.second ?: return@runCatching null,
                    resource = McpOAuth.canonicalResource(server.url),
                )
            }.fold(
                onSuccess = { fresh ->
                    fresh?.let {
                        tokens.setOauthTokens(serverId, it.accessToken, it.refreshToken, it.expiresAt)
                        it.accessToken
                    } ?: current.first
                },
                onFailure = {
                    // The refresh token is spent or revoked. Dropping it turns
                    // the next request into a 401, which the screen shows as
                    // "sign in again" — which is the truth.
                    android.util.Log.i(TAG, "refresh failed for ${server.name}: ${it.message}")
                    tokens.clearOauthTokens(serverId)
                    null
                },
            )
        }
    }

    /**
     * Discover, register if needed, and open the browser.
     *
     * Returns once the redirect has come back and the tokens are stored, or
     * with a message explaining why it did not.
     */
    suspend fun authorize(server: McpServerEntity, challenge: String? = null): Result<Unit> =
        runCatching {
            val discovered = oauth.discover(server.url, challenge)

            // Written down before registration is attempted, not after.
            //
            // Storing them only on success meant a server that refused to
            // register left no trace that it had ever asked for a sign-in — so
            // the card offered no Authorize button, and the only way to try
            // again after pasting a registration token was to remove the
            // server and add it back. Discovery succeeding is itself the fact
            // worth keeping: it is what says this server does OAuth.
            db.mcpServers().upsert(
                server.copy(
                    oauthIssuer = discovered.issuer,
                    oauthAuthorizeEndpoint = discovered.authorizationEndpoint,
                    oauthTokenEndpoint = discovered.tokenEndpoint,
                    oauthRegistrationEndpoint = discovered.registrationEndpoint,
                    oauthScope = discovered.scopesSupported.takeIf { it.isNotEmpty() }
                        ?.joinToString(" "),
                ),
            )

            // Reuse the registration when there is one. Registering again on
            // every sign-in leaves a trail of dead client ids on the server and
            // gains nothing.
            val clientId = server.oauthClientId
            val clientSecret = server.oauthClientSecret
            val (id, secret) = if (clientId != null) {
                clientId to clientSecret
            } else {
                oauth.register(
                    server = discovered,
                    redirectUri = McpOAuth.REDIRECT_URI,
                    // Whatever was typed into the Authorization field. Before a
                    // sign-in there is nothing for it to authenticate *to*, so
                    // a server that wants a registration token is the one place
                    // it is useful — and the one place it would otherwise sit
                    // unused while registration failed with a bare 401.
                    initialAccessToken = server.authHeader
                        ?.trim()
                        ?.removePrefix("Bearer ")
                        ?.takeIf { it.isNotBlank() },
                )
            }

            // Re-read: the row was rewritten above, and copying from the stale
            // `server` here would put the endpoints back to what they were.
            val registered = db.mcpServers().getAll().firstOrNull { it.id == server.id } ?: server
            db.mcpServers().upsert(
                registered.copy(
                    oauthClientId = id,
                    oauthClientSecret = secret,
                    lastError = null,
                ),
            )

            val pending = oauth.begin(
                serverId = server.id,
                server = discovered,
                clientId = id,
                clientSecret = secret,
                redirectUri = McpOAuth.REDIRECT_URI,
                resource = McpOAuth.canonicalResource(server.url),
                scope = server.oauthScope,
            )

            // Listening starts before the browser opens. A fast redirect — an
            // already-signed-in session bounces straight back — can otherwise
            // arrive before the collector exists and be missed entirely.
            val redirect = coroutineScope {
                val waiting = async {
                    withTimeoutOrNull(AUTH_TIMEOUT_MILLIS) {
                        OAuthRedirects.results.first()
                    }
                }
                context.startActivity(
                    Intent(Intent.ACTION_VIEW, Uri.parse(pending.authorizeUrl))
                        .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK),
                )
                waiting.await()
            } ?: error("The sign-in was not completed.")

            when (redirect) {
                is OAuthRedirectResult.Failed ->
                    error(listOfNotNull(redirect.error, redirect.description).joinToString(": "))

                is OAuthRedirectResult.Code -> {
                    // The state is the only thing tying this redirect to this
                    // request. A mismatch means the code belongs to someone
                    // else's authorisation, and exchanging it would be the
                    // whole point of CSRF on an OAuth client.
                    if (redirect.state != pending.state) {
                        error("The sign-in came back with the wrong state and was not used.")
                    }
                    val issued = oauth.exchange(pending, redirect.code, McpOAuth.REDIRECT_URI)
                    tokens.setOauthTokens(
                        server.id,
                        issued.accessToken,
                        issued.refreshToken,
                        issued.expiresAt,
                    )
                    android.util.Log.i(TAG, "authorized ${server.name} against ${discovered.issuer}")
                }
            }
        }

    /** Forget the tokens without forgetting the server or its registration. */
    fun signOut(serverId: String) = tokens.clearOauthTokens(serverId)

    private companion object {
        const val TAG = "McpAuthorizer"

        /** Long enough to find a password; short enough not to wait forever. */
        const val AUTH_TIMEOUT_MILLIS = 5 * 60 * 1000L
    }
}

/** The stored endpoints as an [AuthServer], or null when discovery has not run. */
fun McpServerEntity.authServer(): AuthServer? {
    val authorize = oauthAuthorizeEndpoint ?: return null
    val token = oauthTokenEndpoint ?: return null
    return AuthServer(
        issuer = oauthIssuer.orEmpty(),
        authorizationEndpoint = authorize,
        tokenEndpoint = token,
        registrationEndpoint = oauthRegistrationEndpoint,
    )
}
