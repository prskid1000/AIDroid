package ai.ondevice.data.hf

import ai.ondevice.data.secure.TokenStore
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.Json
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.IOException

/** The two endpoints SPEC §3.1 verified, plus search. */
class HfApi(
    private val client: OkHttpClient,
    private val tokens: TokenStore,
) {

    private val json = Json {
        ignoreUnknownKeys = true
        isLenient = true
        coerceInputValues = true
    }

    /** `GET /api/models/{owner}/{repo}` — architecture, context, chat template, siblings. */
    suspend fun modelInfo(repoId: String, revision: String? = null): Result<HfModelInfo> =
        withContext(Dispatchers.IO) {
            val url = buildString {
                append("$BASE/api/models/$repoId")
                if (revision != null) append("/revision/$revision")
                append("?blobs=false")
            }
            request(Request.Builder().url(url).get()).mapCatching { body ->
                json.decodeFromString(HfModelInfo.serializer(), body)
            }
        }

    /** `POST /api/models/{id}/paths-info/{revision}` — exact size, sha256 via `lfs.oid`, the commit id to pin to, and the security scan verdict. */
    suspend fun pathsInfo(
        repoId: String,
        paths: List<String>,
        revision: String = "main",
    ): Result<List<HfPathInfo>> = withContext(Dispatchers.IO) {
        val payload = buildString {
            append("{\"paths\":[")
            append(paths.joinToString(",") { "\"${it.replace("\"", "\\\"")}\"" })
            append("],\"expand\":true}")
        }
        val body = payload.toRequestBody("application/json".toMediaType())
        request(Request.Builder().url("$BASE/api/models/$repoId/paths-info/$revision").post(body))
            .mapCatching { json.decodeFromString(kotlinx.serialization.builtins.ListSerializer(HfPathInfo.serializer()), it) }
    }

    /** Repo search, used by the "search for {repo}-GGUF" remedy on the PyTorch-weights-only refusal and by the Add-model search field. */
    suspend fun search(query: String, limit: Int = 25): Result<List<HfSearchResult>> =
        withContext(Dispatchers.IO) {
            val encoded = java.net.URLEncoder.encode(query, "UTF-8")
            val url = "$BASE/api/models?search=$encoded&limit=$limit&sort=downloads&direction=-1"
            request(Request.Builder().url(url).get()).mapCatching {
                json.decodeFromString(kotlinx.serialization.builtins.ListSerializer(HfSearchResult.serializer()), it)
            }
        }

    /** The GGUF header fallback: an HTTP Range request for the first slice of a file. */
    suspend fun rangeGet(url: String, bytes: Int): Result<ByteArray> = withContext(Dispatchers.IO) {
        runCatching {
            val req = Request.Builder()
                .url(url)
                .header("Range", "bytes=0-${bytes - 1}")
                .apply { tokens.hfToken?.let { header("Authorization", "Bearer $it") } }
                .get()
                .build()
            client.newCall(req).execute().use { response ->
                if (!response.isSuccessful && response.code != 206) {
                    throw HfException(response.code, "Range request failed: HTTP ${response.code}")
                }
                response.body?.bytes() ?: throw IOException("Empty range response")
            }
        }
    }

    fun resolveUrl(repoId: String, filename: String, revision: String = "main"): String =
        "$BASE/$repoId/resolve/$revision/$filename"

    private fun request(builder: Request.Builder): Result<String> = runCatching {
        tokens.hfToken?.let { builder.header("Authorization", "Bearer $it") }
        builder.header("User-Agent", USER_AGENT)
        client.newCall(builder.build()).execute().use { response ->
            val body = response.body?.string().orEmpty()
            if (!response.isSuccessful) {
                throw HfException(response.code, body.take(500))
            }
            body
        }
    }

    companion object {
        const val BASE = "https://huggingface.co"
        private const val USER_AGENT = "OnDeviceAI-Android/0.1 (local inference client)"

        /** Mirrors the resolver offers when a repo ships PyTorch weights only. */
        val GGUF_MIRRORS = listOf("bartowski", "unsloth", "mradermacher")
    }
}

class HfException(val code: Int, message: String) : IOException("HTTP $code: $message") {
    val isAuthFailure: Boolean get() = code == 401 || code == 403
    val isNotFound: Boolean get() = code == 404
}
