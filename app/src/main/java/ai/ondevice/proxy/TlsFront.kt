package ai.ondevice.proxy

import ai.ondevice.engine.EngineLog
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import java.io.InputStream
import java.io.OutputStream
import java.net.InetAddress
import java.net.InetSocketAddress
import java.net.Socket
import javax.net.ssl.SSLContext
import javax.net.ssl.SSLServerSocket

/**
 * TLS on the outside, the same plaintext server on the inside.
 *
 * Ktor's CIO engine has no server-side TLS — that is a Netty and Jetty feature,
 * and neither belongs on a phone next to five native runtimes. The alternative
 * is this: the real server binds loopback on a port nobody is told about, and
 * this holds the address people were given and copies bytes between the two.
 * Everything the routing does — streaming, the ping interval, a forty-minute
 * clip on one connection — is unchanged, because nothing here understands HTTP.
 *
 * **Every write is flushed.** A copy loop that buffers is the difference
 * between server-sent events arriving as they are produced and arriving all at
 * once when the answer ends, and the second one looks exactly like a model that
 * has hung. This is the whole reason the pipe is written out rather than being
 * `InputStream.copyTo`.
 *
 * The cost is that every request now arrives from `127.0.0.1`, so the request
 * log's last-resort client column says that instead of the caller's address. It
 * is the last resort: a profile name comes first and the User-Agent second, and
 * both still arrive intact.
 */
class TlsFront {

    private var socket: SSLServerSocket? = null
    private var accepting: Job? = null

    val listening: Boolean get() = socket?.isClosed == false

    /** What it actually bound to, which is the question when it was asked for 0. */
    val port: Int get() = socket?.localPort ?: -1

    /**
     * Bind [address]:[port] and forward everything to [upstreamPort] on loopback.
     *
     * Throws when the bind fails, because a front that is not listening is the
     * whole feature not working and [ProxyServer] turns that into a refusal
     * somebody can read.
     */
    fun start(scope: CoroutineScope, context: SSLContext, address: String, port: Int, upstreamPort: Int) {
        stop()
        val server = (context.serverSocketFactory.createServerSocket() as SSLServerSocket).apply {
            reuseAddress = true
            bind(InetSocketAddress(InetAddress.getByName(address), port), BACKLOG)
            // Whatever this device's provider offers, minus the versions that
            // are no longer worth offering. Named as an intersection rather than
            // as a list, so a platform that drops one does not leave us asking
            // for something that no longer exists.
            enabledProtocols = supportedProtocols.filter { it in WANTED_PROTOCOLS }.toTypedArray()
        }
        socket = server
        accepting = scope.launch(Dispatchers.IO) {
            while (isActive && !server.isClosed) {
                val client = runCatching { server.accept() }.getOrNull() ?: break
                // Handshake happens on first read, inside the connection's own
                // coroutine. Doing it here would let one client that opened a
                // socket and then said nothing stop every other client from
                // being accepted.
                launch(Dispatchers.IO) { forward(client, upstreamPort) }
            }
        }
    }

    fun stop() {
        accepting?.cancel()
        accepting = null
        runCatching { socket?.close() }
        socket = null
    }

    private suspend fun forward(client: Socket, upstreamPort: Int) = kotlinx.coroutines.coroutineScope {
        val upstream = runCatching {
            Socket().apply {
                tcpNoDelay = true
                connect(InetSocketAddress(InetAddress.getByName("127.0.0.1"), upstreamPort), CONNECT_MILLIS)
            }
        }.getOrElse {
            EngineLog.w("ProxyTls", "could not reach the server behind TLS: ${it.message}")
            runCatching { client.close() }
            return@coroutineScope
        }

        client.tcpNoDelay = true
        // No read timeout on either side. A generation holds the connection open
        // and says nothing for minutes at a time, and a timeout here would cut
        // it in a place that looks like the model crashed.
        client.soTimeout = 0
        upstream.soTimeout = 0

        val close = {
            runCatching { client.close() }
            runCatching { upstream.close() }
            Unit
        }

        val up = launch(Dispatchers.IO) {
            runCatching { pipe(client.getInputStream(), upstream.getOutputStream()) }
            close()
        }
        val down = launch(Dispatchers.IO) {
            runCatching { pipe(upstream.getInputStream(), client.getOutputStream()) }
            close()
        }
        up.join()
        down.join()
    }

    private fun pipe(from: InputStream, to: OutputStream) {
        val buffer = ByteArray(BUFFER_BYTES)
        while (true) {
            val read = from.read(buffer)
            if (read < 0) break
            to.write(buffer, 0, read)
            // Per chunk. See the note at the top: an SSE frame sitting in a
            // buffer is a frame that never arrived.
            to.flush()
        }
    }

    private companion object {
        const val BACKLOG = 32
        const val BUFFER_BYTES = 16 * 1024
        const val CONNECT_MILLIS = 5_000

        val WANTED_PROTOCOLS = setOf("TLSv1.3", "TLSv1.2")
    }
}
