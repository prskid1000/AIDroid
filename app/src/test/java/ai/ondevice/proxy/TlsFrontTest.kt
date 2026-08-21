package ai.ondevice.proxy

import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.net.InetAddress
import java.net.ServerSocket
import java.security.KeyPairGenerator
import java.security.KeyStore
import java.security.cert.X509Certificate
import javax.net.ssl.KeyManagerFactory
import javax.net.ssl.SSLContext
import javax.net.ssl.SSLSocket
import javax.net.ssl.TrustManagerFactory
import kotlin.concurrent.thread

/**
 * The TLS path, end to end, without a phone in it.
 *
 * Everything this covers fails somewhere other than here: a certificate the JVM
 * will not accept as an anchor, a SAN a client checks and rejects, a pipe that
 * buffers. On a device all three arrive as the same sentence in somebody else's
 * terminal — "certificate verify failed", or a request that never returns — and
 * the phone is the last place anyone looks. So the assertions below are made by
 * a real client: a real handshake, with hostname verification on, against a real
 * plaintext server behind the front.
 */
class TlsFrontTest {

    private val password = "test".toCharArray()

    @Test
    fun `a client that trusts the certificate reaches the server behind it`() {
        val keys = KeyPairGenerator.getInstance("RSA").apply { initialize(2048) }.generateKeyPair()
        val certificate = SelfSigned.certificate(keys, listOf("127.0.0.1", "localhost"))

        val upstream = ServerSocket(0, 4, InetAddress.getByName("127.0.0.1"))
        val serving = thread { answerOnce(upstream) }

        val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
        val front = TlsFront()
        try {
            front.start(scope, serverContext(keys, certificate), "127.0.0.1", 0, upstream.localPort)

            val socket = clientContext(certificate).socketFactory
                .createSocket("127.0.0.1", front.port) as SSLSocket
            // The check a browser and curl do and a bare `SSLSocket` does not.
            // Without it this test would pass on a certificate naming nothing at
            // all, which is the failure most likely to reach a device.
            socket.sslParameters = socket.sslParameters.apply {
                endpointIdentificationAlgorithm = "HTTPS"
            }
            socket.startHandshake()

            socket.getOutputStream().write(
                "GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n"
                    .toByteArray(Charsets.US_ASCII),
            )
            socket.getOutputStream().flush()

            val answer = socket.getInputStream().readBytes().toString(Charsets.US_ASCII)
            assertTrue(answer, answer.startsWith("HTTP/1.1 200 OK"))
            assertTrue(answer, answer.endsWith("behind the front"))
            socket.close()
        } finally {
            front.stop()
            scope.cancel()
            upstream.close()
            serving.join(2_000)
        }
    }

    @Test
    fun `the port it reports is the port it bound`() {
        val keys = KeyPairGenerator.getInstance("RSA").apply { initialize(2048) }.generateKeyPair()
        val certificate = SelfSigned.certificate(keys, listOf("127.0.0.1"))
        val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
        val front = TlsFront()
        try {
            front.start(scope, serverContext(keys, certificate), "127.0.0.1", 0, 1)
            // Asked for 0, so the kernel chose — and `ProxyServer` has to be able
            // to find out what it chose, exactly as it does for the CIO server.
            assertTrue(front.port > 0)
            assertTrue(front.listening)
            front.stop()
            assertEquals(-1, front.port)
        } finally {
            front.stop()
            scope.cancel()
        }
    }

    /** One request, one answer, then gone. Enough to prove bytes cross both ways. */
    private fun answerOnce(server: ServerSocket) {
        runCatching {
            server.accept().use { socket ->
                val reader = socket.getInputStream().bufferedReader()
                while (true) {
                    val line = reader.readLine() ?: break
                    if (line.isEmpty()) break
                }
                val body = "behind the front"
                socket.getOutputStream().apply {
                    write(
                        ("HTTP/1.1 200 OK\r\nContent-Length: ${body.length}\r\n\r\n$body")
                            .toByteArray(Charsets.US_ASCII),
                    )
                    flush()
                }
            }
        }
    }

    private fun serverContext(keys: java.security.KeyPair, certificate: X509Certificate): SSLContext {
        val store = KeyStore.getInstance("PKCS12").apply {
            load(null, password)
            setKeyEntry("proxy", keys.private, password, arrayOf(certificate))
        }
        val managers = KeyManagerFactory.getInstance(KeyManagerFactory.getDefaultAlgorithm())
            .apply { init(store, password) }
        return SSLContext.getInstance("TLS").apply { init(managers.keyManagers, null, null) }
    }

    /**
     * A client told to trust this one certificate and nothing else.
     *
     * The same thing `curl --cacert` and `NODE_EXTRA_CA_CERTS` do, and the
     * reason the certificate carries `CA:TRUE`: PKIX will not take an anchor
     * whose basic constraints do not say it is one.
     */
    private fun clientContext(certificate: X509Certificate): SSLContext {
        val store = KeyStore.getInstance("PKCS12").apply {
            load(null, password)
            setCertificateEntry("proxy", certificate)
        }
        val managers = TrustManagerFactory.getInstance(TrustManagerFactory.getDefaultAlgorithm())
            .apply { init(store) }
        return SSLContext.getInstance("TLS").apply { init(null, managers.trustManagers, null) }
    }
}
