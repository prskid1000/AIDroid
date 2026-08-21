package ai.ondevice.proxy

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.security.KeyPairGenerator
import java.security.cert.X509Certificate

/**
 * The certificate, checked by the thing that has to accept it.
 *
 * This is exactly the kind of code SPEC calls invisible on a device: a byte out
 * of place in the DER does not produce a wrong picture on a screen, it produces
 * a TLS handshake that fails on the other machine with a message about the
 * certificate being unparseable — at which point the phone is the last place
 * anybody looks. Every assertion below reads the certificate back through
 * `java.security`, so what is being tested is whether a real X.509 parser agrees
 * with what was written, not whether the bytes match some other copy of them.
 */
class SelfSignedTest {

    private val keys by lazy {
        KeyPairGenerator.getInstance("RSA").apply { initialize(2048) }.generateKeyPair()
    }

    private fun certificate(vararg names: String): X509Certificate =
        SelfSigned.certificate(keys, names.toList())

    @Test
    fun `parses, verifies against its own key and is valid now`() {
        val cert = certificate("100.101.102.103", "phone.tail1234.ts.net")

        // Self-signed: the signature over the TBS has to check out against the
        // public key inside it. This is the assertion that catches a length byte
        // written wrongly, because the parser would then be hashing a different
        // span than the signer did.
        cert.verify(keys.public)
        cert.checkValidity()
        assertEquals(cert.subjectX500Principal, cert.issuerX500Principal)
        assertEquals(3, cert.version)
    }

    @Test
    fun `addresses go in as addresses and names as names`() {
        val cert = certificate("100.101.102.103", "phone.tail1234.ts.net", "127.0.0.1", "localhost")
        val entries = cert.subjectAlternativeNames.orEmpty().map { it[0] as Int to it[1] as String }

        // The tags are the point. A client dialling an address checks
        // `iPAddress` (7) and never looks at `dNSName` (2), so an address filed
        // under the wrong one is a certificate that names the host and is
        // rejected anyway.
        assertTrue(7 to "100.101.102.103" in entries)
        assertTrue(7 to "127.0.0.1" in entries)
        assertTrue(2 to "phone.tail1234.ts.net" in entries)
        assertTrue(2 to "localhost" in entries)
    }

    @Test
    fun `is a trust anchor a client can be given`() {
        val cert = certificate("100.101.102.103")

        // A leaf cannot be trusted directly: `curl --cacert` and every other
        // way of handing a client one certificate wants an anchor, and an
        // anchor without CA:TRUE is refused before its name is read. Anything
        // other than -1 means basicConstraints says CA.
        assertTrue(cert.basicConstraints >= 0)

        val usage = cert.keyUsage
        assertTrue("digitalSignature", usage[0])
        assertTrue("keyEncipherment", usage[2])
        assertTrue("keyCertSign", usage[5])

        assertTrue(cert.extendedKeyUsage.contains("1.3.6.1.5.5.7.3.1"))
        assertTrue(cert.criticalExtensionOIDs.contains("2.5.29.19"))
    }

    @Test
    fun `a serial with a high bit set is still a positive integer`() {
        // Ten of them, because the failure is one bit of one random number: a
        // serial whose top bit is set encodes as negative unless the leading
        // zero byte is there, and roughly half of all serials have it. One
        // certificate would pass this by luck.
        repeat(10) {
            assertTrue(certificate("127.0.0.1").serialNumber.signum() > 0)
        }
    }
}
