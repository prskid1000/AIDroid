package ai.ondevice.proxy

import ai.ondevice.data.secure.TokenStore
import ai.ondevice.engine.EngineLog
import android.content.Context
import java.io.ByteArrayInputStream
import java.io.File
import java.math.BigInteger
import java.security.KeyPairGenerator
import java.security.KeyStore
import java.security.MessageDigest
import java.security.SecureRandom
import java.security.Signature
import java.security.cert.CertificateFactory
import java.security.cert.X509Certificate
import java.text.SimpleDateFormat
import java.util.Locale
import java.util.TimeZone
import javax.inject.Inject
import javax.inject.Singleton
import javax.net.ssl.KeyManagerFactory
import javax.net.ssl.SSLContext

/**
 * The key and certificate the proxy presents, and the only place either is made.
 *
 * There is no certificate authority a phone can ask. Nothing here is reachable
 * from the public internet — the whole point of the tailnet default is that it
 * is not — so ACME and its HTTP-01 challenge have nothing to talk to, and
 * Tailscale's own `tailscale cert`, which would solve this outright, needs the
 * command-line client the Android app does not ship. What is left is a
 * certificate this device signs for itself, and the honest version of that is
 * one a client can actually be told to trust: a real SAN list, `CA:TRUE` so it
 * is a usable trust anchor, and a fingerprint on screen to check it against.
 *
 * **Written by hand, down to the DER.** Android has no API that produces an
 * X.509 certificate with extensions. `AndroidKeyStore` will generate a key pair
 * and wrap it in a self-signed certificate for free, and that certificate has no
 * subjectAltName and no basic constraints — which means every modern client
 * refuses it and no client can be told to trust it either. Ktor ships
 * `ktor-network-tls-certificates`, which builds one properly and then puts it in
 * a `KeyStore.getInstance("JKS")`; there is no JKS provider on Android. So the
 * ASN.1 below is the shortest path to a certificate that works, and it is
 * confined to [Der] so the rest of this file reads as the structure of a
 * certificate rather than as byte arrays.
 *
 * The keystore is written once and reused, because the fingerprint is the thing
 * a client pins and a fingerprint that changes on every restart is not one. It
 * is regenerated only when the address it was made for is no longer one this
 * device answers on — see [material].
 */
@Singleton
class ProxyTls @Inject constructor(
    @dagger.hilt.android.qualifiers.ApplicationContext private val context: Context,
    private val tokens: TokenStore,
) {

    /** What the front needs to serve, and what the screen needs to show. */
    data class Material(
        val sslContext: SSLContext,
        val certificate: X509Certificate,
    ) {
        /** SHA-256 over the DER, which is what every client calls "the fingerprint". */
        val fingerprint: String
            get() = MessageDigest.getInstance("SHA-256")
                .digest(certificate.encoded)
                .joinToString(":") { "%02X".format(it) }

        val pem: String
            get() {
                val body = android.util.Base64.encodeToString(
                    certificate.encoded,
                    android.util.Base64.NO_WRAP,
                ).chunked(64).joinToString("\n")
                return "-----BEGIN CERTIFICATE-----\n" + body + "\n-----END CERTIFICATE-----\n"
            }

        /** The names and addresses this certificate is valid for. */
        val names: List<String>
            get() = runCatching {
                certificate.subjectAlternativeNames.orEmpty()
                    .mapNotNull { it.getOrNull(1) as? String }
            }.getOrDefault(emptyList())
    }

    private val file: File get() = File(File(context.filesDir, "tls"), KEYSTORE)

    private var cached: Material? = null

    /**
     * The material to serve on [names], generating it if there is none that fits.
     *
     * Regeneration is decided by the SAN list rather than by age: a certificate
     * naming a `100.x` address is worthless the moment Tailscale hands this
     * device a different one, and a client connecting to an address the
     * certificate does not name gets a hostname mismatch, which reads as an
     * attack rather than as a stale file. The common case — the same address as
     * last time — reuses the file, so the fingerprint on the screen does not
     * move under a client that pinned it.
     */
    @Synchronized
    fun material(names: List<String>): Material {
        val wanted = names.filter { it.isNotBlank() }.distinct()
        load()?.let { existing ->
            if (wanted.all { it in existing.names }) {
                cached = existing
                return existing
            }
            EngineLog.i(
                "ProxyTls",
                "certificate does not cover " +
                    wanted.filterNot { it in existing.names }.joinToString(", ") +
                    ", making a new one",
            )
        }
        return generate(wanted).also { cached = it }
    }

    /** Thrown away, so the next [material] makes a new one. */
    @Synchronized
    fun forget() {
        runCatching { file.delete() }
        cached = null
    }

    /** What is on disk, for a screen that wants to show it without starting anything. */
    @Synchronized
    fun current(): Material? = cached ?: load()?.also { cached = it }

    // ── the keystore ────────────────────────────────────────────────────

    private fun load(): Material? {
        if (!file.isFile) return null
        val password = tokens.proxyKeystorePassword?.toCharArray() ?: return null
        return runCatching {
            val store = KeyStore.getInstance(STORE_TYPE)
            file.inputStream().use { store.load(it, password) }
            val certificate = store.getCertificate(ALIAS) as? X509Certificate
                ?: error("no certificate under $ALIAS")
            Material(sslContextFor(store, password), certificate)
        }.onFailure {
            EngineLog.w("ProxyTls", "stored certificate would not load: ${it.message}")
        }.getOrNull()
    }

    private fun generate(names: List<String>): Material {
        val keys = KeyPairGenerator.getInstance("RSA")
            .apply { initialize(KEY_BITS) }
            .generateKeyPair()
        val certificate = SelfSigned.certificate(keys, names)

        val password = (tokens.proxyKeystorePassword ?: newPassword()).toCharArray()
        val store = KeyStore.getInstance(STORE_TYPE)
        store.load(null, password)
        store.setKeyEntry(ALIAS, keys.private, password, arrayOf(certificate))
        runCatching {
            file.parentFile?.mkdirs()
            file.outputStream().use { store.store(it, password) }
        }.onFailure {
            // Servable but not durable, and said out loud rather than swallowed:
            // the fingerprint will change on the next restart, and a client that
            // pinned this one will then refuse to connect for a reason nobody
            // would otherwise be able to explain.
            EngineLog.w(
                "ProxyTls",
                "certificate could not be saved, so it will not survive a restart: ${it.message}",
            )
        }
        EngineLog.i("ProxyTls", "made a certificate for " + names.joinToString(", "))
        return Material(sslContextFor(store, password), certificate)
    }

    private fun newPassword(): String {
        val bytes = ByteArray(24)
        SecureRandom().nextBytes(bytes)
        return bytes.joinToString("") { "%02x".format(it) }
            .also { tokens.proxyKeystorePassword = it }
    }

    private fun sslContextFor(store: KeyStore, password: CharArray): SSLContext {
        val managers = KeyManagerFactory.getInstance(KeyManagerFactory.getDefaultAlgorithm())
            .apply { init(store, password) }
        return SSLContext.getInstance("TLS").apply {
            init(managers.keyManagers, null, SecureRandom())
        }
    }

    private companion object {
        const val KEYSTORE = "proxy.p12"

        /** PKCS12, because Android has no JKS provider and never had one. */
        const val STORE_TYPE = "PKCS12"
        const val ALIAS = "proxy"
        const val KEY_BITS = 2048
    }
}

/**
 * A self-signed certificate, assembled field by field.
 *
 * RFC 5280 §4.1 read downwards: what follows is that structure and nothing else.
 * The public key goes in verbatim — `PublicKey.getEncoded()` is already a DER
 * `SubjectPublicKeyInfo`, which is the one part of this nobody has to write.
 */
internal object SelfSigned {

    fun certificate(keys: java.security.KeyPair, names: List<String>): X509Certificate {
        val serial = BigInteger(64, SecureRandom())
            .let { if (it.signum() == 0) BigInteger.ONE else it }
        val now = System.currentTimeMillis()
        val subject = Der.name(names.firstOrNull() ?: "on-device")

        val tbs = Der.sequence(
            // [0] EXPLICIT version — 2 is v3, and only v3 may carry extensions.
            Der.explicit(0, Der.integer(BigInteger.valueOf(2))),
            Der.integer(serial),
            SIGNATURE_ALGORITHM,
            subject,
            Der.sequence(Der.utcTime(now - BACKDATE_MILLIS), Der.utcTime(now + LIFETIME_MILLIS)),
            subject,
            keys.public.encoded,
            Der.explicit(3, Der.sequence(*extensions(names).toTypedArray())),
        )

        val signature = Signature.getInstance("SHA256withRSA").run {
            initSign(keys.private)
            update(tbs)
            sign()
        }

        val der = Der.sequence(tbs, SIGNATURE_ALGORITHM, Der.bitString(signature))
        return CertificateFactory.getInstance("X.509")
            .generateCertificate(ByteArrayInputStream(der)) as X509Certificate
    }

    private fun extensions(names: List<String>): List<ByteArray> = listOf(
        // CA:TRUE, and that is deliberate. A client cannot be told to trust a
        // leaf: `curl --cacert`, `NODE_EXTRA_CA_CERTS` and Java's own trust
        // manager all want an anchor, and an anchor whose basic constraints do
        // not say it is one is rejected before the name is even looked at. This
        // is the shape `openssl req -x509` produces, which clients already take.
        Der.extension(
            OID_BASIC_CONSTRAINTS,
            critical = true,
            value = Der.sequence(Der.bool(true)),
        ),
        // digitalSignature, keyEncipherment, keyCertSign — 0xA4 is those three
        // bits, and the last of them is what makes this consistent with CA:TRUE.
        Der.extension(
            OID_KEY_USAGE,
            critical = true,
            value = Der.bitString(byteArrayOf(0xA4.toByte()), unusedBits = 2),
        ),
        Der.extension(
            OID_EXT_KEY_USAGE,
            critical = false,
            value = Der.sequence(Der.oid(OID_SERVER_AUTH)),
        ),
        Der.extension(
            OID_SUBJECT_ALT_NAME,
            critical = false,
            value = Der.sequence(*names.map(::generalName).toTypedArray()),
        ),
    )

    /**
     * One SAN entry, as either an address or a name.
     *
     * Both kinds are needed and they are not interchangeable: a client told to
     * connect to `100.x.y.z` checks `iPAddress` and ignores `dNSName` entirely,
     * and a client given a MagicDNS name checks only the other one.
     */
    private fun generalName(value: String): ByteArray {
        val octets = value.split('.')
            .takeIf { parts -> parts.size == 4 && parts.all { (it.toIntOrNull() ?: -1) in 0..255 } }
            ?.map { it.toInt().toByte() }
            ?.toByteArray()
        return if (octets != null) {
            Der.primitive(7, octets)
        } else {
            Der.primitive(2, value.toByteArray(Charsets.US_ASCII))
        }
    }

    /** sha256WithRSAEncryption, with the NULL parameters RFC 4055 requires. */
    private val SIGNATURE_ALGORITHM = Der.sequence(Der.oid("1.2.840.113549.1.1.11"), Der.nul())

    private const val OID_BASIC_CONSTRAINTS = "2.5.29.19"
    private const val OID_KEY_USAGE = "2.5.29.15"
    private const val OID_SUBJECT_ALT_NAME = "2.5.29.17"
    private const val OID_EXT_KEY_USAGE = "2.5.29.37"
    private const val OID_SERVER_AUTH = "1.3.6.1.5.5.7.3.1"

    /** A phone whose clock runs a few minutes fast must not issue a future certificate. */
    private const val BACKDATE_MILLIS = 24L * 60 * 60 * 1000

    /**
     * Ten years, which also keeps `notAfter` inside the range `UTCTime` can
     * express — from 2050 the encoding has to change to `GeneralizedTime`, and a
     * certificate this device can reissue in a second has no reason to reach
     * that far.
     */
    private const val LIFETIME_MILLIS = 3_650L * 24 * 60 * 60 * 1000
}

/**
 * Just enough DER to write one certificate.
 *
 * Not a general encoder and not trying to be. Everything here is a tag, a length
 * and a body; the only part with any subtlety is the length, which is a single
 * byte below 128 and a byte count followed by the bytes above it.
 */
private object Der {

    fun sequence(vararg parts: ByteArray): ByteArray =
        tlv(0x30, parts.fold(ByteArray(0)) { all, part -> all + part })

    fun set(body: ByteArray): ByteArray = tlv(0x31, body)

    fun integer(value: BigInteger): ByteArray = tlv(0x02, value.toByteArray())

    fun bool(value: Boolean): ByteArray = tlv(0x01, byteArrayOf(if (value) 0xFF.toByte() else 0))

    fun nul(): ByteArray = byteArrayOf(0x05, 0x00)

    fun octetString(body: ByteArray): ByteArray = tlv(0x04, body)

    fun bitString(body: ByteArray, unusedBits: Int = 0): ByteArray =
        tlv(0x03, byteArrayOf(unusedBits.toByte()) + body)

    fun utf8String(value: String): ByteArray = tlv(0x0C, value.toByteArray(Charsets.UTF_8))

    /** Context-specific, constructed: the wrapper an EXPLICIT tag is. */
    fun explicit(tag: Int, body: ByteArray): ByteArray = tlv(0xA0 or tag, body)

    /** Context-specific, primitive: what a `GeneralName` alternative is. */
    fun primitive(tag: Int, body: ByteArray): ByteArray = tlv(0x80 or tag, body)

    fun utcTime(millis: Long): ByteArray {
        val format = SimpleDateFormat("yyMMddHHmmss'Z'", Locale.US)
            .apply { timeZone = TimeZone.getTimeZone("UTC") }
        return tlv(0x17, format.format(java.util.Date(millis)).toByteArray(Charsets.US_ASCII))
    }

    /** A one-attribute distinguished name — a CN and nothing else. */
    fun name(commonName: String): ByteArray =
        sequence(set(sequence(oid("2.5.4.3"), utf8String(commonName))))

    fun extension(id: String, critical: Boolean, value: ByteArray): ByteArray =
        if (critical) {
            sequence(oid(id), bool(true), octetString(value))
        } else {
            // `critical` is DEFAULT FALSE, and DER forbids encoding a default.
            sequence(oid(id), octetString(value))
        }

    fun oid(dotted: String): ByteArray {
        val parts = dotted.split('.').map { it.toLong() }
        // The first two arcs share one byte, which is the single irregularity
        // in the whole encoding.
        var body = byteArrayOf((parts[0] * 40 + parts[1]).toByte())
        parts.drop(2).forEach { body += base128(it) }
        return tlv(0x06, body)
    }

    private fun base128(value: Long): ByteArray {
        if (value < 128) return byteArrayOf(value.toByte())
        var bytes = ByteArray(0)
        var remaining = value
        var last = true
        while (remaining > 0) {
            val septet = (remaining and 0x7F).toInt()
            bytes = byteArrayOf((if (last) septet else septet or 0x80).toByte()) + bytes
            last = false
            remaining = remaining shr 7
        }
        return bytes
    }

    private fun tlv(tag: Int, body: ByteArray): ByteArray {
        val length = if (body.size < 0x80) {
            byteArrayOf(body.size.toByte())
        } else {
            var bytes = ByteArray(0)
            var remaining = body.size
            while (remaining > 0) {
                bytes = byteArrayOf((remaining and 0xFF).toByte()) + bytes
                remaining = remaining shr 8
            }
            byteArrayOf((0x80 or bytes.size).toByte()) + bytes
        }
        return byteArrayOf(tag.toByte()) + length + body
    }
}
