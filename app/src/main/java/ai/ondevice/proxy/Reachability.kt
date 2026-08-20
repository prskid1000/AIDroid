package ai.ondevice.proxy

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.net.Inet4Address
import java.net.InetAddress
import java.net.NetworkInterface

/**
 * Where this device can be reached, and whether it can be reached at all.
 *
 * Everything here is inference from the network stack, because there is no
 * other source. The Tailscale Android app ships no CLI, so there is no
 * `tailscale status --json` to read the way telecode's tray does — see
 * `docs/proxy-plan.md` 8. What there *is* is a tun interface holding a
 * `100.64.0.0/10` address whenever the tailnet is up, and that is enough to
 * answer the only question the screen actually asks.
 */
object Reachability {

    /**
     * A Tailscale address, or null when the tailnet is down.
     *
     * `100.64.0.0/10` is the CGNAT range Tailscale allocates from. Nothing else
     * on a phone hands out an address in it, so presence is a reliable signal —
     * and its absence is the honest answer "Tailscale is not connected" rather
     * than a URL that resolves to nothing.
     */
    fun tailnetAddress(): String? = ipv4Addresses().firstOrNull { isTailscale(it) }

    /** Every routable IPv4 this device holds, tailnet first. */
    fun localAddresses(): List<String> {
        val all = ipv4Addresses()
        return all.filter { isTailscale(it) } + all.filterNot { isTailscale(it) }
    }

    fun isTailscale(address: String): Boolean {
        val parts = address.split('.')
        if (parts.size != 4) return false
        val first = parts[0].toIntOrNull() ?: return false
        val second = parts[1].toIntOrNull() ?: return false
        // 100.64.0.0/10 — the second octet runs 64..127.
        return first == 100 && second in 64..127
    }

    /**
     * The MagicDNS name for an address, when there is one.
     *
     * A reverse lookup, because it is the only route to the name without the
     * CLI: with MagicDNS on, Tailscale's resolver answers PTR queries for
     * tailnet addresses. It fails quietly and often — MagicDNS off, a resolver
     * that does not go through the tunnel, a device that has not registered —
     * and null is a supported answer. The raw address always works, so a name
     * is a nicety and is never guessed at.
     *
     * Blocking, so it is suspend: a reverse lookup that has to leave the device
     * is not something to do on a UI thread.
     */
    suspend fun magicDnsName(address: String): String? = withContext(Dispatchers.IO) {
        runCatching {
            val resolved = InetAddress.getByName(address).canonicalHostName
            resolved
                .takeIf { it != address && it.contains('.') }
                ?.trimEnd('.')
                ?.takeIf { it.endsWith(TAILNET_SUFFIX) }
        }.getOrNull()
    }

    /**
     * What to bind to, given the configured policy.
     *
     * Returns null when the policy asks for the tailnet and there is no tailnet
     * — which is a refusal, not a fallback. Falling back to `0.0.0.0` here
     * would put a generation server on whatever Wi-Fi the phone is on, silently,
     * because a VPN was off. That is the failure this whole file exists to
     * avoid.
     */
    fun resolveBindAddress(policy: String): BindResult = when (policy) {
        ProxySpecs.BIND_LOOPBACK -> BindResult.Ok("127.0.0.1")
        ProxySpecs.BIND_ALL -> BindResult.Ok("0.0.0.0")
        else -> tailnetAddress()
            ?.let { BindResult.Ok(it) }
            ?: BindResult.NoTailnet
    }

    sealed interface BindResult {
        data class Ok(val address: String) : BindResult
        data object NoTailnet : BindResult
    }

    private fun ipv4Addresses(): List<String> = runCatching {
        NetworkInterface.getNetworkInterfaces()
            .asSequence()
            .filter { runCatching { it.isUp && !it.isLoopback }.getOrDefault(false) }
            .flatMap { it.inetAddresses.asSequence() }
            .filterIsInstance<Inet4Address>()
            .mapNotNull { it.hostAddress }
            .toList()
    }.getOrDefault(emptyList())

    private const val TAILNET_SUFFIX = ".ts.net"
}
