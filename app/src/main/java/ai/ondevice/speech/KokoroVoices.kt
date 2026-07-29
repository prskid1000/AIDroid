package ai.ondevice.speech

/**
 * Kokoro's voice catalogue.
 *
 * This is a *display* catalogue, not a capability claim. The names, languages
 * and genders come from the voice-pack naming scheme Kokoro itself uses —
 * `<language><gender>_<name>`, so `bf_emma` is British English, female, "Emma".
 * Which of them can actually speak is decided by [catalogue]'s `available`
 * flag, which the caller derives from whether the runtime and a voice pack are
 * installed. Listing a voice the device cannot produce is fine and useful;
 * *speaking* in a different one and calling it Kokoro is not.
 */
object KokoroVoices {

    /**
     * [available] is whether Kokoro's runtime and weights are installed. A
     * voice also needs a *front end* for its language, and this build's
     * phonemiser is espeak-ng, which is not what Kokoro's own pipeline uses for
     * Japanese or Mandarin. Those voices are listed and marked rather than
     * quietly offered, because producing espeak's guess at Japanese and calling
     * it Kokoro would be exactly the substitution §1.2 forbids.
     */
    fun catalogue(available: Boolean): List<SynthVoice> = NAMES.map { id ->
        SynthVoice(
            id = id,
            displayName = id.substringAfter('_').replaceFirstChar(Char::uppercase),
            locale = localeTag(id),
            localeLabel = "${languageLabel(id)} · ${genderLabel(id)}",
            quality = 400,
            provider = SynthProvider.KOKORO,
            available = available && hasPhonemiser(id),
        )
    }

    /** Whether espeak-ng can pronounce this voice's language in this build. */
    fun hasPhonemiser(id: String): Boolean = id.firstOrNull() in setOf('a', 'b', 'e', 'f', 'h', 'i', 'p')

    fun languageOf(id: String): String = languageLabel(id)

    /** The voice pack filename inside a Kokoro model directory. */
    fun packName(id: String): String = "$id.bin"

    private fun languageLabel(id: String): String = when (id.first()) {
        'a' -> "American English"
        'b' -> "British English"
        'e' -> "Spanish"
        'f' -> "French"
        'h' -> "Hindi"
        'i' -> "Italian"
        'j' -> "Japanese"
        'p' -> "Portuguese"
        'z' -> "Mandarin"
        else -> "Unknown"
    }

    private fun localeTag(id: String): String = when (id.first()) {
        'a' -> "en-US"
        'b' -> "en-GB"
        'e' -> "es"
        'f' -> "fr-FR"
        'h' -> "hi"
        'i' -> "it"
        'j' -> "ja"
        'p' -> "pt-BR"
        'z' -> "zh"
        else -> "und"
    }

    private fun genderLabel(id: String): String = when (id.getOrNull(1)) {
        'f' -> "female"
        'm' -> "male"
        else -> "unspecified"
    }

    private val NAMES = listOf(
        // American English
        "af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica", "af_kore",
        "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
        "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael",
        "am_onyx", "am_puck", "am_santa",
        // British English
        "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
        "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
        // Spanish
        "ef_dora", "em_alex", "em_santa",
        // French
        "ff_siwis",
        // Hindi
        "hf_alpha", "hf_beta", "hm_omega", "hm_psi",
        // Italian
        "if_sara", "im_nicola",
        // Japanese
        "jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro", "jm_kumo",
        // Portuguese
        "pf_dora", "pm_alex", "pm_santa",
        // Mandarin
        "zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi",
        "zm_yunjian", "zm_yunxi", "zm_yunxia", "zm_yunyang",
    )
}
