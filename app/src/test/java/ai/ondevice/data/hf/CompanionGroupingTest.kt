package ai.ondevice.data.hf

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/** That auxiliary files are sorted into parts, alternatives and choices correctly. */
class CompanionGroupingTest {

    private fun file(name: String, bytes: Long = 1_000, role: CompanionRole) =
        CompanionFile(RemoteFile(filename = name, sizeBytes = bytes), role)

    private fun groupOf(vararg files: CompanionFile) = CompanionGrouping.group(files.toList()).single()

    // — alternatives —

    @Test
    fun `three projectors at three precisions are one file, and F16 wins`() {
        // unsloth/gemma-3-4b-it-GGUF
        val group = groupOf(
            file("mmproj-BF16.gguf", 676_000_000, CompanionRole.VISION_PROJECTOR),
            file("mmproj-F16.gguf", 672_000_000, CompanionRole.VISION_PROJECTOR),
            file("mmproj-F32.gguf", 1_330_000_000, CompanionRole.VISION_PROJECTOR),
        )
        assertEquals(CompanionGroup.Kind.ALTERNATIVES, group.kind)
        assertEquals(setOf("mmproj-F16.gguf"), group.selected)
        // The point of the exercise: 672 MB queued, not 2.68 GB.
        assertEquals(672_000_000L, group.selectedBytes)
    }

    @Test
    fun `F16 is preferred over BF16 even though BF16 is not the largest`() {
        val group = groupOf(
            file("mmproj-BF16.gguf", 1, CompanionRole.VISION_PROJECTOR),
            file("mmproj-F16.gguf", 999, CompanionRole.VISION_PROJECTOR),
        )
        assertEquals(
            "F16 must win on precision, not on size",
            setOf("mmproj-F16.gguf"),
            group.selected,
        )
    }

    @Test
    fun `a projector at f16 and f32 collapses in lower case too`() {
        // bartowski/Qwen2-VL-7B-Instruct-GGUF
        val group = groupOf(
            file("mmproj-Qwen2-VL-7B-Instruct-f16.gguf", 1, CompanionRole.VISION_PROJECTOR),
            file("mmproj-Qwen2-VL-7B-Instruct-f32.gguf", 2, CompanionRole.VISION_PROJECTOR),
        )
        assertEquals(CompanionGroup.Kind.ALTERNATIVES, group.kind)
        assertEquals(setOf("mmproj-Qwen2-VL-7B-Instruct-f16.gguf"), group.selected)
    }

    @Test
    fun `fp8 encoders are the same encoder, not different ones`() {
        // comfyanonymous/flux_text_encoders. The exponent layout and the
        // `scaled` suffix are how a precision is spelled, not what it is.
        val group = groupOf(
            file("t5xxl_fp16.safetensors", 9_800_000_000, CompanionRole.T5XXL),
            file("t5xxl_fp8_e4m3fn.safetensors", 4_900_000_000, CompanionRole.T5XXL),
            file("t5xxl_fp8_e4m3fn_scaled.safetensors", 5_200_000_000, CompanionRole.T5XXL),
        )
        assertEquals(CompanionGroup.Kind.ALTERNATIVES, group.kind)
        assertEquals(setOf("t5xxl_fp16.safetensors"), group.selected)
    }

    @Test
    fun `the same weights in two containers are one thing, and the pickle loses`() {
        // stabilityai/sd-vae-ft-mse-original
        val group = groupOf(
            file("vae-ft-mse-840000-ema-pruned.ckpt", 334_000_000, CompanionRole.VAE),
            file("vae-ft-mse-840000-ema-pruned.safetensors", 334_000_000, CompanionRole.VAE),
        )
        assertEquals(CompanionGroup.Kind.ALTERNATIVES, group.kind)
        assertEquals(setOf("vae-ft-mse-840000-ema-pruned.safetensors"), group.selected)
    }

    @Test
    fun `a separator is not a difference`() {
        // madebyollin/sdxl-vae-fp16-fix ships both spellings.
        val group = groupOf(
            file("sdxl.vae.safetensors", 1, CompanionRole.VAE),
            file("sdxl_vae.safetensors", 1, CompanionRole.VAE),
        )
        assertEquals(CompanionGroup.Kind.ALTERNATIVES, group.kind)
        assertEquals(1, group.selected.size)
    }

    // — parts —

    @Test
    fun `every voice pack is kept`() {
        // onnx-community/Kokoro-82M-v1.0-ONNX has 55 of these. Losing any of
        // them silently removes a voice from the picker.
        val names = listOf(
            "voices/af.bin", "voices/af_alloy.bin", "voices/af_aoede.bin",
            "voices/af_bella.bin", "voices/af_heart.bin", "voices/am_adam.bin",
            "voices/bf_alice.bin", "voices/bm_george.bin",
        )
        val group = CompanionGrouping.group(
            names.map { file(it, 523_000, CompanionRole.VOICES) },
        ).single()

        assertEquals(CompanionGroup.Kind.PARTS, group.kind)
        assertEquals(names.toSet(), group.selected)
    }

    // — choices —

    @Test
    fun `rival ControlNets stay apart despite all being fp16`() {
        // comfyanonymous/ControlNet-v1-1_fp16_safetensors.
        val names = listOf(
            "control_v11p_sd15_canny_fp16.safetensors",
            "control_v11f1p_sd15_depth_fp16.safetensors",
            "control_v11p_sd15_openpose_fp16.safetensors",
            "control_lora_rank128_v11p_sd15_canny_fp16.safetensors",
        )
        val group = CompanionGrouping.group(
            names.map { file(it, 723_000_000, CompanionRole.CONTROLNET) },
        ).single()

        assertEquals(CompanionGroup.Kind.CHOICES, group.kind)
        // Optional role, several genuinely different files: choosing canny over
        // depth for someone is a guess about the picture they want.
        assertTrue("nothing should be queued by default", group.selected.isEmpty())
        assertEquals(0L, group.selectedBytes)
    }

    @Test
    fun `upscale factors are not precisions`() {
        // ai-forever/Real-ESRGAN
        val group = CompanionGrouping.group(
            listOf("RealESRGAN_x2.pth", "RealESRGAN_x4.pth", "RealESRGAN_x8.pth")
                .map { file(it, 67_000_000, CompanionRole.UPSCALER) },
        ).single()

        assertEquals(CompanionGroup.Kind.CHOICES, group.kind)
        assertTrue(group.selected.isEmpty())
    }

    @Test
    fun `a required role always ends up with something chosen`() {
        // A VAE is required, so even when the candidates are genuinely different, leaving nothing selected would produce an install that cannot run.
        val group = CompanionGrouping.group(
            listOf("vae-a.safetensors", "vae-b.safetensors")
                .map { file(it, 100, CompanionRole.VAE) },
        ).single()

        assertEquals(CompanionGroup.Kind.CHOICES, group.kind)
        assertEquals(1, group.selected.size)
    }

    // — the shape of the whole thing —

    @Test
    fun `roles are grouped separately and nothing is lost`() {
        val companions = listOf(
            file("mmproj-F16.gguf", 1, CompanionRole.VISION_PROJECTOR),
            file("mmproj-F32.gguf", 2, CompanionRole.VISION_PROJECTOR),
            file("voices/af_heart.bin", 3, CompanionRole.VOICES),
            file("voices/am_adam.bin", 4, CompanionRole.VOICES),
        )
        val groups = CompanionGrouping.group(companions)

        assertEquals(2, groups.size)
        assertEquals(
            companions.size,
            groups.sumOf { it.candidates.size },
        )
        assertEquals(
            "a repo with both should queue one projector and both voices",
            3,
            groups.sumOf { it.selected.size },
        )
    }

    @Test
    fun `a single candidate needs no explanation`() {
        val group = groupOf(file("mmproj-model-f16.gguf", 1, CompanionRole.VISION_PROJECTOR))
        assertEquals(setOf("mmproj-model-f16.gguf"), group.selected)
        assertEquals(null, group.note)
    }

    @Test
    fun `no companions is not a crash`() {
        assertEquals(emptyList<CompanionGroup>(), CompanionGrouping.group(emptyList()))
    }
}
