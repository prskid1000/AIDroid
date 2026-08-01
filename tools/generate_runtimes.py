#!/usr/bin/env python3
"""Regenerate app/src/main/assets/runtimes.json from the pinned upstream sources.

SPEC 2.3 and Appendix A #3 are explicit that the architecture allowlist must be
generated from the source the runtime is actually built from. A hand-kept list
rots: an architecture llama.cpp gained months ago stays invisible to the
resolver, and the app silently reintroduces the model-locking the whole design
exists to avoid.

This is step 4 of the runtime-bump pipeline in SPEC 17.5. Run it after moving a
submodule to a new tag, never instead of moving the tag.

    python tools/generate_runtimes.py

Contract test 17.6 #5 gates the result: the regenerated architecture list must be
a superset of the previous one, or the removal must be deliberate. This script
prints removals loudly rather than writing them silently.
"""

from __future__ import annotations

import json
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
NATIVE = ROOT / "native"
ASSET = ROOT / "app" / "src" / "main" / "assets" / "runtimes.json"


def git(repo: pathlib.Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()


def describe(repo: pathlib.Path) -> tuple[str, str]:
    """(buildTag, shortCommit) for a vendored checkout."""
    commit = git(repo, "rev-parse", "--short", "HEAD")
    try:
        tag = git(repo, "describe", "--tags", "--abbrev=0")
    except subprocess.CalledProcessError:
        # A project that does not tag releases (stable-diffusion.cpp) still needs
        # a build tag the `sinceBuild` gate can order, so follow its own
        # `master-<sha>` convention rather than inventing one.
        tag = f"master-{commit}"
    return tag, commit


def llama_architectures(repo: pathlib.Path) -> list[str]:
    """Read LLM_ARCH_NAMES out of src/llama-arch.cpp."""
    source = (repo / "src" / "llama-arch.cpp").read_text(encoding="utf-8")
    block = re.search(
        r"LLM_ARCH_NAMES\s*=\s*\{(.*?)\n\};", source, re.S
    )
    if not block:
        raise SystemExit("LLM_ARCH_NAMES not found — upstream moved it; update this script.")
    names = re.findall(r'\{\s*LLM_ARCH_\w+\s*,\s*"([^"]+)"\s*\}', block.group(1))
    # LLM_ARCH_UNKNOWN is a sentinel, not a model anyone can run.
    return sorted({n for n in names if n != "(unknown)"})


def whisper_architectures(repo: pathlib.Path) -> list[str]:
    # whisper.cpp is single-architecture by construction.
    return ["whisper"]


def kokoro_architectures(repo: pathlib.Path) -> list[str]:
    """Kokoro's front end is espeak-ng, and that is what is vendored.

    The "architecture" a Kokoro install is gated on is not the ONNX graph --
    every published Kokoro export is the same graph -- it is whether this build
    can turn text into the phonemes the graph expects. So the allowlist is the
    set of languages the staged espeak data can pronounce, read from the
    dictionaries actually present in assets rather than from a list here.
    """
    staged = ROOT / "app" / "src" / "main" / "assets" / "espeak-ng-data"
    if not staged.is_dir():
        return []
    return sorted(p.name[: -len("_dict")] for p in staged.glob("*_dict"))


def sd_architectures(repo: pathlib.Path) -> list[str]:
    """Read the SDVersion enum out of stable-diffusion.cpp's model.h."""
    header = repo / "model.h"
    if not header.exists():
        return ["sd1", "sd2", "sdxl", "sd3", "flux"]
    source = header.read_text(encoding="utf-8")
    block = re.search(r"enum\s+SDVersion\s*\{(.*?)\};", source, re.S)
    if not block:
        return ["sd1", "sd2", "sdxl", "sd3", "flux"]
    names = re.findall(r"VERSION_(\w+)", block.group(1))
    return sorted({n.lower() for n in names if n not in ("COUNT", "Count")})


# The runtimes that share one ggml, and so share its backend set.
GGML_RUNTIMES = {"llama.cpp", "whisper.cpp", "stable-diffusion.cpp"}


def main() -> int:
    existing = json.loads(ASSET.read_text(encoding="utf-8"))
    by_id = {r["id"]: r for r in existing["runtimes"]}

    readers = {
        "llama.cpp": (NATIVE / "llama.cpp", llama_architectures),
        "whisper.cpp": (NATIVE / "whisper.cpp", whisper_architectures),
        "stable-diffusion.cpp": (NATIVE / "stable-diffusion.cpp", sd_architectures),
        "kokoro": (NATIVE / "espeak-ng", kokoro_architectures),
    }

    removed_any = False
    for runtime_id, (repo, reader) in readers.items():
        if not repo.exists():
            print(f"  skip {runtime_id}: {repo} is not checked out")
            continue
        entry = by_id.get(runtime_id)
        if entry is None:
            print(f"  skip {runtime_id}: not in {ASSET.name}")
            continue

        tag, commit = describe(repo)
        architectures = reader(repo)
        removed = sorted(set(entry.get("architectures", [])) - set(architectures))
        if removed:
            removed_any = True
            print(f"  !! {runtime_id} DROPS {len(removed)} architecture(s): {', '.join(removed)}")

        entry["buildTag"] = tag
        entry["upstreamCommit"] = commit
        entry["architectures"] = architectures
        entry["installed"] = True
        # What CMakeLists compiles, which is a property of the APK and not of
        # any phone. arm64 additionally builds ggml's OpenCL backend; x86_64 is
        # the emulator, which has no Adreno behind it.
        #
        # Compiled is not the same as present: whether a device has a driver
        # behind libOpenCL.so is unknowable from here, so this list is only the
        # fallback. RuntimeRegistry.backendsFor asks the loaded binary first and
        # reports what ggml registered on the phone in front of it — SPEC 8.2,
        # do not assert what can be measured.
        #
        # The ONNX runtimes are CPU and only CPU, and that is not a gap in this
        # build. ONNX Runtime ships no OpenCL and no Vulkan execution provider
        # for Android at all, so there is no route from Kokoro or OmniVoice to
        # the Adreno. Its one accelerator here is Qualcomm's QNN, which needs a
        # different artifact — 174 MB of it — and then refuses these graphs
        # anyway: both are variable-length, and QNN's HTP backend rejects a
        # dynamic shape outright rather than partitioning around it.
        entry["backends"] = (
            ["CPU", "OPENCL"] if runtime_id in GGML_RUNTIMES else ["CPU"]
        )
        print(f"  {runtime_id}: {tag} ({commit}) — {len(architectures)} architectures")

    ASSET.write_text(json.dumps(existing, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if removed_any:
        print("\nArchitectures were removed. SPEC 17.6 #5 requires that to be deliberate.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
