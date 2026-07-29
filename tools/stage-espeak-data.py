#!/usr/bin/env python3
"""Stage the espeak-ng data tables the app ships in `assets/`.

espeak-ng's phoneme tables and pronunciation dictionaries are *compiled*
artifacts: the source tree carries `phsource/` and `dictsource/`, and turning
those into `phondata`/`en_dict` needs the `espeak-ng` binary itself. That binary
would have to be built for the *host* to run at build time, which a
cross-compile to Android has no reason to do.

The compiled tables are architecture-independent (little-endian, and every ABI
this app ships is little-endian), so they are staged here from an upstream
build instead, and their format version is asserted against the pinned source.

Full upstream data is ~17 MB across 100+ languages. This stages the subset
Kokoro can actually speak — English, plus the languages espeak phonemises well
enough to be worth offering. Japanese and Mandarin are deliberately absent:
Kokoro's own pipeline uses misaki for those, and espeak's output for them is
not what the model was trained on.

Usage:
    pip download espeakng-loader --no-deps -d <dir> && unzip the wheel
    python tools/stage-espeak-data.py <extracted>/espeakng_loader/espeak-ng-data
"""
import os
import shutil
import struct
import sys

# The version constant in native/espeak-ng/src/libespeak-ng/synthdata.c. If a
# runtime bump changes this, the staged tables must be restaged to match --
# espeak refuses to load data it did not compile itself.
VERSION_PHDATA = 0x014801

# Shared by every language.
CORE = ["phondata", "phonindex", "phontab", "intonations"]

# Kokoro v1.0's language prefixes: a/b English, e Spanish, f French, h Hindi,
# i Italian, p Brazilian Portuguese.
DICTS = ["en_dict", "es_dict", "fr_dict", "hi_dict", "it_dict", "pt_dict"]

# The voice definitions those dictionaries need, as <family>/<file>.
LANGS = [
    "gmw/en", "gmw/en-US", "gmw/en-GB-x-rp", "gmw/en-GB-scotland", "gmw/en-029",
    "roa/es", "roa/es-419", "roa/fr", "roa/fr-BE", "roa/pt", "roa/pt-BR", "roa/it",
    "inc/hi",
]


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    src = sys.argv[1]
    dst = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "app", "src", "main", "assets", "espeak-ng-data")
    dst = os.path.normpath(dst)

    # Refuse to stage tables the pinned library will reject at runtime. This
    # check is the whole reason the mismatch is not discovered on a handset.
    with open(os.path.join(src, "phondata"), "rb") as handle:
        version, rate = struct.unpack("<II", handle.read(8))
    if version != VERSION_PHDATA:
        print(f"phondata is version 0x{version:06x}, but the pinned espeak-ng "
              f"wants 0x{VERSION_PHDATA:06x}. Restage from a matching build.")
        return 1
    print(f"phondata version 0x{version:06x}, sample rate {rate} Hz -- matches the pinned source")

    if os.path.isdir(dst):
        shutil.rmtree(dst)
    os.makedirs(dst)

    total = 0
    for name in CORE + DICTS:
        shutil.copy2(os.path.join(src, name), os.path.join(dst, name))
        total += os.path.getsize(os.path.join(dst, name))
    for rel in LANGS:
        target = os.path.join(dst, "lang", rel)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.copy2(os.path.join(src, "lang", rel), target)
        total += os.path.getsize(target)

    print(f"staged {len(CORE) + len(DICTS) + len(LANGS)} files, {total / 1024:.0f} KiB -> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
