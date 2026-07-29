#!/usr/bin/env bash
# Fetch the pinned upstream runtimes into native/.
#
# They are not checked in: three ggml-family repos come to roughly 400 MB of
# someone else's source, and every clone of this repo would pay for it. What is
# checked in is native/VERSIONS, which pins each one to an exact commit — so
# this script is reproducible in the way that matters.
#
#   ./tools/fetch-native.sh
#
# Then regenerate the architecture allowlist, which is derived from these
# sources rather than hand-maintained:
#
#   python tools/generate_runtimes.py

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

mkdir -p native

while read -r name url ref commit; do
    # Skip comments and blank lines.
    [[ -z "${name:-}" || "$name" == \#* ]] && continue

    dir="native/$name"
    if [[ -d "$dir/.git" ]]; then
        current="$(git -C "$dir" rev-parse HEAD)"
        if [[ "$current" == "$commit" ]]; then
            echo "  $name already at $ref ($commit)"
            continue
        fi
        echo "  $name: $current -> $commit"
        git -C "$dir" fetch --depth 1 origin "$commit"
        git -C "$dir" checkout --detach FETCH_HEAD
    else
        echo "  $name: cloning $ref"
        # Shallow at the ref, then verify the commit is the one we pinned —
        # a tag can be moved, a commit cannot.
        git clone --depth 1 --branch "$ref" --recursive "$url" "$dir"
    fi

    actual="$(git -C "$dir" rev-parse HEAD)"
    if [[ "$actual" != "$commit" ]]; then
        echo "!! $name is at $actual but VERSIONS pins $commit" >&2
        echo "   The ref moved. Update VERSIONS deliberately rather than building this." >&2
        exit 1
    fi
done < native/VERSIONS

echo "All runtimes at their pinned commits."
