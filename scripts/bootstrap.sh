#!/usr/bin/env bash
# Install the pinned Basalt release in this checkout, then sync Python deps.
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
basalt_version="0.1.7"
basalt_sha256="8ab56b2ab27315a9c64cb8ae9c99d0d4d894e2cc87bc9841d8defe30b86e6d24"
basalt_prefix="$project_dir/.deps/basalt/$basalt_version"
basalt_archive="basalt-$basalt_version-x86_64-unknown-linux-gnu.tar.gz"
basalt_url="https://gitlab.com/VladyslavUsenko/basalt/-/releases/$basalt_version/downloads/$basalt_archive"
sync_python=true

fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

case "${1:-}" in
    --basalt-only) sync_python=false; shift ;;
    -h|--help)
        printf '%s\n' \
            'Usage: ./scripts/bootstrap.sh [--basalt-only]' \
            'Target: Ubuntu 22.04+ on x86_64 (Linux glibc).' \
            'Installs Basalt 0.1.7 into .deps/basalt/0.1.7 in this checkout.' \
            'Default: also runs uv sync --frozen --python 3.10 into .venv.' \
            'Prerequisites: curl, tar, sha256sum, ffmpeg/ffprobe, and uv.' \
            'Does not edit shell profiles or install system packages.'
        exit 0 ;;
    "") ;;
    *) fail "Unknown argument: $1 (see --help)" ;;
esac
[[ $# -eq 0 ]] || fail 'Unexpected extra arguments (see --help).'
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || \
    fail 'This pinned archive supports Linux x86_64 only.'

for dependency in curl tar sha256sum ffmpeg ffprobe; do
    command -v "$dependency" >/dev/null || \
        fail "Missing $dependency. On Ubuntu: sudo apt-get install curl ca-certificates tar coreutils ffmpeg"
done
if "$sync_python"; then
    command -v uv >/dev/null || fail 'Install uv first: https://docs.astral.sh/uv/getting-started/installation/'
    [[ -f "$project_dir/uv.lock" ]] || fail 'Missing uv.lock; run from a complete project checkout.'
fi

smoke_test_basalt() {
    local prefix="$1"
    [[ -x "$prefix/bin/basalt_vio" && -d "$prefix/lib" && \
       -f "$prefix/etc/basalt/euroc_config.json" ]] || \
        fail "Incomplete Basalt installation: $prefix"
    if ! env LD_LIBRARY_PATH="$prefix/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        "$prefix/bin/basalt_vio" --help >/dev/null; then
        fail 'Basalt cannot start. Check the loader error above. On Ubuntu, common runtime packages are: libegl1 libgl1 libglu1-mesa libx11-6 libxcursor1 libxinerama1 libxrandr2 libxi6 libxtst6'
    fi
}

if [[ -e "$basalt_prefix" ]]; then
    [[ -f "$basalt_prefix/.archive-sha256" ]] || \
        fail "Unrecognized existing directory: $basalt_prefix. Move it aside before retrying."
    [[ "$(< "$basalt_prefix/.archive-sha256")" == "$basalt_sha256" ]] || \
        fail "Release checksum receipt mismatch: $basalt_prefix. Move it aside before retrying."
    smoke_test_basalt "$basalt_prefix"
    printf 'Basalt %s already installed: %s\n' "$basalt_version" "$basalt_prefix"
else
    mkdir -p "$project_dir/.deps/basalt"
    bootstrap_tmp="$(mktemp -d "$project_dir/.deps/basalt/.install.XXXXXXXX")"
    cleanup() {
        # Only remove the private staging directory created by mktemp above.
        if [[ -n "${bootstrap_tmp:-}" && \
              "$bootstrap_tmp" == "$project_dir/.deps/basalt/.install."* && \
              -d "$bootstrap_tmp" ]]; then
            rm -rf -- "$bootstrap_tmp"
        fi
    }
    trap cleanup EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    printf 'Downloading Basalt %s...\n' "$basalt_version"
    curl --fail --location --show-error --retry 3 --connect-timeout 20 \
        --max-time 600 --output "$bootstrap_tmp/basalt.tar.gz" "$basalt_url"
    printf '%s  %s\n' "$basalt_sha256" "$bootstrap_tmp/basalt.tar.gz" | sha256sum --check --status || \
        fail 'Basalt archive SHA256 mismatch; installation stopped.'
    tar -xzf "$bootstrap_tmp/basalt.tar.gz" -C "$bootstrap_tmp" --no-same-owner
    [[ -d "$bootstrap_tmp/release/data" ]] || fail 'Unexpected Basalt archive layout.'
    mkdir -p "$bootstrap_tmp/release/etc"
    mv -- "$bootstrap_tmp/release/data" "$bootstrap_tmp/release/etc/basalt"
    smoke_test_basalt "$bootstrap_tmp/release"
    printf '%s\n' "$basalt_sha256" > "$bootstrap_tmp/release/.archive-sha256"
    mv -T -- "$bootstrap_tmp/release" "$basalt_prefix"
    printf 'Installed Basalt: %s\n' "$basalt_prefix"
fi

if "$sync_python"; then
    cd -- "$project_dir"
    # Ignore an activated environment or custom uv target; use this project's .venv.
    env -u VIRTUAL_ENV -u UV_PROJECT_ENVIRONMENT uv sync --project "$project_dir" --frozen --python 3.10
    "$project_dir/.venv/bin/python" -c \
        'import cv2, mediapipe, mcap, numpy; print("Python imports OK")'
fi
printf '\nBasalt ready: %s\n' "$basalt_prefix/bin/basalt_vio"
if "$sync_python"; then
    printf 'From %s run:\n  uv run main.py --mode process --mcap-path /path/to/raw.mcap\n' "$project_dir"
else
    printf 'Python setup skipped. Run bootstrap.sh without --basalt-only to sync .venv.\n'
fi
