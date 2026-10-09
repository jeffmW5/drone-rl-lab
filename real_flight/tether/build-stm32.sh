#!/usr/bin/env bash
set -euo pipefail
root="${1:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
platform="${2:-cf2}"
case "$platform" in cf2|cf21bl) ;; *) echo 'Use cf2 or cf21bl' >&2; exit 2;; esac
target="$root/firmware/stm32-$platform"
mkdir -p "$target/src"
cp "$root/stm32/Makefile" "$root/stm32/app-config" "$root/stm32/Kbuild" "$target/"
cp "$root/stm32/src/"* "$target/src/"
cd "$target"
make "${platform}_defconfig"
make -j4
