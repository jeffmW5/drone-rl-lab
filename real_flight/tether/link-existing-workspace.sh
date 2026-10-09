#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
previous="${1:-/home/n33du/ai/crazyflie-tether}"
for entry in vendor runs generated firmware; do
  if [[ ! -d "$previous/$entry" ]]; then
    echo "Missing existing artifact directory: $previous/$entry" >&2
    exit 1
  fi
  if [[ -e "$entry" || -L "$entry" ]]; then
    echo "Refusing to overwrite: $PWD/$entry" >&2
    exit 1
  fi
done
for entry in vendor runs generated firmware; do
  ln -s -- "$previous/$entry" "$entry"
done
if [[ -f "$previous/sdk-image.json" && ! -e sdk-image.json ]]; then
  cp -- "$previous/sdk-image.json" sdk-image.json
fi
echo "Linked ignored local artifacts. Source files remain in this checkout."
