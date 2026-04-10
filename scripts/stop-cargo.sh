#!/usr/bin/env bash
# Stop cargo/rustc before building to avoid target/ file locks (required workflow).
# Run from repo root: ./scripts/stop-cargo.sh
set -euo pipefail

stopped=0
if command -v killall >/dev/null 2>&1; then
  for name in cargo rustc; do
    if killall "$name" 2>/dev/null; then
      stopped=1
      echo "Stopped: $name"
    fi
  done
elif command -v pkill >/dev/null 2>&1; then
  for name in cargo rustc; do
    if pkill -x "$name" 2>/dev/null; then
      stopped=1
      echo "Stopped: $name"
    fi
  done
else
  echo "Neither killall nor pkill found; stop cargo/rustc manually." >&2
  exit 1
fi

if [[ "$stopped" -eq 0 ]]; then
  echo "No cargo/rustc processes found."
fi
