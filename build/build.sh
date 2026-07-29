#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$root/dist"
mojo build --emit shared-lib "$root/src/argon2.mojo" \
  -o "$root/dist/libmojo-argon2-cffi.so"
