#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${1:-$HERE/upstream}"
BASE=6edb48ce34f03dc5f2fa36c13f45e628a724d630
if [[ -e "$TARGET" ]]; then
  echo "Target already exists: $TARGET. Choose an empty path." >&2
  exit 1
fi
git clone https://github.com/Thinklab-SJTU/GenSCO.git "$TARGET"
git -C "$TARGET" checkout --detach "$BASE"
git -C "$TARGET" apply --check "$HERE/pisg.patch"
git -C "$TARGET" apply "$HERE/pisg.patch"
echo "PISG patch applied at $TARGET"
