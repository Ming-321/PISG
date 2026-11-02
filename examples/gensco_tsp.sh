#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$HERE/../reproduce/run.sh" --table gensco_tsp --mode sample --seeds 41 --run-id example-gensco "$@"
