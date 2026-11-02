#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$HERE/../reproduce/run.sh" --table main_tsp --mode sample --only difusco_tsp100_ts100_tg0 --seeds 41 --run-id example-difusco "$@"
