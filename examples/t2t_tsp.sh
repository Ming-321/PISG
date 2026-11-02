#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$HERE/../reproduce/run.sh" --table main_tsp --mode sample --only t2t_tsp100_ts50_tg0 --seeds 41 --run-id example-t2t "$@"
