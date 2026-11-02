#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$HERE/../reproduce/run.sh" --table main_tsp --mode sample --only fastt2t_tsp500_ts5_tg0 --seeds 41 --run-id example-fastt2t "$@"
