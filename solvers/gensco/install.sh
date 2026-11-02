#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
conda env create -f "$HERE/environment.yml"
conda run --no-capture-output -n gensco python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
bash "$HERE/apply.sh"
conda run --no-capture-output -n gensco make -C "$HERE/upstream/lib"
