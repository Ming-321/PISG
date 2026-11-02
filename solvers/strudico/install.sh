#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
conda env create -f "$HERE/environment.yml"
cd "$HERE/diffusion/utils/cython_merge"
conda run --no-capture-output -n strudico python setup.py build_ext --inplace
