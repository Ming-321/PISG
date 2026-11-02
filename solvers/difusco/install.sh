#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
conda env create -f "$HERE/environment.yml"
cd "$HERE/difusco/utils/cython_merge"
conda run --no-capture-output -n difusco python setup.py build_ext --inplace
