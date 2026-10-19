#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
python3 scripts/verify.py
python3 scripts/benchmark.py "$@"
python3 scripts/tables.py
python3 scripts/figures.py
