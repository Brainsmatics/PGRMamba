#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export MAX_JOBS="${MAX_JOBS:-2}"
python - <<'PY'
from pathlib import Path
from torch.utils.cpp_extension import CUDA_HOME
if CUDA_HOME is None or not (Path(CUDA_HOME) / 'bin/nvcc').is_file():
    raise SystemExit('A CUDA development toolkit is required. Set CUDA_HOME to its directory and add CUDA_HOME/bin to PATH.')
print(f'Building CUDA extensions with {CUDA_HOME}')
PY
# Build this project's matching extension versions, rather than downloading
# generic wheels with potentially different Python/CUDA interfaces.
CAUSAL_CONV1D_FORCE_BUILD=TRUE python -m pip install --no-build-isolation --no-deps "$project_dir/causal-conv1d"
MAMBA_FORCE_BUILD=TRUE python -m pip install --no-build-isolation --no-deps "$project_dir/mamba"
