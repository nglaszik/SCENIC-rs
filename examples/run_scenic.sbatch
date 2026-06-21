#!/bin/bash
#SBATCH --job-name=scenic-rs
#SBATCH --cpus-per-task=64
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=scenic_rs_%j.log
# Full scenic-rs run (grnboost2 -> ctx -> AUCell) on the whole dataset, no Dask.
# Submit:  sbatch examples/run_scenic.sbatch
# Override args by appending them, e.g.:
#   sbatch examples/run_scenic.sbatch --mask-dropouts --min-cells 5
#
# Notes on resources:
#   - GRN is the long pole (scales with cells × genes × TFs); 24h is generous.
#   - scenic-rs memory is flat in cores, so bump --cpus-per-task freely to go
#     faster. Keep --mem ~96G (h5ad load + dense matrix + two ranking DBs).
set -euo pipefail

# scenic-rs uses rayon; pin it to the cores SLURM gave us. Keep BLAS single-thread.
export RAYON_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

PY="${SCENIC_RS_PYTHON:-$HOME/venvs/base/bin/python}"   # env with scenic_rs + anndata
SCRIPT="$HOME/jupyternotebooks/scenic-rs/examples/run_full_scenic.py"

echo "host=$(hostname)  cores=$RAYON_NUM_THREADS  python=$PY"
"$PY" "$SCRIPT" "$@"
