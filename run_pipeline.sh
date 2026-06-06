#!/usr/bin/env bash
# =============================================================================
# End-to-end reproduction of the Brain-Cancer-Prediction-Model analysis.
#
# Usage (from the project root):
#     bash run_pipeline.sh
#
# Prerequisites:
#   - Python deps:  python -m pip install -r requirements.txt
#   - R deps:       Rscript R-dependencies.R
#   - To execute the R notebooks you also need IRkernel registered with Jupyter.
#
# Outputs land in:  results/ (model objects), figures/ (plots), report/ (PDFs).
# =============================================================================
set -euo pipefail

# Always run from the project root (directory containing this script).
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python}"
RSCRIPT="${RSCRIPT:-Rscript}"
NBCONVERT="$PYTHON -m jupyter nbconvert --to notebook --execute --inplace"

echo "==> [1/5] Exploratory data analysis"
$RSCRIPT analysis/00_eda.R

echo "==> [2/5] Baseline models (Python notebooks)"
$NBCONVERT analysis/baselines/01_logistic_regression.ipynb
$NBCONVERT analysis/baselines/02_svm.ipynb
$NBCONVERT analysis/baselines/03_random_forest.ipynb

echo "==> [3/5] Multi-block models"
# Python cooperative learning
$NBCONVERT analysis/multiblock/01_cooperative_learning_python.ipynb
# SGCCA + the multinomial-lasso reference (standalone R scripts)
$RSCRIPT analysis/multiblock/02_sgcca.R
$RSCRIPT analysis/multiblock/07_multinomial_lasso_ungrouped.R
# R notebooks (require IRkernel). Comment out if the R kernel is unavailable.
$NBCONVERT analysis/multiblock/03_sgcca_lda.ipynb
$NBCONVERT analysis/multiblock/04_cooperative_lda.ipynb
$NBCONVERT analysis/multiblock/05_cooperative_ovr.ipynb
$NBCONVERT analysis/multiblock/06_multinomial_lasso.ipynb

echo "==> [4/5] Stability selection (bootstrap)"
$RSCRIPT analysis/stability/01_stability_bootstrap.R
$RSCRIPT analysis/stability/02_sgcca_stability_bootstrap.R

echo "==> [5/5] Figures and report PDF"
$PYTHON report/build_figures.py
$PYTHON report/build_figures_v2.py
$PYTHON report/build_fig7_cooperative_multinomial.py
$PYTHON report/build_final_comparison.py
$PYTHON report/build_pdf.py

echo "==> Pipeline complete. See results/, figures/ and report/."
