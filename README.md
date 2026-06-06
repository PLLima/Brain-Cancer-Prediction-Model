# Brain-Cancer-Prediction-Model

Interpretable multi-omics model for predicting the **localization of pediatric
high-grade gliomas** (cortical `cort`, diffuse intrinsic pontine glioma `dipg`, or
midline `midl`) from gene-expression and CGH data, with the longer-term aim of
identifying discriminant genes for targeted therapy.

## Dataset

IGR / Necker Enfants Malades cohort, **n = 53 patients** (39 train / 14 test), split
into two omics blocks:

| Block | Features | Description |
|---|---|---|
| **GE** | 15,702 | gene expression (Affymetrix microarray) |
| **CGH** | 1,229 | chromosomal alterations |

Three classes (`cort`, `dipg`, `midl`) in an extreme **p ≫ n** regime, with `midl`
strongly under-represented (8 train / 3 test). The pre-split CSVs live in
[`data/`](data/).

## Headline result

A **supervised sparse multi-block** pipeline — SGCCA followed by LDA — reaches a
held-out **test balanced accuracy of 0.924** (13/14 correct), matching the
state-of-the-art RGCCA reference (Girka et al., JSS 2025). Three other sparse
supervised methods reach the same cross-validation ceiling (≈ 0.83), showing the gain
comes from **per-block sparse supervised selection**, not from any single algorithm.
Full results and figures: [`report/RESULTS.md`](report/RESULTS.md).

## Repository structure

```
.
├── data/                 # input CSVs (GE / CGH / y, train + test) — do not edit
├── analysis/             # core analysis code (numbered by pipeline stage)
│   ├── 00_eda.R          # exploratory data analysis
│   ├── baselines/        # single-block & naive-fusion baselines (logreg, SVM, RF)
│   ├── multiblock/       # SGCCA, cooperative learning, multinomial lasso
│   ├── refinements/      # class-weighting (inverse-prevalence) for class imbalance
│   └── stability/        # bootstrap stability selection of stable genes
├── experiments/          # archived exploratory dead-ends (sparse PCA, fused/group lasso, SMOTE)
├── results/              # model outputs (.rds) produced by the analysis code
├── figures/              # generated figures (.png / .pdf)
├── report/               # write-up (RESULTS.md, METHODOLOGY.md) + figure/PDF build scripts
├── requirements.txt      # Python dependencies
├── R-dependencies.R      # R dependency installer
└── run_pipeline.sh       # end-to-end reproduction
```

The method-by-method walkthrough — including how each historical "NBxx" label maps to
a file — is in [`report/METHODOLOGY.md`](report/METHODOLOGY.md).

## Installation

**Python** (3.13 recommended):
```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate     |  macOS/Linux:  source .venv/bin/activate
python -m pip install -r requirements.txt
```

**R** (4.x):
```bash
Rscript R-dependencies.R
```
To execute the R *notebooks* (`analysis/multiblock/*.ipynb`), also register the R
Jupyter kernel once: in R, `IRkernel::installspec()`.

## Reproduction

Run the whole pipeline from the project root:
```bash
bash run_pipeline.sh
```
This runs, in order: EDA → baselines → multi-block models → stability selection →
figures and the report PDF. Outputs are written to `results/`, `figures/`, and
`report/`.

Each script and notebook resolves the project root automatically (it walks up to the
folder containing `data/`), so individual steps can also be run directly, e.g.:
```bash
Rscript analysis/00_eda.R
Rscript analysis/multiblock/02_sgcca.R
python report/build_pdf.py
```

## References

- Girka, Camenen, Peltier, Gloaguen, Guillemot, Le Brusquet, Tenenhaus —
  *RGCCA: A Unified Framework for Multiblock Data Analysis*, JSS 2025.
- Ding, Tibshirani, Hastie — *Cooperative Learning for Multiview Analysis*, PNAS 2022.
- Zou, Hastie, Tibshirani — *Sparse Principal Component Analysis*, 2006.

## License

See [LICENSE](LICENSE).
