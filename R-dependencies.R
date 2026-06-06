# R package dependencies for the analysis scripts and R notebooks.
# Install once with:  Rscript R-dependencies.R
#
# Covers:
#   - SGCCA / multi-block      : RGCCA, mixOmics
#   - Cooperative learning     : multiview
#   - Penalised regression     : glmnet, gglasso, elasticnet
#   - CV / utilities / I/O     : caret, data.table, FNN, MASS
#   - Plotting                 : ggplot2, reshape2
#
# To execute the R *notebooks* (analysis/multiblock/*.ipynb) you also need the
# IRkernel package registered with Jupyter:  IRkernel::installspec()

repos <- "https://cloud.r-project.org"

cran_pkgs <- c(
  "RGCCA", "multiview", "glmnet", "gglasso", "elasticnet",
  "caret", "data.table", "FNN", "MASS", "ggplot2", "reshape2"
)

to_install <- setdiff(cran_pkgs, rownames(installed.packages()))
if (length(to_install)) {
  message("Installing CRAN packages: ", paste(to_install, collapse = ", "))
  install.packages(to_install, repos = repos)
}

# mixOmics is distributed through Bioconductor, not CRAN.
if (!requireNamespace("mixOmics", quietly = TRUE)) {
  if (!requireNamespace("BiocManager", quietly = TRUE)) {
    install.packages("BiocManager", repos = repos)
  }
  BiocManager::install("mixOmics", update = FALSE, ask = FALSE)
}

message("R dependencies are installed.")
