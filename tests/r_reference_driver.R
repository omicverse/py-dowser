#!/usr/bin/env Rscript
# R reference driver for py-dowser parity tests.
#
# Runs dowser 2.4.1 on its own bundled ExampleAirr dataset and writes the
# results that tests/test_r_parity.py compares against pydowser:
#   formatClones  -> per-clone sequence counts + germline lengths
#   getTrees      -> per-clone parsimony score + tree length + divergence
#   correlationTest on multi-timepoint clones
#
# Usage:  Rscript r_reference_driver.R <output_dir>

suppressMessages({
  library(dowser)
  library(phangorn)
})

args <- commandArgs(trailingOnly = TRUE)
out_dir <- if (length(args) >= 1) args[1] else "."
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

data(ExampleAirr)

# ---------------------------------------------------------------- formatClones
sel <- c("3128", "3100", "3141", "3170", "3184", "3192")
sub <- ExampleAirr[ExampleAirr$clone_id %in% sel, ]
clones <- formatClones(sub, traits = "sample_id")

fc <- data.frame(
  clone_id = clones$clone_id,
  seqs     = clones$seqs,
  germ_len = sapply(clones$data, function(x) nchar(x@germline)),
  stringsAsFactors = FALSE
)
write.csv(fc, file.path(out_dir, "formatclones.csv"), row.names = FALSE)

# ------------------------------------------------------------------- getTrees
trees <- getTrees(clones)
rows <- list()
div_rows <- list()
for (i in seq_len(nrow(trees))) {
  cl <- trees$data[[i]]
  tr <- trees$trees[[i]]
  seqs <- c(cl@data$sequence, cl@germline)
  names(seqs) <- c(cl@data$sequence_id, "Germline")
  sl <- strsplit(seqs, "")
  d <- phyDat(as.DNAbin(t(as.matrix(dplyr::bind_rows(sl)))))
  pscore <- parsimony(tr, d)
  rows[[i]] <- data.frame(
    clone_id   = trees$clone_id[i],
    tips       = length(tr$tip.label),
    parsimony  = pscore,
    tree_len   = sum(tr$edge.length),
    stringsAsFactors = FALSE
  )
  div <- getDivergence(tr)
  div_rows[[i]] <- data.frame(
    clone_id = trees$clone_id[i],
    tip      = names(div),
    divergence = as.numeric(div),
    stringsAsFactors = FALSE
  )
}
write.csv(do.call(rbind, rows), file.path(out_dir, "gettrees.csv"),
          row.names = FALSE)
write.csv(do.call(rbind, div_rows), file.path(out_dir, "divergence.csv"),
          row.names = FALSE)

# ------------------------------------------------------------ correlationTest
g <- tapply(ExampleAirr$timepoint, ExampleAirr$clone_id,
            function(x) length(unique(x)))
multi <- names(g[g > 1])
csub <- ExampleAirr[ExampleAirr$clone_id %in% multi, ]
cclones <- formatClones(csub, traits = "timepoint")
ctrees <- getTrees(cclones)
set.seed(1)
ct <- correlationTest(ctrees, permutations = 1000, time = "timepoint")
cor_df <- data.frame(
  clone_id    = ct$clone_id,
  seqs        = ct$seqs,
  correlation = ct$correlation,
  slope       = ct$slope,
  p           = ct$p,
  nclust      = ct$nclust,
  stringsAsFactors = FALSE
)
write.csv(cor_df, file.path(out_dir, "correlationtest.csv"), row.names = FALSE)

cat("R reference driver complete.\n")
