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

# -------------------------------------------------------------- createGermlines
# Reconstruct a germline from a synthetic AIRR row whose V/D/J genes exactly
# match dowser's bundled minimal 3-gene IMGT reference, so pydowser and R run
# on identical input and the result must be bit-exact.
vdj_dir <- system.file("extdata", "germlines", "imgt", "human", "vdj",
                       package = "dowser")
imgt <- readIMGT(vdj_dir)
sv <- toupper(imgt$IGH$V[["IGHV3-11*05"]])
sd <- toupper(imgt$IGH$D[["IGHD3-10*01"]])
sj <- toupper(imgt$IGH$J[["IGHJ5*02"]])
np1 <- 5; np2 <- 7
fullseq <- paste0(sv, paste(rep("A", np1), collapse = ""), sd,
                  paste(rep("C", np2), collapse = ""), sj)
gdf <- data.frame(
  sequence_id = "s1", sequence_alignment = fullseq, clone_id = "c1",
  v_call = "IGHV3-11*05", d_call = "IGHD3-10*01", j_call = "IGHJ5*02",
  v_germline_start = 1, v_germline_end = nchar(sv),
  d_germline_start = 1, d_germline_end = nchar(sd),
  j_germline_start = 1, j_germline_end = nchar(sj),
  np1_length = np1, np2_length = np2, locus = "IGH",
  stringsAsFactors = FALSE)
gres <- createGermlines(gdf, imgt)
write.csv(data.frame(
  germline_full  = gres$germline_alignment,
  germline_dmask = gres$germline_alignment_d_mask,
  stringsAsFactors = FALSE),
  file.path(out_dir, "creategermlines.csv"), row.names = FALSE)

# ---------------------------------------------------------------- maskCodons
mc1 <- maskCodons("s1", "ACGACG", "ACGACG")
mc2 <- maskCodons("s2", "ACGTTTACG", "ACGACG")
mc3 <- maskCodons("s4", "ACGTTACGTAC", "ACGACGTAC")
mask_df <- data.frame(
  case        = c("identical", "clean_insertion", "partial_gap"),
  masked      = c(ifelse(is.na(mc1$sequence_masked_v), "NA", mc1$sequence_masked_v),
                  ifelse(is.na(mc2$sequence_masked_v), "NA", mc2$sequence_masked_v),
                  ifelse(is.na(mc3$sequence_masked_v), "NA", mc3$sequence_masked_v)),
  note        = c(mc1$masking_note, mc2$masking_note, mc3$masking_note),
  stringsAsFactors = FALSE)
write.csv(mask_df, file.path(out_dir, "maskcodons.csv"), row.names = FALSE)

# ---------------------------------------------------------------- makeSkyline
# Pure-algorithm skyline: build a synthetic BEAST log + trees file and run R
# makeSkyline on it; pydowser must match bit-exactly.
sky_dir <- file.path(out_dir, "sky")
dir.create(sky_dir, showWarnings = FALSE)
writeLines(c(
  "Sample\tTreeHeight\tbPopSizes.1\tbPopSizes.2\tbGroupSizes.1\tbGroupSizes.2",
  "0\t1.0\t10.0\t20.0\t2\t2",
  "1\t1.2\t12.0\t22.0\t2\t2",
  "2\t0.9\t9.0\t19.0\t2\t2",
  "3\t1.1\t11.0\t21.0\t2\t2"), file.path(sky_dir, "x.log"))
writeLines(c(
  "#NEXUS", "BEGIN TREES;", "TRANSLATE",
  "1 t1,", "2 t2,", "3 t3,", "4 t4,", "5 t5;",
  "tree STATE_0 = ((1:0.5,2:0.5):0.5,(3:0.3,(4:0.1,5:0.1):0.2):0.7):0.0;",
  "tree STATE_1 = ((1:0.6,2:0.6):0.6,(3:0.4,(4:0.2,5:0.2):0.2):0.8):0.0;",
  "tree STATE_2 = ((1:0.45,2:0.45):0.45,(3:0.25,(4:0.1,5:0.1):0.15):0.65):0.0;",
  "tree STATE_3 = ((1:0.55,2:0.55):0.55,(3:0.35,(4:0.15,5:0.15):0.2):0.75):0.0;",
  "END;"), file.path(sky_dir, "x.trees"))
sky <- makeSkyline(file.path(sky_dir, "x.log"), file.path(sky_dir, "x.trees"),
                   burnin = 0, bins = 10, clone_id = "c1", max_height = "min")
write.csv(as.data.frame(sky), file.path(out_dir, "makeskyline.csv"),
          row.names = FALSE)

cat("R reference driver complete.\n")
