# Logit-Path Tracing Clean Code

This directory contains the compact, paper-relevant implementation used for
the reproducibility package.

It provides:

- exact next-token argmax partitions for affine logit controls on an interval
  or a two-dimensional simplex;
- epsilon-robust subregion clipping under the paper's pairwise-gap error
  bound;
- a smoke test that compares the recovered cells with dense-grid argmax
  decoding on toy affine logits;
- a paper-number reproduction script for the released experiment artifacts.

## Quick Start

```bash
python -m pip install -e .
python scripts/smoke_lpt.py --out-dir outputs/smoke
```

The smoke test writes `outputs/smoke/smoke_summary.json` and exits with a
nonzero status if any dense-grid point disagrees with the recovered partition.

To recompute paper tables from the released experiment artifacts:

```bash
python scripts/reproduce_paper_numbers.py \
  --artifact-root /path/to/extracted_artifacts \
  --out-dir outputs/recomputed_tables
```

To regenerate the deterministic property-existence table:

```bash
python scripts/derive_property_existence.py \
  --artifact-root /path/to/extracted_artifacts \
  --out-dir outputs/recomputed_tables
```
