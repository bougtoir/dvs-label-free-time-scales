# Data-derived time scales for label-free anomaly detection in sparse event streams

This repository contains the analysis code, the result files and the table and figure code for the paper
"Data-derived time scales for label-free anomaly detection in sparse event streams" (T. Onishi). With them you
can re-run the analyses from the public raw data and regenerate every number, table and figure from the result
files. The manuscript and the submission documents are not part of this repository.

## Contents

| Path | Content |
|------|---------|
| `event_driven.py` | the proposed event-driven Poisson detector (per-pixel rates from inter-event intervals, rate-adaptive window, exact upper-tail p-value) and the event-driven ablations |
| `sp_evaluation.py` | EBSSA loader, labels from the published object annotations, comparators, grouped nested cross-validation, paired tests (`results/sp_evaluation_summary.json`, `results/sp_per_recording.json`) |
| `zero_knowledge.py` | fixed deployment without labels, cross-sensor transfer, threshold transfer (`results/zero_knowledge.json`) |
| `hybrid_screening.py` | window handed over to YNoise and the two-stage cascade (`results/hybrid_screening.json`) |
| `operating_points.py` | partial AUC and operating points (`results/partial_auc.json`, `results/self_calibration.json`) |
| `sparsity_cost.py` | cost scaling on thinned streams (`results/sparsity_cost.json`) |
| `qualitative_example.py` | per-stage example images (`results/qualitative_example.json`, `.npz`) |
| `dnd21_evaluation.py`, `aedat2.py` | DND21 mixtures, leave-one-signal-source-out evaluation, AEDAT-2.0 reader (`results/dnd21_evaluation_summary.json`, `results/dnd21_per_recording.json`) |
| `aednet_transfer.py` | zero-shot transfer of the released AEDNet weights (`results/aednet_transfer.json`) |
| `comparator_feasibility.py` | comparator metadata of Supplementary Table S1 (`results/comparator_feasibility.json`) |
| `pr_effect_sizes.py` | paired effect sizes and bootstrap confidence intervals (`results/pr_effect_sizes.json`) |
| `revision_r1/r1_ebssa_analyses.py` | pass fraction versus FPR/TPR, one-factor sensitivity, Poisson diagnostics, end-to-end timing (`results/r1_ebssa_analyses.json`) |
| `revision_r1/r1_misfit_sims.py` | controlled departures from the homogeneous Poisson background (`results/r1_misfit_sims.json`) |
| `revision_r1/r1_dnd21_noise_diagnostics.py` | diagnostics on the DND21 noise-only recordings (`results/r1_dnd21_noise_diagnostics.json`) |
| `revision_r1/semantic_audit.py`, `r1_zero_count.py`, `r1_quantile_bound.py`, `r1_summarise.py` | window/tie semantics and null calibration, zero-count correction, finite-sample bound of the pass fraction, summaries (`results/r1_*.json`) |
| `paper/paper_numbers*.py` | every number quoted in the paper, read from `results/*.json` |
| `paper/make_tables.py` | Tables 1-3 and S1-S11 (`paper/tables.json`, `paper/tables/*.csv`) and `paper/numbers.json` |
| `paper/make_figures.py`, `paper/figure_base.py` | the main and supplementary figures and the graphical abstract (`paper/fig*.png`, 300 dpi, and TIFF; `paper/figure_text.json`) |
| `results/SHA256SUMS` | SHA-256 of every tracked result file |
| `revision_r1/DATA_ACQUISITION_LEDGER.json` | source, retrieval time, size and SHA-256 of every raw file used for the reproduction |
| `tests/test_aedat2_pfd.py` | unit test of the AEDAT-2.0 reader and the PFD comparator |

## Data

The raw data are not redistributed here. Download them from the publishers and use them under the publishers'
conditions; the ledger records the SHA-256 of the files used.

* **EBSSA** (Afshar et al., IEEE Sensors Journal 2020), labelled split, `labelled_ebssa.h5` (12,106,528,328 bytes,
  SHA-256 `50a54035...` in full in `sp_evaluation.py`). `sp_evaluation.load_recordings` downloads it through
  `tonic.datasets.EBSSA` (Tonic 1.6.0) into `data/EBSSA/`. If the Google Drive quota blocks the download, the same
  file is served by the authors' institutional mirror (`EBSSA_H5_MIRROR` in `sp_evaluation.py`); save it as
  `data/EBSSA/labelled_ebssa.h5`.
* **DND21** (Guo and Delbruck, IEEE TPAMI 2022, https://sites.google.com/view/dnd21): four signal recordings and two
  DAVIS346 noise recordings of the public release. `dnd21_evaluation.py` downloads them with `gdown` into
  `data/dnd21/` (file ids in `dnd21_evaluation.FILES`).
* **AEDNet** (Fang et al., ACM Multimedia 2022): `aednet_transfer.py` clones https://github.com/Fanghuachen/AEDNet at a
  pinned commit and checks the SHA-256 of the released weights.

## Reproduction

```bash
pip install -r requirements.txt          # Python 3.10 was used; the torch CPU wheel comes from the PyTorch CPU index
python3 sp_evaluation.py                 # EBSSA nested CV with all comparators and supervised networks (several hours)
python3 zero_knowledge.py                # about 1.5 h on 2 cores
python3 hybrid_screening.py
python3 operating_points.py
python3 sparsity_cost.py
python3 qualitative_example.py
python3 dnd21_evaluation.py              # about 1.5 h on 2 cores
python3 aednet_transfer.py
python3 comparator_feasibility.py
python3 pr_effect_sizes.py
python3 revision_r1/r1_ebssa_analyses.py # add --no-timing to skip the timing runs
python3 revision_r1/r1_misfit_sims.py
python3 revision_r1/r1_dnd21_noise_diagnostics.py
python3 revision_r1/semantic_audit.py
python3 revision_r1/r1_zero_count.py
python3 revision_r1/r1_quantile_bound.py
python3 revision_r1/r1_summarise.py
python3 paper/make_tables.py             # tables and numbers from results/*.json (seconds)
python3 paper/make_figures.py            # figures from results/*.json (about a minute)
pip install pytest && python3 -m pytest -q tests
```

The table and figure steps need only the files in `results/`; run them alone to regenerate the tables and figures
from the tracked results. Score caches are written to `results/sp_cache/` (git-ignored); delete it for a run from
scratch.

Seeds are fixed in the scripts: `SEED = 42` in `sp_evaluation.py`, `sparsity_cost.py` and `pr_effect_sizes.py`
(also used by `dnd21_evaluation.py` and `aednet_transfer.py`), the supervised networks use
`torch.manual_seed(SEED + fold)`, `r1_misfit_sims.py` 20261120, `semantic_audit.py` 20261119,
`r1_quantile_bound.py` 20261121 and `r1_zero_count.py` 1. Deterministic results reproduce exactly; the supervised CPU
networks are not guaranteed to be bitwise reproducible, and wall-clock timings depend on the machine.
