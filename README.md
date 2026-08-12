# EncoderBench

Benchmarks pretrained brain-MRI foundation encoders on clinical classification
tasks under one fixed protocol, so that differences between numbers come from the
encoder rather than from the preprocessing, the split, or the head.

Every encoder is read with **its own native preprocessing** at a
validation-selected depth, then evaluated in the same experiment families:

| Family | What is trained | Command |
| --- | --- | --- |
| Frozen probe | attention-pooling head only | `encoderbench probe` |
| End-to-end finetune | encoder + head, budgeted at ~16M trainable params | `encoderbench finetune` |
| Zero-shot VLM | nothing (free generation) | `encoderbench zero-shot` |
| Bridge to a frozen LLM | linear or resampler bridge into frozen MedGemma | `encoderbench bridge` |

**Encoders:** MedSigLIP, BrainGemma3D, MASS, BrainIAC, AnatCL.
**Baselines:** SynthSeg posteriors, and classical regional volumetry
([`scripts/volumetry_baseline.py`](scripts/volumetry_baseline.py)).
**Tasks:** ADNI AD/CN (`ad`), ADNI 3-class CN/MCI/AD, UCLA CNP SCZ/CN (`scz`),
BSNIP2 SZ/HC (`bsnip2`).

A structural MRI is not a clinical schizophrenia diagnostic tool; the SCZ and
BSNIP2 tasks measure research-cohort signal only.

## Repository layout

```text
encoders/
├── src/encoderbench/     the installable package — everything else depends on it
├── tests/                pytest unit tests over src/
├── scripts/              standalone drivers: manifests, preprocessing, caches,
│                         experiment runners, comparison tables
├── slurm/                three representative sbatch wrappers (examples only)
├── config/               YAML experiment configs
├── EEG_exp/              separate BIOT-on-EEG experiment, own README
└── data/manifests/       cohort CSVs live here (not published — see below)
```

The dependency direction is one-way: `slurm → scripts → src ← tests`. Nothing in
`src/` imports from `scripts/` or `slurm/`.

Inside `src/encoderbench/`:

| Module | Role |
| --- | --- |
| `cli.py`, `__main__.py` | the `encoderbench` command surface |
| `config.py` | YAML loading and invariant checks (fails before a run starts) |
| `manifest.py`, `preprocessing.py` | cohort CSVs, split construction, volume loading |
| `extractors.py`, `models.py` | frozen per-encoder token extraction, native grids |
| `cache.py` | `[n_volumes, tokens, dim]` feature caches |
| `training.py`, `linear_head.py` | attention-pooling head, probe training |
| `finetune.py`, `finetune_variants.py` | budgeted end-to-end finetuning |
| `cv_protocol.py`, `selection.py` | repeated-random-split protocol, model selection |
| `metrics.py`, `metrics_multiclass.py` | balanced accuracy, ROC-AUC, subject-cluster bootstrap |
| `llm.py`, `prompts.py`, `parsing.py`, `zero_shot.py` | frozen-VLM generation and scoring |
| `report.py` | seed aggregation and paired comparisons |
| `bsnip2/` | BSNIP2-specific manifest building and preprocessing |

## Before anything will run

This repository contains code only. Running it end-to-end additionally requires:

1. **Cohort data.** ADNI, UCLA CNP (ds000030), and BSNIP2 are access-controlled;
   apply through their own channels. Nothing here redistributes imaging data.
2. **Encoder checkpoints.** Each encoder's released weights, obtained from its
   own project. No model is downloaded at runtime — every Hugging Face load uses
   `local_files_only=True`.
3. **Manifests you build yourself.** `data/manifests/` ships empty. See below.
4. **Path edits.** [`config/default.yaml`](config/default.yaml) hardcodes absolute
   paths for the cluster this was developed on. Point `workspace_root`,
   `output_root`, `checkpoints:`, and `manifests:` at your own locations.

### Manifest format

Every cohort is a six-column CSV. This is the contract the whole pipeline runs on:

```csv
path,file_id,subject_id,group,is_repeat,split
/data/ADNI/0002_ss_box.nii.gz,0002,011_S_0002,CN,0,test
/data/ADNI/0003_ss_box.nii.gz,0003,011_S_0003,AD,0,train
```

- `subject_id` drives grouping, so repeat scans of one subject never straddle a split.
- `is_repeat` marks additional visits of an already-present subject.
- `split` is one of `train` / `validation` / `test`, assigned once and reused by
  every experiment, so runs stay comparable.

Build them with `encoderbench build-manifest`, or for the cohort-specific cases
with [`scripts/build_manifest_adni_full_mass.py`](scripts/build_manifest_adni_full_mass.py)
and [`src/encoderbench/bsnip2/build_manifest.py`](src/encoderbench/bsnip2/build_manifest.py).

## Installation

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -e '.[test]'
cp .env.example .env      # then edit the data paths in it
pytest
encoderbench --help
```

Python ≥3.10, Linux, and a CUDA GPU with bf16 support for the finetune, zero-shot,
and bridge paths. Frozen probes on a cached feature file run on CPU.

## Pipeline

Commands take the config before the subcommand:
`encoderbench --config config/default.yaml <subcommand> <task> <encoder>`.

**1. Build the manifest and audit it**

```bash
encoderbench build-manifest ad --source-manifest "$ADNI_SOURCE_MANIFEST"
encoderbench audit-data ad
encoderbench smoke ad mass --device cuda    # checkpoint keys, grid, dtype, finiteness
```

**2. Choose the read depth (validation only)**

```bash
python scripts/layer_sweep.py               # per-encoder tap selection
```

The test split is never touched here, so the choice stays honest. The winners are
recorded in `config/default.yaml` under `layers:`.

**3. Cache frozen features once**

```bash
encoderbench cache ad mass --layer 3 --native-tokens --device cuda \
  --out artifacts/cache/ad/mass_native.npz
```

`--native-tokens` skips the 4×4×4 pooling and keeps the encoder's own spatial grid
(MASS layer 3 → 4096 tokens). Omit it for the pooled 64-token protocol. Caches carry
manifest identity and load audits, and are reused by every supervised run below.

**4. Train**

```bash
encoderbench probe    ad mass --cache <cache> --seeds all --device cuda --out-dir <dir>
encoderbench finetune ad mass --manifest <csv> --cache <cache> --layer 3 \
  --native-tokens --warm-start-dir <probe-dir> --seeds all --device cuda --out-dir <dir>
```

Seeds 0/1/2 always run together and no seed is ever selected for reporting.
`--warm-start-dir` initializes the finetune head from that probe's own head, which
isolates the encoder's contribution from head initialization luck. Rerunning
resumes; `--restart` replaces.

**5. Aggregate**

```bash
python scripts/final_comparison.py
python scripts/compute_accuracy_columns.py    # adds plain accuracy next to balanced
encoderbench report
```

Balanced accuracy is insensitive to class skew by construction, so the gap between
the two accuracy columns reads out directly how much a number benefits from the
class balance. AUC is the primary metric; BA thresholds are fit on validation only.

Aggregate reports never copy subject IDs, paths, or per-row predictions.

## Other protocols

- **Repeated random splits** (instead of one fixed split):
  [`scripts/run_cv_bsnip2_mass.py`](scripts/run_cv_bsnip2_mass.py),
  driven by `cv_protocol.py` — six modes from frozen attention to full finetune.
- **3-class CN/MCI/AD:** [`scripts/run_probe_adni_mass_multiclass.py`](scripts/run_probe_adni_mass_multiclass.py)
  and `run_finetune_adni_mass_multiclass.py`, with `config/adni_mass_scaling_exp_multiclass.yaml`.
- **Dataset scaling:** the `config/adni_*_scaling*.yaml` series compares frozen-probe
  against joint-finetune as the training cohort grows.
  [`slurm/sbatch_scaling_exp_dataset2.sh`](slurm/sbatch_scaling_exp_dataset2.sh)
  shows the full A/B invocation, including self-resubmission on timeout.
- **EEG:** [`EEG_exp/`](EEG_exp/) is a self-contained BIOT-on-EEG_AD Alzheimer-vs-control
  experiment with its own README, requirements, and tests.

## Slurm

The three published scripts under [`slurm/`](slurm/) are **examples, not a portable
harness** — they hardcode a partition, a conda prefix, and absolute log paths. Read
them for the exact command sequences and hyperparameters; rewrite the headers for
your own scheduler. The remaining ~40 wrappers from the actual runs are kept
locally and git-ignored.

## Tests

```bash
pytest
python -m compileall -q src tests
```

Unit tests cover config invariants, split counts and repeat-scan isolation,
normalization and slice sampling, the hierarchical answer parser, forced-wrong
scoring, subject aggregation, paired bootstrap behavior, finetune parameter-group
selection, and the SynthSeg extractor. GPU checkpoint loading is deliberately an
explicit `encoderbench smoke` command rather than a unit test.

## Failure behavior

Invalid config invariants, missing manifests or checkpoints, unexpected encoder
keys, non-finite tokens, wrong grids, trainable parameters inside a frozen encoder,
and failed bridge gradient-isolation audits are all fatal and stop before or at the
start of a run. Inference exceptions during zero-shot generation become `ERR`,
retain their error text, and are forced wrong during scoring rather than dropped.

## Not in this repository

Imaging data, model checkpoints, built manifests, generated `artifacts/`,
run logs, and the internal planning and results documents.
