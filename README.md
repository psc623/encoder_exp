# EncoderBench

EncoderBench is the runnable implementation of [`plan_v2.md`](plan_v2.md). It evaluates four frozen brain-MRI encoders in four deliberately separate experiment families:

1. frozen encoder + identical supervised attention-pooling head;
2. native MedGemma and BrainGemma3D free-generation zero-shot classification;
3. frozen encoder + factorized linear bridge + frozen MedGemma;
4. frozen encoder + two-layer resampler bridge + frozen MedGemma.

The complete ADNI AD/CN workflow is run and configuration-locked before the same workflow transfers to UCLA CNP SCZ/CN. A structural MRI is not a clinical schizophrenia diagnostic tool; the SCZ task measures research-cohort signal only.

## Environment requirements

- Linux and Python 3.10 or newer
- CUDA GPU with bf16 support for MedGemma bridge and native VLM runs
- Enough GPU memory for the 4B MedGemma language backbone; feature extraction requirements vary by encoder
- Local checkpoints already present under `/net/projects2/litian-lab/scpan/model_weights`
- Read access to the ADNI `brain_fm` source manifest/volumes and UCLA CNP BIDS dataset

No model is downloaded at runtime: all Hugging Face loads use `local_files_only=True`.

## Installation

```bash
cd /net/projects2/litian-lab/scpan/encoders
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
cp .env.example .env
# Edit the three data paths, then export them into the shell.
set -a
source .env
set +a
pytest
encoderbench --help
```

For an HPC module/conda environment, omit virtual-environment creation and install the package into that environment. `requirements.txt` is equivalent to the editable command above.

## Configuration

[`config/default.yaml`](config/default.yaml) freezes all cross-encoder and cross-disease settings. Site-specific input locations belong in shell environment variables or command arguments, not in source code. The defaults use:

- axial `axis=2`, 24 slices, inclusive 15–85% sampling, whole slice;
- `(4,4,4)` pooled token grid;
- split seed 0 and training seeds 0/1/2;
- probe AdamW at `1e-3` over the original weight-decay grid;
- bridge AdamW at `1e-4`, weight decay `1e-2`, effective batch 16, 5% warmup, bf16, and patience 7.

The assumptions made where the source plan is underspecified are recorded in [`plan_detail.md`](plan_detail.md).

## Directory structure

```text
encoders/
├── .env.example
├── .gitignore
├── README.md
├── config/default.yaml
├── data/manifests/.gitkeep
├── plan_detail.md
├── plan_v2.md
├── pyproject.toml
├── requirements.txt
├── src/encoderbench/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cache.py
│   ├── cli.py
│   ├── config.py
│   ├── extractors.py
│   ├── llm.py
│   ├── manifest.py
│   ├── metrics.py
│   ├── models.py
│   ├── parsing.py
│   ├── phase.py
│   ├── preprocessing.py
│   ├── prompts.py
│   ├── report.py
│   ├── training.py
│   ├── utils.py
│   ├── workflows.py
│   └── zero_shot.py
└── tests/
    ├── test_config.py
    ├── test_manifest.py
    ├── test_metrics.py
    ├── test_models.py
    ├── test_parsing.py
    └── test_preprocessing.py
```

Generated artifacts are placed under `artifacts/` and are ignored by Git.

## Commands and usage examples

Every command accepts a global config before the subcommand, for example `encoderbench --config config/default.yaml cache ...`.

### 1. Build manifests

```bash
encoderbench build-manifest ad --source-manifest "$ADNI_SOURCE_MANIFEST"
encoderbench build-manifest scz --bids-root "$UCLA_BIDS_ROOT" \
  --participants "$UCLA_PARTICIPANTS_TSV"
```

The ADNI input is an existing canonical six-column `brain_fm` manifest; EncoderBench discards its split assignments, keeps AD/CN, verifies every volume, and regenerates the fixed two-level split. UCLA is joined directly from `participants.tsv` and raw BIDS T1w files. Missing UCLA files are saved beside the manifest as JSON and counts are recomputed rather than adjusted manually.

### 2. Dataset and Phase 0 audits

```bash
encoderbench audit-data ad
for encoder in medsiglip braingemma3d mass brainiac; do
  encoderbench smoke ad "$encoder" --device cuda
done
encoderbench token-audit ad
```

The data audit checks all images and writes aggregate counts, per-volume private audit records, orientations, shapes, spacing, normalized statistics, failures, and representative montages. Each encoder smoke test uses one positive and one CN case and checks the local checkpoint identifier, load-key status, frozen parameter count, native grid, flatten/reshape round trip, pooled shape, dtype, stability, and finite values. The tokenizer audit runs in the actual chat-template generation context.

### 3. Cache frozen features once

```bash
for encoder in medsiglip braingemma3d mass brainiac; do
  encoderbench cache ad "$encoder" --device cuda
done
```

Each cache has shape `[n_volumes,64,D_encoder]`, carries manifest identity and load audits, and is reused by every supervised run.

### 4. Attention probes and negative controls

```bash
for encoder in medsiglip braingemma3d mass brainiac; do
  encoderbench probe ad "$encoder" --seeds all --device cuda
  encoderbench probe ad "$encoder" --seeds all --shuffled-labels --device cuda
done
```

The command saves all seed checkpoints, test probabilities, validation selection details, volume/subject estimates, subject-cluster intervals, parameter counts, and token shapes. It never selects a seed for reporting.

### 5. Native zero-shot VLMs

```bash
encoderbench zero-shot ad medgemma --device cuda
encoderbench zero-shot ad braingemma3d --device cuda
```

Both paths use greedy free generation (`temperature=0`), retain full raw responses, apply the fixed hierarchical parser, and force `UNK`/`ERR` wrong without dropping rows. These results remain separate from supervised results.

### 6. Bridges

Run the linear seed-0 chain first, as required by the gate order, then complete all seeds and capacities:

```bash
encoderbench bridge ad medsiglip linear --seeds 0 --device cuda
for encoder in medsiglip braingemma3d mass brainiac; do
  encoderbench bridge ad "$encoder" linear --seeds all --device cuda
  encoderbench bridge ad "$encoder" resampler --seeds all --device cuda
done
```

Rerunning seed 0 deterministically replaces its same-named files. MedGemma remains frozen while its forward pass retains autograd so gradients reach only the bridge. The first training batch performs a gradient-isolation audit.

### 7. Lock AD and transfer to SCZ

```bash
encoderbench report
encoderbench lock-ad
```

`lock-ad` refuses to proceed until all 24 AD attention/control runs, 24 AD bridge runs, and two AD native zero-shot runs exist. It stores the configuration SHA-256. Every SCZ model command rejects a missing lock or a changed config. Then repeat the same commands with `scz`:

```bash
for encoder in medsiglip braingemma3d mass brainiac; do
  encoderbench smoke scz "$encoder" --device cuda
  encoderbench cache scz "$encoder" --device cuda
  encoderbench probe scz "$encoder" --seeds all --device cuda
  encoderbench probe scz "$encoder" --seeds all --shuffled-labels --device cuda
done
encoderbench token-audit scz
encoderbench zero-shot scz medgemma --device cuda
encoderbench zero-shot scz braingemma3d --device cuda
for encoder in medsiglip braingemma3d mass brainiac; do
  encoderbench bridge scz "$encoder" linear --seeds all --device cuda
  encoderbench bridge scz "$encoder" resampler --seeds all --device cuda
done
encoderbench report
```

## Artifacts and reporting

The report command creates `artifacts/reports/final_report.json` and `.md`. Attention, zero-shot, linear, and resampler results occupy different sections. Supervised sections contain every seed, mean, sample standard deviation, median, and paired subject-bootstrap differences on common test subjects. ADNI subject IDs, paths, montages, and row predictions are not copied into the aggregate report.

## Test execution

```bash
cd /net/projects2/litian-lab/scpan/encoders
source .venv/bin/activate
pytest
python -m compileall -q src tests
encoderbench --help
```

Unit tests cover exact UCLA split counts, repeat-scan isolation, input normalization and slice sampling, the hierarchical parser, forced-wrong scoring, subject aggregation, paired bootstrap behavior, configuration invariants, bridge shapes, and the no-activation linear bridge constraint. Multi-gigabyte checkpoint and cohort tests are deliberately explicit Phase 0 GPU smoke commands rather than unit tests.

## Failure handling

- Invalid config invariants stop before a run starts.
- Missing manifests/checkpoints/NIfTI files produce an actionable error.
- Dataset audit records corrupt files; manifest construction excludes only the UCLA files already required to be skipped by the plan.
- Unexpected encoder/projector keys, non-finite tokens, wrong grids, trainable frozen parameters, and failed gradient isolation are fatal.
- Inference exceptions become `ERR`, retain their error text, and are forced wrong during scoring.
- SCZ runs cannot bypass the AD configuration lock.
