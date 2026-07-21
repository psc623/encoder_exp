# Encoders Test Experiment: Detailed Implementation Plan

## Requirement summary

This project implements the experiment matrix in `plan_v2.md` as a reproducible Python command-line package. It answers four separate questions and keeps their results separate: frozen-encoder attention probes, native VLM zero-shot classification, supervised frozen-encoder-to-frozen-MedGemma bridge tuning, and workflow transfer from ADNI AD/CN to UCLA CNP SCZ/CN.

### Technology stack

- Python 3.10+
- PyTorch for frozen encoders, attention heads, bridges, loss calculation, and training
- Hugging Face Transformers for MedSigLIP and MedGemma loading/processing
- MONAI for the released BrainIAC ViT architecture
- nibabel, NumPy, Pillow, and SciPy for NIfTI I/O, canonical RAS orientation, volume/slice processing, and 3D resampling
- pandas-free CSV/JSON/NPZ artifacts for portable manifests, predictions, audits, and cached features
- scikit-learn for ROC-AUC only; all other binary metrics are implemented explicitly
- PyYAML for one version-controlled experiment configuration
- pytest for unit tests

### Module breakdown

- `config`: validated YAML configuration and canonical workspace/checkpoint paths.
- `data`: manifest generation/validation, exact two-level subject splits, RAS volume normalization, native 2D/3D adapters, montage generation, and dataset audits.
- `encoders`: a common frozen-token-extractor interface plus MedSigLIP, BrainGemma3D, MASS, and BrainIAC implementations. Each returns spatial tokens and a native grid, and the shared path pools them to `4x4x4`.
- `models`: the common attention-pooling head, fixed 3D positional encoding, factorized linear bridge, and two-layer resampler bridge.
- `llm`: canonical disease prompts, chat-template-aware answer-token audit, answer-only sequence loss, candidate sequence scoring, and frozen MedGemma loading.
- `training`: deterministic attention-head and bridge training, weight-decay selection, class weighting, checkpoint selection, gradient accumulation, warmup, clipping, and early stopping.
- `evaluation`: output parsing, forced-wrong handling for `UNK`/`ERR`, volume and subject metrics, subject-cluster bootstrap intervals, paired bootstrap differences, and seed aggregation.
- `commands`/CLI: phase-gated manifest, audit, cache, probe, zero-shot, bridge, and report commands.

### Core functionality

1. Rebuild ADNI and UCLA manifests without modifying `brain_fm`, preserving repeat scans and exact subject-level split rules with split seed 0.
2. Audit all NIfTI inputs and checkpoint identifiers, save aggregate dataset diagnostics and representative montages, and reject subject leakage.
3. Load each encoder locally, freeze it, extract its required last spatial representation without CLS/global/projector/decoder tokens, verify grid round trips, pool to exactly 64 tokens, add a shared fixed positional encoding, and cache features once.
4. Train identical attention-pooling heads with train-only feature normalization, class-weighted cross entropy, the fixed AdamW protocol, weight-decay grid, seeds 0/1/2, and shuffled-label controls.
5. Run native MedGemma and BrainGemma3D free-generation zero-shot inference at temperature 0, save raw responses, and apply the pre-registered hierarchical parser.
6. Train either bridge capacity from cached 64-token features into a frozen canonical MedGemma language model using full answer-sequence class-weighted likelihood. Score A/B with complete candidate sequences and stable `logsumexp` normalization.
7. Produce distinct attention, zero-shot, linear-bridge, and resampler-bridge reports with all seed results, aggregate statistics, subject-cluster confidence intervals, and paired encoder differences.

### Data models

- Manifest row: `path`, `file_id`, `subject_id`, `group`, `is_repeat`, `split`.
- Feature-cache sample: 64-token float tensor plus file ID, subject ID, label, split, encoder name, original grid, feature dimension, and checkpoint audit metadata.
- Prediction row: file/subject IDs, true label, predicted label, positive-class probability when available, experiment metadata, and raw/match status for zero-shot runs.
- Run metadata: resolved configuration, seed, trainable/frozen parameter counts, selected epoch/weight decay, validation metrics/loss, checkpoint hashes, tokenizer audit, and gradient audit.
- Summary: volume and subject point estimates, cluster-bootstrap intervals, per-seed values, mean, sample standard deviation, median, and paired differences.

### External interfaces

- Primary interface: `python -m encoderbench <command> ...`.
- Inputs: YAML config, CSV manifests, local NIfTI paths, local Hugging Face checkpoints, MASS `.pth`, and BrainIAC `.ckpt`.
- Outputs: CSV manifests/predictions, compressed NPZ feature caches, PyTorch checkpoints, JSON audit/run/summary files, and PNG montages.
- No network service is required and model loading defaults to local files only.

## Ambiguities and missing information

1. The ADNI source manifest/data root and UCLA CNP data root are not present under `encoders` and no concrete UCLA path is frozen by the plan. The former `brain_fm/data/manifest.csv` is also absent in the current workspace.
2. The ADNI source manifest construction rules are referenced indirectly through `brain_fm`, but its current `build_manifest.py` depends on site-specific paths and cohort files not included here.
3. The exact encoder-native 3D resize for BrainGemma3D, MASS, and BrainIAC is not fixed in the plan.
4. The released BrainGemma3D code has multiple volume-size defaults and its native projector pools one-dimensional token order to 32 tokens; the plan requires spatial grid auditing and 64-token pooling for probes/bridges but the official projector path for native zero-shot.
5. The exact SCZ radiologist-style native zero-shot prose is not provided, only the required structure and disclaimer.
6. Attention-probe maximum epochs/checkpoint patience and the bootstrap replicate count are not explicitly specified. The existing `brain_fm` probe uses 300 epochs and its scorer uses 2,000 bootstrap samples.
7. Hardware-specific bridge micro-batch size is intentionally left open by the plan.
8. The expected output directory naming convention and report serialization format are unspecified.

## Default assumptions used by the implementation

> **ASSUMPTION A — input discovery:** ADNI is imported from a user-supplied source manifest with the six canonical columns. UCLA is built from user-supplied `participants.tsv` and BIDS root. Environment variables/config fields provide these site-specific paths; commands fail with actionable messages instead of guessing or changing cohorts.

> **ASSUMPTION B — native 3D adapters:** BrainGemma3D uses `(64,128,128)`, matching the architecture loader default; MASS uses `(128,128,128)`, matching released pretraining; BrainIAC uses `(96,96,96)`, matching its MONAI ViT definition. Each adapter uses the same setting for ADNI and UCLA, canonical RAS, whole-volume robust `[0,1]` normalization, and trilinear interpolation only.

> **ASSUMPTION C — feature grid:** all four cached representations are spatially pooled to `(4,4,4)` before either the attention probe or bridge. Native grids and pre-pooling token counts remain in audit metadata. This follows the Phase 0 requirement and the shared MedSigLIP extractor language in sections 6.1 and 9.2.

> **ASSUMPTION D — BrainGemma3D native zero-shot:** the released inflation code, official learned projector/`vis_scale`, canonical checkpoint language model, and its native 32-token projector protocol are retained because section 8 requires the official complete VLM. No LoRA adapter is loaded.

> **ASSUMPTION E — native prompts:** AD uses the existing `brain_fm` `radiologist` axial/24-slice prompt verbatim. SCZ mirrors that structure with schizophrenia cohort-classification wording and an explicit non-diagnostic research disclaimer.

> **ASSUMPTION F — probe/report defaults:** attention probes run at most 300 epochs and select the best validation balanced accuracy (lower validation loss breaks ties); bootstrap uses 2,000 subject resamples. The fixed weight-decay grid is `0, 1e-3, 1e-2, 1e-1, 1` from `brain_fm/src/probe.py`.

> **ASSUMPTION G — effective batch:** bridge effective batch is always 16. The config defaults to micro-batch 1 and accumulation 16; users may lower accumulation only when increasing micro-batch so their product remains 16, enforced by validation.

> **ASSUMPTION H — privacy:** prediction files remain private experiment artifacts. Generated aggregate reports omit ADNI paths, subject IDs, and participant images; montages stay in the audit output and are not copied into public reports.

> **ASSUMPTION I — startup verification:** CI/unit verification does not load multi-gigabyte checkpoints or require cohort data/GPU. Full phase smoke tests are separate explicit commands because successful scientific execution depends on site data availability and GPU memory.

> **ASSUMPTION J — BrainIAC CLS compatibility:** the released BrainIAC code constructs MONAI `ViT` with `classification=False`, so that checkpoint emits exactly 216 patch tokens and no learned CLS token despite its source comment calling the first patch “CLS.” The adapter accepts this released patch-only output; if a compatible checkpoint emits 217 tokens, it removes the first CLS token. Both cases must end at the required `6x6x6` spatial grid.
