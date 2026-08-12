# BIOT on EEG_AD: Alzheimer (A) vs Control (C)

This experiment evaluates the released BIOT EEG encoder on subject-level
Alzheimer-vs-control classification. Group F (FTD) is excluded before any
split or preprocessing.

## What the pipeline fixes

1. Reads the embedded MATLAB-v5 EEGLAB `.set` files with SciPy.
2. Uses the denoised `derivatives/` recordings by default.
3. Converts the 19 referential electrodes to BIOT's ordered 16 bipolar montage.
4. Maps legacy `T3/T4/T5/T6` locations to modern `T7/T8/P7/P8` semantics.
5. Resamples 500 Hz to 200 Hz with polyphase anti-alias filtering.
6. Creates non-overlapping 10-second `[16, 2000]` windows.
7. Normalizes each window/channel by its absolute-amplitude 95th percentile.
8. Splits subjects before windows and aggregates window probabilities by subject.
9. Selects the BA threshold on validation subjects only.
10. Reports subject-level balanced accuracy and ROC-AUC.

Cache construction is sequential by default. The supplied Slurm script uses
four preprocessing workers; cached arrays are reused on later runs.

The self-contained BIOT implementation preserves the released checkpoint key
structure and strictly loads all encoder tensors. It does not require MNE or the
unpinned upstream `linear_attention_transformer` package.

## Why the original inputs do not directly match

EEG_AD stores 19 referential channels at 500 Hz, whereas the released BIOT EEG
model expects its first 16 channel-token IDs to have a specific bipolar order
and was trained with 200 Hz, 10-second samples. Passing the original matrix
directly would therefore give BIOT the wrong channel meaning, sampling grid,
length, and scale. This pipeline resolves all four mismatches explicitly:

- EEG_AD `T3/T4/T5/T6` are interpreted as modern `T7/T8/P7/P8` positions.
- The 19 referential electrodes are differenced into BIOT's 16 bipolar channels.
- Polyphase resampling changes 500 Hz to 200 Hz before segmentation.
- Each 10-second channel is divided by its own absolute 95th percentile, exactly
  matching the released BIOT data loaders' amplitude normalization.

The conversion is feasible because EEG_AD contains every electrode needed for
the 16 bipolar derivations. No channel is fabricated or silently padded.

## Pretrained parameters

No additional download is needed in this workspace. The default command loads:

```text
/net/projects2/litian-lab/scpan/github_repo/BIOT/pretrained-models/EEG-six-datasets-18-channels.ckpt
```

It contains encoder parameters only. The binary A/C classification head is new
and must be trained on EEG_AD; BIOT cannot perform zero-shot A/C classification.

## Environment

The existing project environment already has all required packages:

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate /net/projects2/litian-lab/scpan/encoders/med310
cd /net/projects2/litian-lab/scpan/encoders/EEG_exp
```

## Inspect and smoke-test preprocessing

```bash
python train_biot_ac.py inspect --subjects 2
```

This preprocesses two subjects, validates the output, strictly loads the
six-dataset checkpoint, and performs one BIOT forward pass.

## Main experiment

Run the recommended pretrained-frozen, pretrained-fine-tuned, and from-scratch
comparison:

```bash
sbatch run_biot_ac.slurm
```

Or run one regime/fold interactively:

```bash
python train_biot_ac.py train \
  --regimes pretrained_frozen \
  --folds 0 \
  --epochs 3 \
  --device cuda
```

For a fast pipeline-only debug run, add `--train-windows-per-subject 1` and
`--eval-windows-per-subject 1`. Do not use the evaluation cap for reported
experiments; the default evaluates every non-overlapping window.

The ready-made equivalent is `sbatch run_biot_ac_smoke.slurm`; its metrics are
pipeline diagnostics only and are written under `outputs/smoke/`.

Supported regimes:

- `pretrained_frozen`: strict-load BIOT and train only the A/C head.
- `pretrained_last2`: fine-tune the last two Transformer blocks and the head.
- `pretrained_full`: fine-tune the entire pretrained encoder and head.
- `scratch`: identical 18-channel-capacity architecture without pretrained weights.

The input has 16 channels in all regimes. For an 18-channel checkpoint, BIOT
uses channel embedding IDs 0-15; the two SHHS embeddings are not padded or used.

## Outputs

The default output directory is `outputs/biot_ac/`. Each regime/fold contains:

- `history.csv`
- `validation_subject_predictions.csv`
- `test_subject_predictions.csv`
- `metrics.json`
- `best_model.pt`

Regime-level `oof_subject_predictions.csv` and `summary.json` contain the final
five-fold subject-level results. Cache and output directories are git-ignored.
The completed formal run is summarized in [`RESULTS.md`](RESULTS.md).

## Tests

```bash
pytest -q test_eeg_ad_biot.py
```
