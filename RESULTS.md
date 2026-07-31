# Experiment log — ADNI1 Screening 1.5T, AD vs CN

Running record of every completed experiment group. Newest group last. All test
numbers come from the same held-out split (271 volumes / 120 AD) scored once;
validation (42 volumes) is used only for model/hyper-parameter selection.

Data: `/net/projects2/litian-lab/scpan/dataset/ADNI_processed/ADNI_processed_clean`,
manifest `data/manifests/adni.csv` (548 volumes, 416 subjects, 235 train / 42 validation / 271 test).

---

## Group 1 — FreeSurfer SynthSeg preprocessing (2026-07-27)

`mri_synthseg --robust --post --vol --qc` over all 1072 scans, FreeSurfer 7.4.1.

- 1072/1072 succeeded, zero errors. Outputs: `synthseg_outputs/{post,seg,vol,qc}/` (11 GB).
- Includes all 525 MCI scans, so MCI-conversion work can reuse this without re-running.
- GPU enabled by installing `nvidia-cudnn-cu11==8.6.0.163` (FreeSurfer ships 8.5, TF 2.12
  needs >=8.6 and otherwise fails with "DNN library is not found"); see `fs_gpu_env.sh`.
- Throughput is bound by CPU preprocessing, not the network (0.06 s/volume on GPU vs 18 s
  on 8 CPU threads). Shard count is the sensitive knob: **4 shards is optimal**
  (1-2 s/step); 16 shards collapses to 316 s/step because 16 CUDA contexts thrash one GPU.
- Whole dataset: ~55 min on one A100 with 4 shards / 32 CPUs.

---

## Group 2 — Frozen encoder + attention-pooling classifier (2026-07-27)

Each encoder frozen, read with **its own native preprocessing** at a depth chosen on the
**validation split only**; the sole trained component is a fresh attention-pooling head.
3 seeds (0/1/2), reported as mean ± sample SD.

| model | layer | volume BA | volume AUC | subject BA | subject AUC |
|---|---:|---:|---:|---:|---:|
| MASS | 3 | 0.687 ± 0.073 | 0.760 ± 0.073 | 0.690 ± 0.075 | 0.765 ± 0.066 |
| MedSigLIP | 9 | 0.680 ± 0.020 | 0.763 ± 0.037 | 0.678 ± 0.019 | 0.761 ± 0.029 |
| AnatCL | 0 | 0.644 ± 0.018 | 0.701 ± 0.014 | 0.653 ± 0.019 | 0.708 ± 0.014 |
| BrainIAC | 1 | 0.596 ± 0.022 | 0.643 ± 0.032 | 0.601 ± 0.027 | 0.647 ± 0.025 |
| *SynthSeg posteriors (baseline)* | n/a | **0.752 ± 0.019** | **0.852 ± 0.019** | 0.762 ± 0.020 | 0.859 ± 0.015 |
| *SynthSeg volumetry (baseline)* | n/a | **0.831** | **0.899** | 0.828 | 0.895 |

**Headline: both FreeSurfer baselines beat all four foundation-model encoders.** Classical
regional volumetry (0.831 BA) leads the best encoder (MASS, 0.687) by ~14 BA points. The four
encoders are not separable from each other at 3 seeds.

Volumetry coefficients follow known AD neuroanatomy, which validates the pipeline:
left hippocampus −0.222, left amygdala −0.210, right hippocampus −0.186,
inferior lateral ventricles +0.141/+0.132.

### Protocol bugs found and fixed before this run

1. **Wrong intensity normalization for every 3D encoder.** All received a generic [0,1]
   percentile rescaling (brain-voxel mean 0.624 / std 0.256) while BrainIAC
   (`dataset.py`: `NormalizeIntensityd(nonzero=True)`) and MASS (`inference.py`: percentile
   clip then per-volume z-score) were trained on z-scored input (mean 0 / std 1) — a 3.9x
   scale error plus a large offset that a frozen encoder cannot absorb. Now each encoder
   uses its own recipe. MedSigLIP was always correct (it uses its own `SiglipImageProcessor`).
2. **Only the deepest layer was ever read.** All four encoders turned out to peak at an
   intermediate depth (validation AUC, deepest in brackets): MedSigLIP 0.858 [0.812],
   MASS 0.870 [0.822], BrainIAC 0.808 [0.579], AnatCL 0.691 [0.590].
3. **Head normalization divided by ~0.** Statistics were taken over each volume's token-mean;
   AnatCL's two permanently dead ReLU channels then had std exactly 0, were scaled to ~1.2e5,
   saturated 100 % of the attention MLP's tanh units and collapsed its attention to uniform
   weights for every subject. Statistics are now taken over individual tokens with a
   relative floor.

Effect on test BA: BrainIAC 0.525 → 0.596, AnatCL 0.597 → 0.644, MASS 0.656 → 0.687,
MedSigLIP 0.752 → 0.680.

**Honest caveat on MedSigLIP.** It got worse. An ablation on the archived deepest-layer cache
(preprocessing identical throughout) splits the loss: pre-fix head 0.752 ± 0.046, fixed head
0.712 ± 0.051, fixed head at the selected layer 9 0.680 ± 0.020. The seed spread is as large
as the gaps, so the three variants are not separable at 3 seeds. The layer is reported as
validation selected it — re-picking it after seeing test numbers would leak the test split.
The real lesson is that layer selection on 42 validation volumes is unreliable: it helped
three encoders and did not help MedSigLIP.

Artifacts: `artifacts/reports/frozen_encoder_comparison.md`,
`artifacts/reports/volumetry_baseline_ad.json`, `artifacts/audits/ad/layer_sweep_*.json`.
Pre-fix caches and probes are kept under `artifacts/{cache,attention}/ad_oldprotocol/`.

---

## Group 3 — Budgeted end-to-end fine-tuning (submitted 2026-07-27)

Same encoders, same layer taps, same native preprocessing, same attention head — but
output-side blocks up to a shared parameter budget are unfrozen and trained end to end
from the original MRI (not the feature cache). 3 seeds, jobs 1254222-1254225.

### Why the earlier fine-tune run was uninformative

Four separate causes, found by reading the code rather than from the numbers:

1. **Training budget far too small.** `max_epochs: 5` with `gradient_accumulation: 8` over
   235 training volumes is 30 steps/epoch, i.e. **150 optimizer steps total** for a 16M
   parameter budget. `patience: 7` could never fire. The README always said 30 epochs.
   → `max_epochs: 30`.
2. **The head-normalization bug was fixed in `training.py` but not in `finetune.py`.**
   `_aligned_cache` still took statistics over each volume's token-mean, so AnatCL's dead
   ReLU channels still had std 0 and blew up to ~1e5. → both paths now call
   `token_normalization`.
3. **Fatal once layer selection existed: the unfrozen blocks were downstream of the tap.**
   `finetune_groups()` returned the *deepest* blocks, but features are now read at layer
   9/3/1/0. Those blocks contribute nothing to the loss, so their `.grad` stays `None` and
   the built-in gradient audit raises. → `_stage_groups()` only offers modules upstream of
   the tap. Verified: 0 missing gradients for all four encoders.
4. **Layers past the tap were still being executed.** `_siglip_layer_outputs` ran all 27
   SigLIP layers regardless of depth. → `stop_after` ends the walk at the tap.

### Parameter budget: 16M → 8M

235 training volumes cannot support 16M trainable parameters (~68k per sample). Complete
blocks are never split, so each encoder still unfreezes at least one whole block; the budget
only decides whether a *second* one is added. Measured trainable counts at the selected taps:

| encoder | tap | unfrozen modules (at 16M) | trainable |
|---|---:|---|---:|
| MedSigLIP | 9 | `layers.9` | 15,239,504 |
| BrainIAC | 1 | `blocks.1`, `blocks.0` | 14,171,136 |
| MASS | 3 | `down3`,`down2`,`down1`,`inc` | 9,345,888 |
| AnatCL | 0 | `layer1` | 442,880 |

At 8M, BrainIAC drops to one block and MASS to fewer stages. MedSigLIP is pinned at 15.2M
regardless because a single SigLIP block already exceeds any smaller budget, and AnatCL is
capped at 442k by its own architecture — reading at layer 0 leaves only `layer1` upstream.
So the budget cannot equalise capacity across encoders once layer selection is in play; the
actual count is recorded per run and must be reported alongside the metric.

### Expectation stated before the results were known

Frozen probing already reaches 0.687 (MASS). Fine-tuning 0.4M–15M parameters on 235 volumes
carries a high overfitting risk, so a gain is not the expected outcome. All three outcomes —
improvement, parity, or degradation — will be reported as measured. Parity or degradation is
itself a useful finding (frozen features are saturated at this sample size); it will not be
tuned away against the test split.

### Results (12/12 runs, zero errors)

| model | trainable | volume BA | volume AUC | subject BA | subject AUC |
|---|---:|---:|---:|---:|---:|
| **MASS (fine-tuned)** | 7,077,888 | **0.774 ± 0.019** | **0.852 ± 0.011** | 0.771 ± 0.020 | 0.850 ± 0.007 |
| **MedSigLIP (fine-tuned)** | 15,239,504 | 0.712 ± 0.066 | 0.807 ± 0.080 | 0.716 ± 0.055 | 0.805 ± 0.085 |
| **BrainIAC (fine-tuned)** | 7,085,568 | 0.624 ± 0.023 | 0.673 ± 0.027 | 0.624 ± 0.023 | 0.664 ± 0.018 |
| **AnatCL (fine-tuned)** | 442,880 | 0.556 ± 0.098 | 0.613 ± 0.157 | 0.557 ± 0.099 | 0.613 ± 0.176 |
| *SynthSeg volumetry (baseline)* | 0 | **0.831** | **0.899** | 0.828 | 0.895 |
| *SynthSeg posteriors (baseline)* | 0 | 0.752 ± 0.019 | 0.852 ± 0.019 | 0.762 ± 0.020 | 0.859 ± 0.015 |

Change vs the frozen probe on the identical tap and preprocessing:

| encoder | frozen BA | fine-tuned BA | delta | best epoch (of 30) |
|---|---:|---:|---:|---:|
| MASS | 0.687 ± 0.073 | 0.774 ± 0.019 | **+0.087** | 11 |
| MedSigLIP | 0.680 ± 0.020 | 0.712 ± 0.066 | +0.032 | 10 |
| BrainIAC | 0.596 ± 0.022 | 0.624 ± 0.023 | +0.028 | 9 |
| AnatCL | 0.644 ± 0.018 | 0.556 ± 0.098 | **−0.088** | 7 |

**The fixes worked.** Fine-tuning now helps three of four encoders, where previously it was
indistinguishable from (or worse than) frozen probing. Early stopping fires at epoch 7–11 of
30, confirming that the old `max_epochs: 5` really was cutting training off mid-way.

**MASS is the standout**: +0.087 BA, and its seed spread *tightens* (±0.073 → ±0.019), which
is the signature of genuine learning rather than noise. At 0.774 BA / 0.852 AUC it now beats
the SynthSeg posterior baseline (0.752) and matches its AUC exactly.

**AnatCL degrades, and this is interpretable rather than mysterious.** It has only 442k
trainable parameters — its tap is at layer 0, so `layer1` is the only module upstream — and
that module is the earliest conv stage. Training the very first stage on 235 volumes destabilises
everything downstream, which shows up as the largest seed spread in the whole study (±0.098 BA,
±0.157 AUC). It is also the encoder with a known domain mismatch (trained on CAT12/VBM
grey-matter maps, fed skull-stripped T1).

### Headline after all three groups

Classical regional volumetry (0.831 BA / 0.899 AUC) **still leads every encoder**, including
the best fine-tuned one (MASS, 0.774 / 0.852). The gap narrowed from ~14 BA points to ~6, but
it did not close. The honest summary for the paper: on ADNI AD/CN, none of these brain-MRI
foundation models — frozen or fine-tuned within a 8M-parameter budget — outperforms measuring
the hippocampus.

Artifacts: `artifacts/finetune/ad/*/finetune_seed_*_summary.json` (per-seed metrics,
subject-clustered bootstrap CIs, selected parameter groups, gradient audits).
Jobs 1254222-1254225, submitted with `sbatch` (`freesurfer_install/sbatch_finetune.sh`).

---

## Notes added on review (2026-07-28), before Group 4

**Is every encoder using its own native preprocessing?** Audited by diffing our
normalization code against each repo's actual training-time transform, not just its
documented protocol. Two more real bugs turned up, both now fixed and byte-exact
verified against the source code they're supposed to match:

1. **BrainIAC background was not pinned to 0.** `NormalizeIntensityd(nonzero=True)`
   (BrainIAC's own `dataset.py`) only rewrites non-zero voxels; the rest of the array
   -- background, already 0 on this skull-stripped data -- passes through unchanged.
   Our `zscore_volume` instead applied `(volume - mean) / std` to the *whole* array,
   pushing ~52% of every volume (the background) to a uniform -3.05 SD block BrainIAC
   never saw in training. Fixed as `zscore_nonzero_volume`
   (`src/encoderbench/preprocessing.py`).
2. **Resize/normalize order was backwards for every 3D encoder.** BrainIAC's Compose
   is `Resized` then `NormalizeIntensityd`; MASS's `preprocess_target_image` resamples
   and crops *before* calling `normalize_image`. `volume_to_tensor` did it the other
   way round (normalize, then resize). Trilinear-interpolating already-normalized
   voxels changes which fractional values land at the brain/background boundary,
   which changes the foreground mask nonzero-style stats are computed over -- 39% of
   one measured volume's foreground voxels differed by >0.05 z-score units (max 2.67)
   between the two orderings. Fixed by reordering `volume_to_tensor`.

**BrainIAC also has a real preprocessing standard beyond `dataset.py`**, found in
`BrainIAC/src/preprocessing/mri_preprocess_3d_simple.py`: N4 bias correction, rigid
registration to a bundled atlas (`atlases/temp_head.nii.gz`), then HD-BET skull
stripping. None of this runs in our pipeline -- we feed ADNI's already skull-stripped,
box-cropped files (a different strip tool, no registration, no N4). Per direct
instruction this is being left as a disclosed gap, not implemented, alongside MASS's
already-disclosed spacing gap (MASS resamples to 1.5mm isotropic before its own body
crop; we have no un-cropped original to resample from).

**AnatCL is not being re-run or dropped**, per instruction, but its preprocessing gap
is worth spelling out precisely since it's categorically different from BrainIAC's/
MASS's. AnatCL expects CAT12/VBM gray-matter *density* maps: SPM12-segmented tissue
probability, spatially **normalized to MNI template space** (DARTEL/geodesic
shooting), then modulated by the warp's Jacobian and usually smoothed. We substitute
skull-stripped T1 **intensity** images in each subject's own **native scanner space**.
The shape (121x128x121) coincidentally matches CAT12's standard MNI grid at 1.5mm --
that's a coordinate-system size, not a content match. The two differ in both what a
voxel value *means* (tissue-class probability vs. T1 signal) and what a voxel
*coordinate* means (same anatomical location across all subjects vs. nothing in
common across subjects). This is a wrong-modality gap, not a wrong-scale one, and
plausibly explains why AnatCL is both the weakest encoder and the one that collapses
worst under fine-tuning (+/-0.098 BA across seeds, the largest spread in the study).
Fixing it for real needs SPM12+CAT12 (MATLAB-dependent segmentation, DARTEL
registration, modulation) -- out of scope here.

**SynthSeg posteriors vs. SynthSeg volumetry are two different uses of the same
segmentation run, not two names for the same thing.** *Posteriors* (`SynthSegExtractor`)
reads the 33-class per-voxel posterior-probability volume from `mri_synthseg --post`
and pools it through the identical attention-pooling pipeline (4x4x4 grid, same
position encoding) used for every learned encoder -- it's compared as if it were a
frozen backbone. *Volumetry* (`scripts/volumetry_baseline.py`) instead reads the scalar
per-structure volumes from `mri_synthseg --vol` (e.g. "left hippocampus in mm^3"),
divides by intracranial volume, and fits plain logistic regression -- no pooling, no
position encoding, no learned representation at all. It's the classical
"how much better than measuring the hippocampus" reference line the code comment
names explicitly. That volumetry (0.831 BA) beats posteriors (0.752 BA) is itself a
finding: discrete anatomical volumes carried more usable signal here than the dense
probability maps did.

---

## Group 4 — BrainIAC & MASS re-run after the two bugs above, with overfitting-aware selection (2026-07-28)

Scope, per instruction: only BrainIAC and MASS re-run (their preprocessing changed);
MedSigLIP/AnatCL/SynthSeg untouched, Group 2/3 numbers above are unmodified. Layer
selection was re-swept on the corrected preprocessing (validation split only).
finetune epoch budget cut 30 -> 12. Jobs 1259835 (brainiac, failed at the finetune
step on a stale pre-fix checkpoint -- probe had already completed), 1265190
(brainiac finetune retry with `--restart`), 1265191 (mass, full chain).

### Methodology change: overfitting is now detected from data, not assumed

Added `encoderbench.training.detect_overfitting(train_loss, val_loss)`: the standard
early-stopping criterion:

> validation loss stays at or above its running minimum for `patience` consecutive
> epochs *while training loss keeps falling* -> overfitting, use the running-minimum
> epoch. If validation and training plateau together (or validation keeps improving),
> that is convergence, not overfitting -> use the final epoch reached.

This replaces two things that were silently optimistic before:

- **`probe`** previously trained 300 epochs x 5 weight-decays (1500 candidates) and
  kept whichever single one had the best validation balanced accuracy -- on 42
  validation volumes, picking 1 winner out of 1500 noisy evaluations risks
  capitalizing on validation noise rather than finding a real optimum. It now applies
  `detect_overfitting` per weight-decay curve and uses the running-minimum epoch when
  overfitting is detected, the final epoch otherwise (`patience=15`).
- **`finetune`** already had patience-based early stopping (`patience=7`), which is a
  reasonable mitigation on its own, but always reported the best-so-far checkpoint
  even when the validation curve had simply plateaued (not diverged) -- i.e. even
  the "did not overfit" case was quietly using a cherry-picked epoch. It now reports
  the actual final epoch reached in that case (`patience=7`, matching the training
  loop's own patience so the diagnostic window is achievable within a 12-epoch cap).

Every probe/finetune summary JSON now carries an `"overfitting"` field with the full
diagnosis (`is_overfitting`, `selected_epoch`, `onset_epoch`, the train/val loss
values at divergence, and the plain-language criterion that fired).

### BrainIAC: layer re-selected 1 -> 10; frozen probe roughly flat, fine-tune slightly worse

Validation-only sweep on the fixed preprocessing (val AUC): layer 10 = 0.714, tied
with deepest/-1 (0.714); the whole curve is much flatter than before (0.44-0.71)
-- unlike the earlier, buggy-preprocessing sweep, depth barely matters now.

| run | seed | selected epoch | overfitting? | volume BA | volume AUC | subject BA | subject AUC |
|---|---:|---:|---|---:|---:|---:|---:|
| probe | 0 | 3 | yes (onset ep. 3) | 0.6024 | 0.6418 | 0.6264 | 0.6582 |
| probe | 1 | 3 | yes | 0.6147 | 0.6578 | 0.6218 | 0.6608 |
| probe | 2 | 2 | yes | 0.6075 | 0.6522 | 0.6108 | 0.6595 |
| **probe mean** | | | | **0.6082 ± 0.0050** | 0.6506 ± 0.0066 | 0.6197 ± 0.0066 | 0.6595 ± 0.0011 |
| finetune | 0 | 8/8 | no (final epoch) | 0.5897 | 0.6420 | 0.5986 | 0.6429 |
| finetune | 1 | 8/8 | no | 0.5872 | 0.6119 | 0.6014 | 0.6208 |
| finetune | 2 | 10/10 | no | 0.6199 | 0.6511 | 0.6521 | 0.6644 |
| **finetune mean** | | | | **0.5989 ± 0.0149** | 0.6350 ± 0.0167 | 0.6174 ± 0.0246 | 0.6427 ± 0.0178 |

All three probe runs hit genuine early overfitting by epoch 2-3: validation loss
bottoms out almost immediately and never recovers for 15 straight epochs while
training loss keeps falling toward ~0.08-0.09. All three finetune runs stopped via
patience (8-10 of a 12-epoch budget) but were *not* flagged as overfitting -- training
and validation loss plateaued together rather than diverging, so the final epoch's
own metrics are reported rather than an earlier lucky peak.

### MASS: layer unchanged at 3; both probe and fine-tune slightly lower than before

Validation-only sweep on the fixed preprocessing: layer 3 = 0.899 AUC (vs. deepest
-1 = 0.817) -- same winning layer as the pre-fix sweep, and a higher validation AUC
than the pre-fix number (0.870), for whatever that predicts about downstream test
performance (see caveat below).

| run | seed | selected epoch | overfitting? | volume BA | volume AUC | subject BA | subject AUC |
|---|---:|---:|---|---:|---:|---:|---:|
| probe | 0 | 30 | yes | 0.6498 | 0.7286 | 0.6385 | 0.7270 |
| probe | 1 | 112 | yes | 0.7342 | 0.8134 | 0.7336 | 0.8122 |
| probe | 2 | 10 | yes | 0.5280 | 0.6541 | 0.5468 | 0.6661 |
| **probe mean** | | | | **0.6373 ± 0.0846** | 0.7320 ± 0.0650 | 0.6396 ± 0.0762 | 0.7351 ± 0.0599 |
| finetune | 0 | 12/12 | no (final epoch) | 0.7440 | 0.8280 | 0.7511 | 0.8329 |
| finetune | 1 | 11/11 | no | 0.7948 | 0.8704 | 0.7909 | 0.8685 |
| finetune | 2 | 12/12 | no | 0.7325 | 0.8236 | 0.7292 | 0.8199 |
| **finetune mean** | | | | **0.7571 ± 0.0271** | 0.8407 ± 0.0211 | 0.7571 ± 0.0255 | 0.8404 ± 0.0205 |

### Honest before/after comparison -- this is not a clean win

| encoder | run | old (pre-fix preprocessing + naive best-of-N selection) | new (fixed preprocessing + overfitting-aware selection) | delta |
|---|---|---:|---:|---:|
| BrainIAC | probe | 0.596 ± 0.022 | 0.608 ± 0.005 | +0.012 |
| BrainIAC | finetune | 0.624 ± 0.023 | 0.599 ± 0.015 | **-0.025** |
| MASS | probe | 0.687 ± 0.073 | 0.637 ± 0.085 | **-0.050** |
| MASS | finetune | 0.774 ± 0.019 | 0.757 ± 0.027 | -0.017 |

Three of four numbers went down, not up, despite the preprocessing fixes being real
and verified. **This is not evidence the fixes were wrong** -- it's evidence the old
numbers were partly inflated by a selection procedure that quietly cherry-picked
optimistic validation epochs (up to 1 winner out of 1500 candidates for probe;
always the best-so-far checkpoint for finetune, even when the curve had only
plateaured rather than genuinely overfit). Two changes happened simultaneously here
-- corrected preprocessing and a less-optimistic selection rule -- and this run
cannot cleanly separate how much of the drop is which, since both were requested
together. If a clean attribution matters later, the isolating experiment is: fixed
preprocessing + the *old* naive best-of-N selection, to see how much of today's drop
survives on its own.

Updated headline: classical volumetry (0.831 BA / 0.899 AUC) leads **by a wider
margin** under this more conservative selection than it did in Group 3 -- the best
fine-tuned encoder is now MASS at 0.757 BA (was 0.774), an 0.074 BA gap versus
volumetry's 0.057 BA gap before. The qualitative conclusion is unchanged and, if
anything, reinforced: no encoder here, frozen or fine-tuned, beats measuring the
hippocampus.

Artifacts: `artifacts/attention/ad/{brainiac,mass}/probe_seed_*_summary.json`,
`artifacts/finetune/ad/{brainiac,mass}/finetune_seed_*_summary.json` (each now
carries an `"overfitting"` diagnosis field), `artifacts/audits/ad/layer_sweep_{brainiac,mass}.json`.
Jobs 1259835 (partial), 1265190, 1265191.
