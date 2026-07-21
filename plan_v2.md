# Encoders Test Experiment Plan v2

## 1. 项目目标

本项目回答四个彼此独立的问题：

1. 在 encoder 全冻结时，哪一个 encoder 的表征最适合 AD/CN 分类？
2. MedGemma 和 BrainGemma3D 的完整 VLM 是否具备 zero-shot 分类能力？
3. 当 encoder 和 MedGemma LLM 都冻结、只训练一个受控 bridge 时，哪种 encoder 最容易被 LLM 利用？
4. 上述结论能否在 UCLA CNP 的 SCZ/CN 任务上复现？

这些结果必须分开报告，不能把 supervised attention head、zero-shot LLM 和 supervised bridge tuning 混合成一个总排名。

工作目录确定为：

```text
/net/projects2/litian-lab/scpan/encoders
```

所有权重统一放在：

```text
/net/projects2/litian-lab/scpan/model_weights
```

MASS 权重已确认存在：

```text
model_weights/MASS/mass_base.pth
```

---

## 2. 已冻结的实验决定

- MedGemma encoder probe 使用独立 MedSigLIP-448：

  ```text
  /net/projects2/litian-lab/scpan/model_weights/medsiglip
  ```

- 四个 encoder 全部冻结。
- attention probe 只训练完全相同的 attention-pooling head。
- BrainGemma3D encoder probe 使用 projector 前的 3D vision features。
- 不进行任何 LoRA 或 encoder/LLM fine-tuning。
- ADNI 严格复用 `brain_fm`：

  - 50% test；
  - 剩余 train 中 15% validation；
  - subject-level stratification；
  - split seed 0。

- UCLA CNP 严格复用 `brain_fm`：

  - 50% test；
  - 剩余 train 中 15% validation；
  - subject-level stratification；
  - split seed 0。

- Zero-shot LLM 实验也迁移到 SCZ。
- Bridge 权重不共享，只共享设计和训练协议。
- Attention head 和 bridge 的正式结果均运行 3 个训练随机种子：`0,1,2`。
- 数据 split 始终固定为 seed 0，不随 attention-head/bridge 训练 seed 改变。

需要明确：新增 bridge 实验属于 supervised bridge tuning，不是 zero-shot；但 encoder 和 MedGemma LLM 始终冻结。

---

## 3. 完整实验矩阵

### ADNI：AD vs CN

| 实验组 | 模型数量 | 训练部分 |
|---|---:|---|
| Frozen encoder + attention head | 4 × 3 seeds | attention head |
| Native VLM zero-shot | 2 | 无 |
| Frozen encoder + Linear bridge + MedGemma | 4 × 3 seeds | Linear bridge |
| Frozen encoder + Resampler bridge + MedGemma | 4 × 3 seeds | Resampler bridge |

### UCLA CNP：SCZ vs CN

完整复用上述矩阵，但：

- 重新训练 SCZ-specific attention head；
- 重新训练 SCZ-specific bridge；
- 使用 SCZ-specific prompt；
- 不从 AD bridge 继续训练；
- 不改变任何 architecture 或优化设置。

这属于 workflow transfer，而不是 AD classifier 到 SCZ 的直接 cross-disease transfer。

最终主实验训练运行数量为：

```text
attention head: 4 encoders × 3 seeds × 2 diseases = 24 runs
bridge: 4 encoders × 2 bridge capacities × 3 seeds × 2 diseases = 48 runs
shuffled-label attention control: 额外 4 encoders × 3 seeds × 2 diseases = 24 runs
```

因此采用 phase gate，不一次启动全部任务。

---

# 4. Phase 0：权重、数据和接口审计

正式实验前，每个模型只测试 1 个 AD 和 1 个 CN volume。

必须检查：

- checkpoint 加载路径和 SHA/checkpoint identifier；
- missing/unexpected keys；
- MASS 是否优先使用 `ema_model`；
- encoder trainable parameter 数量必须为 0；
- MedGemma backbone trainable parameter 数量必须为 0；
- encoder 处于 `eval()`；
- 输出 shape、dtype、NaN/Inf；
- tokenizer 中答案 `A`、`B` 的确切 tokenization；
- prompt token、visual token 和完整 answer token sequence 的 label mask；
- bridge 梯度非零，但 encoder/LLM gradient 为零。

四个 encoder 都必须把 patch-grid 审计作为 Phase 0 标准 smoke test，而不是只检查 BrainGemma3D：

- 记录 native patch/feature grid shape 和 flatten 后的 `[N_tokens, D_encoder]`；
- 验证 `N_tokens` 等于各空间轴乘积，grid reshape/flatten 可无损往返；
- 记录统一 pooling 前后 shape，并确认 pooling 后均为 `4×4×4 = 64` token；
- 检查轴顺序、token 顺序和空间位置一一对应；
- 明确排除 CLS、global pooled token、padding token 和 projector 输出；
- 用 1 个 AD 和 1 个 CN 样本确认 shape 稳定且无 NaN/Inf。

BrainGemma3D 需额外确认：

- 3D inflation 后的 grid 与上述通用 patch-grid 审计一致；
- attention probe 使用 projector 前 token；
- native zero-shot 使用官方 projector；
- bridge 实验不使用 BrainGemma3D 官方 projector，而使用统一的新 bridge。

---

# 5. Phase 1：数据与 manifest

## 5.1 ADNI

重新生成当前目录可用的 manifest，不修改原始 `brain_fm` manifest。

保持现有 cohort：

- 548 volumes；
- 416 subjects；
- AD 188 subjects；
- CN 228 subjects；
- repeat scan 不跨 split。

ADNI 输入必须直接使用与 `brain_fm` 相同的已预处理 volume：AFNI `@SSwarper` 产出的 skull-stripped、boxed、MNI-registered T1，float32、约 1 mm spacing、shape 可变。本项目不对 ADNI 再做 skull stripping、registration、N4、HD-BET 或第二套强度校正。

训练仍按 volume 进行，以严格复现 `brain_fm`。评估同时提供：

- volume-level point estimate：与 `brain_fm` 直接可比；
- subject-level estimate：同一 subject 多次 scan 的概率先平均；
- 95% CI 使用 subject-cluster bootstrap，避免把 repeat scan 当作独立受试者。

## 5.2 UCLA CNP

使用已有 175 subjects：

- CN 125；
- SCZ 50；
- 每人一个 T1w。

严格复用 [build_manifest_scz.py](/net/projects2/litian-lab/scpan/brain_fm/src/build_manifest_scz.py) 和 [ft_dataset.py](/net/projects2/litian-lab/scpan/brain_fm/src/ft_dataset.py) 的两级切分：

1. 仅保留 `participants.tsv` 中 `diagnosis=SCHZ` 或 `CONTROL` 且 T1w 文件存在、非空的受试者，映射为 `SCZ`/`CN`。
2. manifest train/test 切分时只实例化一次 `random.Random(0)`；每个类别内先按 `subject_id` 排序，再按数据中类别的稳定出现顺序用该同一个 RNG 依次 shuffle，取 Python 内置 `round(n_class × 0.5)` 个受试者进入 test；其余进入 manifest train。不得为每个类别重新初始化 RNG。
3. validation 只从 manifest train 中产生：该级切分重新实例化一次 `random.Random(0)`，再以同样的排序、类别顺序和共享 RNG 规则依次 shuffle，取 `max(1, round(n_class × 0.15))` 个受试者进入 validation。
4. 三个 split 对所有 encoder、attention head 和 bridge 完全固定；训练 seed `0,1,2` 不重新切数据。

在 175 个受试者全部可用时，预期计数必须为：

| split | CN | SCZ | total |
|---|---:|---:|---:|
| train（扣除 validation 后） | 54 | 21 | 75 |
| validation | 9 | 4 | 13 |
| test | 62 | 25 | 87 |
| total | 125 | 50 | 175 |

若计数不同，必须先记录缺失/损坏文件并重新计算实际计数；不得通过移动受试者来手工凑数。

UCLA 必须保持与 `brain_fm` 相同的 raw BIDS T1w 输入：

```text
sub-XXXXX/anat/sub-XXXXX_T1w.nii.gz
```

不增加 skull stripping、registration、N4、HD-BET 或其他 dataset-specific preprocessing。

## 5.3 两个数据集共享的输入处理

两者共同复用 [volume_to_slices.py](/net/projects2/litian-lab/scpan/brain_fm/src/volume_to_slices.py) 的数据约定：

- canonical RAS；
- 按整个 volume 的正值非零体素计算 1st/99th percentile；若没有正值体素才退回全部体素；
- clip 到上述 bounds 并线性缩放到 `[0,1]`；
- slice-based 2D 路径再乘 255、round 为 `uint8`、复制为三通道 RGB，并由各 encoder 的原生 processor 做最终 resize/normalize；
- slice-based 路径锁定 `brain_fm` Stage 1/3 的正式配置：`axis=2`（axial，inferior-to-superior）、`num_slices=24`、`slice_range=(0.15,0.85)`、`roi_frac=1.0`，在范围内包含端点地均匀采样；同一实验配置不得因数据集或 encoder 单独改变；
- 3D encoder 必需的 tensor resize/resampling 只算 encoder-native adapter；其规则须在 Phase 0 固定并对 ADNI/UCLA 完全一致，不得引入额外 skull stripping、registration 或强度校正。

因此，“与 `brain_fm` 一样”指 cohort 原始/预处理状态和上述共同输入处理均一致：ADNI 沿用已有预处理，UCLA 沿用 raw T1w；不能把 ADNI 的 `@SSwarper` 结果错误地复刻到 UCLA。

## 5.4 数据检查

每个数据集必须记录：

- label 和 split 计数；
- subject overlap；
- 缺失/损坏 NIfTI；
- orientation；
- 原始 shape 和 spacing；
- normalization 后统计；
- 代表性 montage。

ADNI participant-level 图像、路径和预测不得进入公开 report。

---

# 6. Phase 2：Frozen encoder + common attention head

## 6.1 Encoder token 定义

| Encoder | attention head 输入 |
|---|---|
| MedSigLIP | 与 §9.2 共用同一 token extractor：24 个 axial slice 各取最后一个 vision block 的 2D patch tokens，组成 pseudo-3D grid，再 adaptive average pool 到 `4×4×4`；输入为 64 个 patch-level spatial tokens，不使用 global pooled embedding |
| BrainGemma3D | projector 前的最后一层 3D patch tokens |
| MASS | encoder 最深层 feature map 展平为空间 token；不使用 decoder |
| BrainIAC | 最后一个 ViT block 的 patch tokens，去掉 CLS |

所有输出统一表示为：

```text
[B, N_tokens, D_encoder]
```

token 数和维度允许不同，但 attention head 的结构与训练方法完全一致。

## 6.2 Common attention head

沿用 [probe.py](/net/projects2/litian-lab/scpan/brain_fm/src/probe.py) 的设计：

```text
LayerNorm/standardization
D_encoder → 128 → tanh → 1
softmax over tokens
weighted token sum
linear classifier → 2 classes
```

共同训练协议：

- encoder features 预先缓存；
- train-only normalization statistics；
- class-weighted cross entropy；
- AdamW；
- `lr=1e-3`；
- 原有 weight-decay grid；
- validation balanced accuracy 选择 checkpoint；
- 训练 seeds `0,1,2`；
- 数据 split 固定为 seed 0；
- shuffled-label negative control 也运行相同的 3 个训练 seeds。

每个 seed 控制 head initialization、batch shuffle 和 sampler order。四个 encoder 在同一 seed 下使用相同 sample order；不得挑选最佳 seed 作为主结果。

这一阶段严格回答“冻结表征 + 小型分类头”的能力。

---

# 7. Phase 3：AD attention-head evaluation

主指标：

- balanced accuracy；
- ROC-AUC；
- sensitivity；
- specificity；
- F1；
- 95% subject-cluster bootstrap CI。

比较 encoder 时使用同一批 test subjects，并计算 paired bootstrap difference，例如：

```text
Δ balanced accuracy = BrainGemma3D − MedSigLIP
```

不以重叠/不重叠的单模型 CI 代替 paired comparison。

---

# 8. Phase 4：Native VLM zero-shot

## 8.1 模型

1. 完整 MedGemma：

   ```text
   model_weights/medgemma
   ```

2. 完整 BrainGemma3D：

   ```text
   model_weights/braingemma3D
   ```

   使用其 3D vision encoder、官方 projector 和 MedGemma language model。

两者全部冻结，无训练、无 LoRA。

## 8.2 AD prompt

保持 `brain_fm` radiologist-style T1 AD/CN prompt、相同 slice 数、相同解析方法和 free generation。

## 8.3 SCZ prompt

使用相同 prompt 结构，但内容改为研究性 SCZ/CN cohort classification。报告中明确：

- 结构 MRI 不能作为临床精神分裂症确诊工具；
- 该实验测试的是 cohort-level signal，而不是临床诊断能力。

## 8.4 评估

- temperature 0；
- free generation，不以 teacher-forced class score 替代；
- 保存完整 raw response，并严格复用 `brain_fm` 的分层 parser：

  1. 优先匹配规范格式 `Final Answer: <LABEL>`（允许冒号/连字符和大小写差异），命中时 `matched=true`；
  2. 未命中时，取回答中最后一个独立出现的合法缩写；AD 任务为 `AD|CN`，SCZ 任务为 `SCZ|CN`，此时保留预测但 `matched=false`；
  3. 仍未命中时，使用与 `brain_fm` 相同的有限语义 fallback：只有回答含 `Alzheimer` 且不含 `normal` 时才判 AD；含 `normal` 或 `no evidence` 时判 CN。SCZ 对称处理：只有含 `schizophrenia`/`schizophrenic` 且不含 `normal` 时才判 SCZ；含 `normal` 或 `no evidence` 时判 CN。禁止增加依赖单个实验输出的临时规则；
  4. 空回答或仍无法解析记为 `UNK`，推理异常记为 `ERR`。

- fallback 得到合法类别的回答照常参与指标，但 `matched=false`，并单独计入 valid-fallback rate；
- `UNK/ERR` 按 `brain_fm/src/score.py` 的规则强制计错：对每一行映射为该行真实类别的相反类，不能丢弃；
- 同时报告 exact-format rate、valid-fallback rate、总 nonexact-format rate 和 `UNK/ERR` rate，并保留 raw output 供审计；
- 不使用 teacher-forced loss；
- 不把 zero-shot 结果与 supervised head 排入同一个表。

---

# 9. Phase 4.5：Frozen encoder → Bridge → Frozen MedGemma

这是新增的核心实验。

## 9.1 公平性原则

所有 encoder 都使用：

- 同一个 canonical MedGemma backbone：

  ```text
  model_weights/medgemma
  ```

- 同样数量的 visual tokens；
- 相同 bridge hidden width；
- 相同层数；
- 相同初始化方法；
- 相同 prompt；
- 相同 answer-only loss；
- 相同 batch order；
- 相同 optimizer、learning rate、weight decay；
- 相同 early stopping；
- 相同 seeds。

Bridge 权重不共享。共享权重会迫使不同 feature space 使用同一个坐标系统，反而可能不公平。

## 9.2 统一空间 token grid

我把你写的 MASS `4×4×4` 理解为 64 个空间 token。四个 encoder 都转换成：

```text
[B, 64, D_encoder]
```

具体方式：

### MedSigLIP

不使用最终 global pooled embedding。

- 取每张 slice 最后一个 vision block 的 2D patch tokens；
- 按 24 个 axial slices 组成 pseudo-3D feature grid；
- adaptive average pool 到 `4×4×4`。

### BrainGemma3D

- 取 projector 前的最后一层 3D patch grid；
- 恢复其 `(D,H,W)` 空间结构；
- adaptive average pool 到 `4×4×4`。

### MASS

- 只取 encoder 最深层 3D feature map；
- 不使用 segmentation decoder、prior encoder 或 fusion decoder；
- adaptive average pool 到 `4×4×4`。

### BrainIAC

- 取最后一个 ViT block token；
- 去掉 CLS；
- 216 个 patch tokens 恢复为 `6×6×6`；
- adaptive average pool 到 `4×4×4`。

pooling 后加入相同的固定 3D positional encoding。位置编码不训练，避免不同 encoder 获得不同额外容量。

---

## 9.3 Linear bridge

单一 `D_encoder → 2560` Linear 的参数量差异过大，因此采用 factorized linear mapping：

```text
Input LayerNorm
Linear(D_encoder → 512, no bias)
Linear(512 → 2560)
Final LayerNorm
```

两个 Linear 中间没有 activation，因此整体仍然是一个 rank-512 linear projection，而不是 nonlinear MLP。

大致参数量：

| Encoder | D | Linear bridge 参数量 |
|---|---:|---:|
| MedSigLIP | 1152 | ≈1.91M |
| BrainGemma3D | 1152 | ≈1.91M |
| BrainIAC | 768 | ≈1.71M |
| MASS | 512 | ≈1.58M |

差异来源仅是 encoder native dimension。强行 zero-padding 到统一维度虽然能让参数量完全相同，但会引入更不自然的结构偏置，因此不建议。

---

## 9.4 Resampler bridge

使用相同的 64 个 learnable queries：

```text
source projection: D_encoder → 512
64 learned queries, width 512

2 × {
    query self-attention
    cross-attention(query, encoder tokens)
    FFN: 512 → 2048 → 512
}

output projection: 512 → 2560
final normalization
```

固定：

- 8 attention heads；
- 2 layers；
- query count 64；
- hidden size 512；
- FFN size 2048；
- dropout 0；
- 相同初始化。

Resampler 的主体参数量相同，只有输入 projection 随 `D_encoder` 略有差异，因此四个模型的总参数量会比 Linear bridge 更接近。

Linear 和 Resampler 应分别形成独立比较表。不能因为 Resampler 绝对性能更高，就把它和 Linear 结果混成一个 encoder 排名。

---

## 9.5 Visual token 注入 MedGemma

统一使用：

```text
[BOS] + [64 projected visual tokens] + [prompt tokens] + [answer token sequence]
```

- attention mask 覆盖全部有效 token；
- position IDs 连续；
- visual tokens 和 prompt tokens 的 labels 全部设为 `-100`；
- 只有完整答案 token sequence 参与 loss；
- 不使用 MedGemma 自带 vision tower；
- 不使用 BrainGemma3D 官方 projector；
- MedGemma 参数 `requires_grad=False`，但不能用 `torch.no_grad()` 包裹 LLM forward，因为梯度需要传回 bridge。

---

## 9.6 Prompt 与 loss

AD prompt 固定为：

```text
Classify this brain MRI as:
A. Alzheimer's disease
B. Cognitively normal
Output only A or B.
```

Target：

```text
AD → A
CN → B
```

SCZ prompt 固定为：

```text
Classify this brain MRI as:
A. Schizophrenia
B. Cognitively normal
Output only A or B.
```

训练前必须检查 tokenizer 对 prompt 后的 `A` 和 `B` 是否各为一个 token。若不是单 token，则使用完整答案 token sequence 的 summed log-likelihood，不能只比较第一个 subtoken。

tokenizer 审计必须在“实际 chat template + assistant generation prefix”的上下文中完成，不能只单独调用 `tokenizer("A")`/`tokenizer("B")`。记录两个候选的完整 token IDs、长度、是否包含前导空格以及模板是否自动加入 EOS。训练与测试必须使用同一套候选序列；若 EOS 被定义为 target 的一部分，则两边都包含，否则两边都不包含。

Loss：

```text
answer-only class-weighted next-token cross-entropy
```

对类别 `c` 的答案 token sequence `y_c=(y_c,1,...,y_c,Lc)`，单样本 loss 定义为：

```text
L_c = -w_c × Σ_t log P(y_c,t | visual tokens, prompt, y_c,<t)
```

也就是先对完整答案 sequence 求和，再按样本做 class weighting 和 batch mean；不按 token 数取平均，避免训练目标与测试时的 summed log-likelihood 不一致。class weights 只根据 training split 计算。

---

## 9.7 A/B 概率测试

测试不做 free generation。

设实际模板上下文中的完整候选序列为：

```text
y_A = (a_1, ..., a_LA)
y_B = (b_1, ..., b_LB)
```

对两个候选分别做 teacher-forced scoring（可合并成一个两候选 batch），取得完整条件序列分数：

```text
S_A = Σ_t log P(a_t | visual tokens, prompt, a_<t)
S_B = Σ_t log P(b_t | visual tokens, prompt, b_<t)
```

这里不做长度归一化。若 tokenizer audit 证明两个完整 target 都各只有一个实际计分 token（包括确认 EOS 不属于 target），上式自然退化为原来的单位置 `sA`/`sB` 快速路径；只比较第一个 subtoken 在任何情况下都不允许。

归一化为：

```text
P(A) = exp(S_A) / [exp(S_A) + exp(S_B)]
P(B) = exp(S_B) / [exp(S_A) + exp(S_B)]
```

实现时使用 `logsumexp(S_A,S_B)` 保持数值稳定。这里的概率是仅在两个预注册候选 `{A,B}` 内重新归一化的 conditional class probability，不是整个词表上的原始 token probability。

预测规则：

```text
P(A) ≥ 0.5 → A
P(A) < 0.5 → B
```

这避免 generation sampling、停止条件和输出格式带来的随机性，并允许计算 ROC-AUC。

---

## 9.8 Bridge 训练协议

建议预注册以下设置：

| 项目 | 固定值 |
|---|---|
| Seeds | 0, 1, 2 |
| Optimizer | AdamW |
| Learning rate | `1e-4` |
| Weight decay | `1e-2` |
| Effective batch size | 16 |
| Max epochs | 50 |
| Warmup | 5% steps |
| Gradient clipping | 1.0 |
| Precision | bf16 |
| Early-stopping patience | 7 epochs |
| Early-stopping metric | validation balanced accuracy |
| Tie-break | lower validation answer loss |

micro-batch 可以根据 GPU 显存调整，但四个 encoder 必须使用相同 effective batch size 和 gradient accumulation。

如果某个模型发生数值不稳定，只能做对所有 encoder 同时生效的全局修改，然后全部重新运行；不能为单个 encoder 单独调 learning rate。

每个 seed 同时控制：

- bridge initialization；
- learned queries；
- batch shuffle；
- sampler order。

数据 split 始终固定为 seed 0，不随 bridge seed 改变。

---

# 10. Phase 5：完整迁移到 SCZ

AD 阶段全部完成并锁定配置后，SCZ 才开始。

依次迁移：

1. 四个 frozen encoder + attention head × 3 seeds，并运行相同 3 seeds 的 shuffled-label controls；
2. 两个 native zero-shot VLM；
3. 四个 Linear bridges × 3 seeds；
4. 四个 Resampler bridges × 3 seeds。

SCZ 不允许：

- 更换 encoder layer；
- 修改 token grid；
- 修改 bridge width；
- 修改 learning rate；
- 修改 early stopping；
- 根据 SCZ test 调 threshold；
- 只为某个 encoder 增加 preprocessing。

允许的变化只有：

- manifest；
- positive label；
- prompt 中疾病名称；
- train-only class weights。

---

# 11. 结果报告规范

## 11.1 Attention head

每个 encoder 报告：

- 3 个独立 seed 的全部结果、mean ± standard deviation 和 median；
- balanced accuracy；
- ROC-AUC；
- sensitivity；
- specificity；
- F1；
- shuffled-label control；
- parameter count；
- token shape。

## 11.2 Zero-shot

分别报告 MedGemma 和 BrainGemma3D：

- balanced accuracy；
- sensitivity/specificity；
- F1；
- exact-format、valid-fallback、nonexact-format 和 UNK/ERR rate；
- prompt；
- generation configuration。

## 11.3 Bridge

每个 disease、capacity、encoder 报告：

- 3 个独立 seed 的全部结果；
- mean ± standard deviation；
- median；
- best epoch；
- validation metric；
- trainable parameter count；
- ROC-AUC；
- balanced accuracy；
- subject-cluster bootstrap CI；
- 与其他 encoder 的 paired difference。

主要结论依据平均表现和 paired comparison，不使用“挑最好 seed”的结果。

如果 Linear 和 Resampler 得到相同 encoder 排名，可作为较强的一致性证据；如果排名不同，则应解释为：

- Linear 更接近纯 encoder separability；
- Resampler 同时测量 encoder 与轻量 vision-language alignment 的兼容性。

---

# 12. 项目复杂度控制与执行顺序

建议按以下 gate 执行：

1. 完成 manifest、loader 和 2-case smoke test。
2. 完成 AD 四个 attention heads × 3 seeds，以及对应的 3-seed shuffled-label controls。
3. 完成 AD 两个 native zero-shot VLM。
4. 先跑 AD Linear bridge 的 seed 0，验证完整训练链。
5. 跑 AD Linear 的全部 3 seeds。
6. 跑 AD Resampler 的全部 3 seeds。
7. 生成 AD 中期报告并冻结所有配置。
8. 完整迁移到 SCZ。
9. 生成最终双数据集报告。

不做：

- LoRA；
- encoder finetuning；
- LLM finetuning；
- segmentation decoder；
- Transformer classification head；
- 大型超参数 sweep；
- test-set model selection。

这样正式 bridge 共 48 runs；每个 encoder feature 只提取一次，后续训练都使用缓存的 64-token grid，避免反复执行大型 3D encoder。
