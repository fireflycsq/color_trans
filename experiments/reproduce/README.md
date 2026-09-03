# 调色实验复现手册

本手册用于重新运行“调色实验记录”中的历史实验，同时保留代码版本、命令、日志、环境、模型和报告。历史记录不会被覆盖。

## 1. 当前复现边界

报告中没有保存 Git commit、完整命令、依赖版本、训练/验证文件清单和基础模型哈希。因此：

- 597/66 完整数据实验可以做参数和趋势复现；
- 15/2 小样本实验必须找回当时的 17 对图片及其划分；
- portrait 实验必须找回当时使用的 global 模型，或先重新训练并明确标记为“新基座复现”；
- 根据报告时间推断的 commit 只代表当时最近的已提交版本；若原实验使用未提交代码，无法逐数值复现。

## 2. 准备资产

不要把大数据和模型复制进 Git worktree。统一使用绝对路径：

```bash
export REPRO_DATA_DIR=/absolute/path/to/dataset/pairs
export REPRO_SMALL_DATA_DIR=/absolute/path/to/historical-15-train-2-val
export REPRO_ICC=/absolute/path/to/profiles/PSOcoated_v3.icc
export REPRO_MODEL_DIR=/absolute/path/to/historical/models
export REPRO_OUTPUT_ROOT=/absolute/path/to/reproduced_experiments
```

检查 ICC：

```bash
shasum -a 256 "$REPRO_ICC"
```

期望值：

```text
c30ad2c01e8f93135ec7682c535e0a81bc2d177c301e196376c5f5838b5c8e86
```

完整数据应产生 597 个训练 pair 和 66 个验证 pair。为了长期稳定复现，建议立即建立显式 manifest，不再依赖目录自动划分。

## 3. 每次运行的标准流程

通用包装器会创建 detached worktree，并保存：

```text
<experiment>/
├── command.sh
├── git-commit.txt
├── environment.txt
├── assets.txt
├── assets.sha256
├── train.log
└── exit-status.txt
```

调用形式：

```bash
bash experiments/reproduce/run_one.sh EXPERIMENT_ID COMMIT -- COMMAND ARGUMENTS...
```

训练命令中的 `--output` 和 `--report` 必须直接指向：

```bash
"$REPRO_OUTPUT_ROOT/EXPERIMENT_ID/model-file"
"$REPRO_OUTPUT_ROOT/EXPERIMENT_ID/report-file"
```

包装器拒绝覆盖已有实验目录。需要重跑时使用新的 ID，例如 `_rerun02`，不要删除第一次结果。

## 4. 历史提交映射与执行顺序

| 顺序 | 实验 | 推断 commit | 数据 | 依赖 |
|---:|---|---|---|---|
| 1 | color_model_v2 | 001e1999 | 593/66 | 无 |
| 2 | residual_lut_v1 | fa1ded3f | 15/2 | 旧小样本 |
| 3 | residual_lut_human | 81812093 | 593/66 | 无 |
| 4 | residual_lut_human_portrait | 81812093 | 593/66 | 第 3 步模型 |
| 5 | residual_lut_human_portrait_hard | 81812093 | 15/2 | 第 3 步模型、旧小样本 |
| 6 | residual_lut_human_skin | 81812093 | 593/66 | 第 3 步模型 |
| 7 | adaptive_portrait（早期） | 8702ba61 | 15/2 | 对应早期 global `.pt` |
| 8 | adaptive_global_20260825_2 | f9b9f289 | 597/66 | 无 |
| 9 | adaptive_global | c0592d91 | 597/66 | 无 |
| 10 | adative_global_20260826_v1 | 101e22d9 | 597/66 | 无 |
| 11 | adative_global_20260826_v2 | 9fd355ae | 597/66 | 无 |
| 12 | adaptive_global_20260826_V3 | 071ae85f | 597/66 | 无 |
| 13 | adaptive_global_20260827_v1 | 80649810 | 597/66 | 无 |
| 14 | adaptive_portrait_20260827_v1 | 80649810 | 597/66 | 对应 global `.pt` |
| 15 | adaptive_portrait_20260828_v1 | 2a6fa0ed | 15/2 | 旧小样本、global `.pt` |
| 16 | adaptive_portrait_lut_only_20260828_v2 | b2f6be6e | 597/66 | global `.pt` |
| 17 | adaptive_portrait_dual_20260831_v1 | 3079ac7c | 15/2 | 旧小样本、global `.pt` |
| 18 | adaptive_portrait_dual_20260901_v1 | 3079ac7c | 597/66 | global `.pt` |

必须先运行 global，再运行引用它的 portrait。若历史基础模型找不到，在新记录中增加：

```text
reproduction_type=new-base
parent_model_sha256=<新模型哈希>
```

不要把它标为 exact reproduction。

## 5. 命令模板

### 5.1 color_model_v2

在 `001e1999` 上运行 `train.py`：

```bash
id=color_model_v2
bash experiments/reproduce/run_one.sh "$id" 001e1999 -- \
  python3 train.py \
  --pair-dir "$REPRO_DATA_DIR" \
  --val-ratio 0.1 --seed 42 \
  --model "$REPRO_OUTPUT_ROOT/$id/color_model_v2.npz" \
  --report "$REPRO_OUTPUT_ROOT/$id/color_model_v2.report.json" \
  --samples-per-image 40000 --max-samples 3000000 \
  --eval-samples-per-image 10000 --max-eval-samples 500000 \
  --ridge 1.0
```

如果该 commit 不支持 `--pair-dir`，使用当时的 `--input-dir/--target-dir` 或先建立 manifest；以该 commit 的 `python3 train.py --help` 为准。

### 5.2 residual_lut_human

```bash
id=residual_lut_human
bash experiments/reproduce/run_one.sh "$id" 81812093 -- \
  python3 train_residual_lut.py \
  --pair-dir "$REPRO_DATA_DIR" --val-ratio 0.1 --seed 42 \
  --target-icc "$REPRO_ICC" \
  --model "$REPRO_OUTPUT_ROOT/$id/residual_lut_human.npz" \
  --report "$REPRO_OUTPUT_ROOT/$id/residual_lut_human.report.json" \
  --grid-size 17 --samples-per-image 40000 --max-samples 3000000 \
  --eval-samples-per-image 10000 --max-eval-samples 500000 \
  --smoothness 0.06
```

其三个 portrait 派生实验使用 `81812093` 的 `train_portrait_skin.py`，并将 `--model` 指向上一步 `.npz`。完整 person、15/2 hard person、完整 skin 分别使用完整数据、旧小样本、完整数据。报告中记录的共同参数为：

```text
grid=17, mask-threshold=0.45, max CMY/K residual=255,
person smoothness=0.02, skin smoothness=0.06, seed=42
```

运行前先查看历史 CLI：

```bash
git show 81812093:train_portrait_skin.py | less
```

### 5.3 Adaptive global 通用模板

```bash
id=EXPERIMENT_ID
commit=HISTORICAL_COMMIT
bash experiments/reproduce/run_one.sh "$id" "$commit" -- \
  python3 train_adaptive_lut.py \
  --stage global \
  --pair-dir "$REPRO_DATA_DIR" \
  --target-icc "$REPRO_ICC" \
  --output "$REPRO_OUTPUT_ROOT/$id/$id.pt" \
  --report "$REPRO_OUTPUT_ROOT/$id/$id.report.json" \
  --val-ratio 0.1 --seed 42 \
  --grid-size 17 --channels 32 --thumbnail 256 \
  --epochs EPOCHS --lr LR \
  --cmyk-weight CMYK --appearance-weight APPEARANCE \
  --luma-weight LUMA --icc-look-weight ICC_LOOK \
  --lut-l1 LUT_L1 --smoothness SMOOTHNESS
```

报告还原出的差异参数：

| 实验 | epochs | lr | appearance | luma | punch | k-punch | warmth | ICC-look | LUT L1 / smooth |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| adaptive_global_20260825_2 | 6 | 1e-3 | 1.0 | 1.0 | 默认 | 默认 | 默认 | 0.35 | .01/.03 |
| adaptive_global | 6 | 1e-3 | 1.0 | 1.0 | 默认 | 默认 | 默认 | 0 | .01/.03 |
| adative_global_20260826_v1 | 6 | 1e-3 | 1.0 | 2.0 | .35 | 当时默认 | 0 | 0 | .01/.03 |
| adative_global_20260826_v2 | 6 | 1e-3 | 1.0 | 2.0 | .35 | .35 | 0 | 0 | .01/.03 |
| adaptive_global_20260826_V3 | 6 | 1e-3 | 1.0 | 1.5 | .35 | .35 | 0 | 0 | .01/.03 |
| adaptive_global_20260827_v1 | 15 | 1e-3 | .5 | .75 | .1 | .15 | 0 | 0 | .01/.03 |

早期 commit 的 CLI 不完全一致。每次正式运行前必须执行：

```bash
python3 train_adaptive_lut.py --help | tee "$REPRO_OUTPUT_ROOT/$id/cli-help.txt"
```

### 5.4 最新完整 dual-mask portrait

```bash
id=adaptive_portrait_dual_20260901_v1
export REPRO_BASE_MODEL="$REPRO_MODEL_DIR/adaptive_global.pt"

bash experiments/reproduce/run_one.sh "$id" 3079ac7c -- \
  python3 train_adaptive_lut.py \
  --stage portrait --region skin \
  --pair-dir "$REPRO_DATA_DIR" --val-ratio 0.1 --seed 42 \
  --target-icc "$REPRO_ICC" --model "$REPRO_BASE_MODEL" \
  --output "$REPRO_OUTPUT_ROOT/$id/$id.pt" \
  --report "$REPRO_OUTPUT_ROOT/$id/$id.report.json" \
  --epochs 20 --lr 5e-5 \
  --portrait-lut-only --portrait-dual-mask \
  --portrait-skin-gate-low 0.25 --portrait-skin-gate-high 0.60 \
  --portrait-residual-limit-cmy 0.022 --portrait-residual-limit-k 0.03 \
  --portrait-neutral-max-regression 0.01 \
  --portrait-skin-max-regression 0.005 \
  --mask-threshold 0.35 \
  --cmyk-weight 1 --appearance-weight 0.25 --luma-weight 0.5 \
  --punch-weight 0 --k-punch-weight 0 --warmth-weight 0 \
  --icc-look-weight 0 --lut-l1 0.06 --smoothness 0.08 \
  --early-stopping-patience 3 --lr-patience 1 \
  --lr-factor 0.5 --min-lr 1e-6
```

参考结果：整体 ΔE≈4.056、portrait≈4.095、skin≈5.413、neutral≈2.760，best epoch≈17。

### 5.5 其余 adaptive portrait 参数

| 实验 | 数据 | commit | region | epochs/lr | 关键参数 |
|---|---|---|---|---|---|
| adaptive_portrait_20260827_v1 | 597/66 | 80649810 | person | 15 / 1e-3 | appearance=1, luma=1.5, punch=.35, k-punch=.35, warmth=.25 |
| adaptive_portrait_20260828_v1 | 15/2 | 2a6fa0ed | person | 15 / 1e-4 | appearance=.5, luma=.75, punch=.1, k-punch=.15, warmth=.25, threshold=.4 |
| adaptive_portrait_lut_only_20260828_v2 | 597/66 | b2f6be6e | skin | 15 / 1e-4 | LUT-only, limits=.05/.04, appearance=.25, k-punch=.1, warmth=.25, L1/smooth=.03/.05, threshold=.4 |
| adaptive_portrait_dual_20260831_v1 | 15/2 | 3079ac7c | skin | 15 / 1e-4 | dual, limits=.025/.06, gate=.15-.50, appearance=.25, L1/smooth=.04/.06, threshold=.4 |

这些实验均应使用各自当时的 global 基础模型。找不到时可以选定一个新 global，但必须更改实验 ID 并记录其 SHA-256。

## 6. 运行后校验与汇总

确认程序退出成功：

```bash
find "$REPRO_OUTPUT_ROOT" -name exit-status.txt -exec sh -c \
  'printf "%s " "$1"; cat "$1"' _ {} \;
```

确认报告和模型存在：

```bash
find "$REPRO_OUTPUT_ROOT" -type f \
  \( -name '*.report.json' -o -name '*.pt' -o -name '*.npz' \) -print
```

每个模型完成后补充模型哈希：

```bash
find "$REPRO_OUTPUT_ROOT" -type f \( -name '*.pt' -o -name '*.npz' \) \
  -exec shasum -a 256 {} \; > "$REPRO_OUTPUT_ROOT/models.sha256"
```

建议最终汇总字段：实验 ID、reproduction type、commit、父模型哈希、数据 manifest 哈希、ICC 哈希、环境、best epoch、整体 mean/p95、portrait/skin/neutral ΔE、CMYK MAE、是否 accepted。

## 7. 推荐实际执行批次

1. 冻结完整数据 manifest、旧小样本 manifest、ICC 和依赖环境。
2. 重跑 `color_model_v2`、`residual_lut_human`。
3. 重跑六个 adaptive global，并统一复评。
4. 根据历史父模型映射重跑 portrait；无法确认父模型的实验标记为 `new-base`。
5. 最后重跑完整 dual-mask，作为当前主线结果。
6. 生成新的总表；历史报告只作为 reference，不覆盖、不改写。

