# RGB → 印刷 CMYK 调色模型

这个项目从像素级对齐的 RGB 输入与 CMYK 目标图中学习印刷调色方向。推荐路径是 **ICC 基线 + 场景自适应 CMYK 残差 LUT**，并由小 CNN 分别编码全图和人像裁切。原有三阶多项式、以及固定（非自适应）CMYK 残差 LUT 仍可加载。

## 推荐：自适应 CMYK 残差 LUT

```text
Input RGB
    ├─ ICC ──────────────────────────────────────► CMYK 基线
    ├─ Thumbnail stats（亮度直方图 / 黑白点）
    ├─ Global Encoder (Small CNN + stats)
    │     ├─ 1D 相对亮度 S 曲线（反差/密度）
    │     ├─ 17³×4 偏色残差 + Confidence
    │     └─ look：黑白拉伸、中间调、S、高光软压、去金黄
    └─ MediaPipe Person/Skin → Crop → Portrait Encoder（同样结构）
         ↓
    CMYK = ICC + 1D_S(相对 luma) + gate × (3D − mean) + look(CMYK) + 遮罩 × 人像
         ↓
    edge-lift → Final CMYK
```

全图 CNN 看 256×256 缩略图，并读入该缩略图的亮度直方图与黑白点，这样夜景灰雾和舞台高光不会共用同一条绝对亮度曲线。1D 按本图拉伸后的相对亮度查表；17³×4 残差仍对 CMYK 均值中心化，只学偏色。look 头默认从接近恒等映射起步（拉伸/S/去黄都是学出来的混合量，不是一上来就满血去灰）。人像分支用 MediaPipe 抠人（或皮肤），裁切后再编码第二套 1D+3D+look，只在遮罩内叠到全局结果上。ICC 只做固定基线、不反传。损失是人工 CMYK 的 Huber，加上 naive RGB→Lab 外观项（`--appearance-weight`）。`--icc-look-weight` 默认 0：打开会逼 look 去拟合 ICC 预览差，容易把印刷 CMYK 拉离 target、val ΔE 差过纯 ICC。已有 `adaptive_cmyk_lut_v2` / `v1` / `adaptive_rgb_lut_v1` 的 `.pt` **不能**接着用这条结构，必须从 `--stage global` 重训。v1/v2 仍可加载推理。

需要 PyTorch：

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install mediapipe   # 人像阶段
mkdir -p models
```

数据配对与下面「数据配对」相同：`--pair-dir`、`--input-dir` / `--target-dir` 或 `--manifest` 均可。必须提供 `--target-icc`。先训全局，再训人像：

```bash
python3 train_adaptive_lut.py \
  --stage global \
  --pair-dir dataset/pairs \
  --target-icc profiles/PSOcoated_v3.icc \
  --model models/adaptive_global.pt \
  --report models/adaptive_global.report.json \
  --val-ratio 0.1 \
  --grid-size 17 \
  --channels 32 \
  --thumbnail 256 \
  --epochs 6 \
  --lr 1e-3 \
  --samples-per-image 8192 \
  --max-samples 1500000 \
  --eval-samples-per-image 4096 \
  --max-eval-samples 250000 \
  --huber-delta 0.125 \
  --luma-weight 1.5 \
  --cmyk-weight 1.0 \
  --appearance-weight 1.0 \
  --icc-look-weight 0 \
  --punch-weight 0.35 \
  --warmth-weight 0.25 \
  --lut-l1 0.01 \
  --smoothness 0.03 \
  --tone-bins 17 \
  --tone-smoothness 0.01 \
  --seed 42

python3 train_adaptive_lut.py \
  --stage portrait \
  --region person \
  --pair-dir dataset/pairs \
  --target-icc profiles/PSOcoated_v3.icc \
  --model models/adaptive_global.pt \
  --output models/adaptive_portrait.pt \
  --report models/adaptive_portrait.report.json \
  --val-ratio 0.1 \
  --epochs 6 \
  --lr 1e-3 \
  --samples-per-image 8192 \
  --max-samples 1500000 \
  --mask-threshold 0.45 \
  --seed 42
```

人像阶段会冻结全局 CNN，只训人像裁切上的第二张 LUT。`--region skin` 只在皮肤上混合人像 LUT。输入和目标分目录时：

```bash
python3 train_adaptive_lut.py \
  --stage global \
  --input-dir dataset/input \
  --target-dir dataset/target \
  --target-icc profiles/PSOcoated_v3.icc \
  --model models/adaptive_global.pt \
  --val-ratio 0.1 \
  --epochs 6
```

`test.py` / `server.py` 直接加载 `.pt`：

```bash
python3 test.py \
  --model models/adaptive_portrait.pt \
  --input in.jpg \
  --output result.tif

Mac 上整图 LUT 插值在 CPU 上（比原先 numpy 查表快，也比 MPS 扫全图快）。CNN 缩略图可用 `--device mps`，一般对总耗时帮助不大。`--device auto` 会选 CUDA，否则 CPU。

`--edge-lift` 仍作用在最终 CMYK 上（默认轮廓减 K）。新训的模型默认**关掉** `--shadow-lift`，以免把调图师压暗的阴影再提亮；需要暗部减墨时再显式传入，例如 `--shadow-lift 0.06`。旧的 `.npz` 若元数据里仍写着 0.06，行为不变。

v3 模型把直方图/相对 1D/look 写进网络。`test.py` / `server.py` 在 v3 上默认再叠一层 **压黑透亮 + 暖肤**（`--de-gray-cool -0.30`，负数是加黄；舞台仍发灰、皮肤偏冷时用这个，不必等重训）。关掉用 `--no-de-gray`。还不够透把 `--de-gray-shadow-lift` 加到 `0.35`；皮肤仍冷把 cool 调到 `-0.45`。高光软压仍是 0.94。v1/v2 仍默认 `--de-gray-cool 0.55` 去金黄。已生成的文件不会自动更新，需重启服务后重新跑图。

下一轮 `--stage global` 会用更重的暗部损失和中间调暖色铰链（`--punch-weight` / `--warmth-weight`），让网络自己往透亮、偏暖靠，不必全靠后处理。

## 数据配对

多项式、残差 LUT 和自适应 LUT 使用同一套图片对参数。下面示例用 `train.py`，把命令换成 `train_adaptive_lut.py` 或 `train_residual_lut.py` 即可（后两者还必须加 `--target-icc`）。

```bash
python3 train.py \
  --input "/path/to/example_input.jpg" \
  --target "/path/to/example_target.jpg" \
  --model model.npz
```

### 大规模图片对联合训练

#### 同一目录的 `_input` / `_target` 数据

如果图片对放在同一个目录：

```text
dataset/pairs/
├── 0001_input.jpg
├── 0001_target.jpg
├── 0002_input.png
├── 0002_target.tif
├── scene_a/0003_input.jpg
└── scene_a/0003_target.jpg
```

直接使用 `--pair-dir`。程序以去掉 `_input` 或 `_target` 后的相对路径作为配对键，并按整组图片对自动划分训练集和验证集：

```bash
mkdir -p models

python3 train.py \
  --pair-dir dataset/pairs \
  --val-ratio 0.1 \
  --model models/color_model_v2.npz \
  --report models/color_model_v2.report.json \
  --samples-per-image 40000 \
  --max-samples 3000000 \
  --eval-samples-per-image 10000 \
  --max-eval-samples 500000 \
  --ridge 1.0 \
  --seed 42
```

默认后缀是 `_input` 和 `_target`。如果文件名是 `xxx_src.jpg`、`xxx_ref.jpg`，可以改为：

```bash
python3 train.py \
  --pair-dir dataset/pairs \
  --input-suffix _src \
  --target-suffix _ref \
  --val-ratio 0.1 \
  --model models/color_model_v2.npz
```

没有以这两个后缀结尾的其他图片会被忽略；只有一侧存在、同一配对键重复或主文件名为空时会停止并明确报错。

#### 输入和目标分目录

目录模式会按相对于根目录的“路径 + 文件主名”配对，扩展名可以不同。例如 `input/set_a/001.png` 会匹配 `target/set_a/001.tif`。

```text
dataset/
├── input/
│   ├── set_a/001.jpg
│   └── set_b/002.png
└── target/
    ├── set_a/001.jpg
    └── set_b/002.tif
```

自动按图片对拆分 90% 训练、10%验证：

```bash
mkdir -p models

python3 train.py \
  --input-dir dataset/input \
  --target-dir dataset/target \
  --model models/color_model_v2.npz \
  --report models/color_model_v2.report.json \
  --val-ratio 0.1 \
  --samples-per-image 40000 \
  --max-samples 3000000 \
  --eval-samples-per-image 10000 \
  --max-eval-samples 500000 \
  --ridge 1.0 \
  --seed 42
```

如果训练集与验证集已经分目录，建议显式指定，避免同一连拍组泄漏到两边：

```bash
python3 train.py \
  --input-dir dataset/train/input \
  --target-dir dataset/train/target \
  --val-input-dir dataset/val/input \
  --val-target-dir dataset/val/target \
  --val-ratio 0 \
  --model models/color_model_v2.npz
```

也可以复制 `dataset_manifest.example.csv`，通过 `split` 列明确指定 `train` 或 `val`：

```bash
python3 train.py \
  --manifest dataset_manifest.csv \
  --model models/color_model_v2.npz
```

训练器具有以下行为：

- 逐张解码并累计 20×20 正规方程，内存不会随图片数量持续增长；
- 每张图在 RGB 立方体内分层采样，减少大面积背景垄断样本；
- `--max-samples` 是所有训练图的总采样上限；
- 自动将带输入 ICC 的 RGB 图规范化为 sRGB；
- 强制目标图为 CMYK、尺寸一致、内嵌 ICC 且所有目标 ICC 二进制一致；
- 生成训练/验证 CMYK MAE、PSNR、ΔE76 分位数、RGB 色域覆盖和最差图片列表；
- 模型和报告均不记录远程机器的绝对路径。

对于 100 张 6000×4000 图片，推荐从以下配置开始：

```text
samples-per-image = 30,000～50,000
max-samples       = 3,000,000～5,000,000
ridge             = 1.0
```

不要循环执行单图训练命令；每次运行都会创建一个新模型，不会增量累积。

### 使用固定目标 ICC 文件训练

如果所有目标 CMYK 数值本来就属于同一个印刷空间，可以通过 `--target-icc` 指定固定配置文件。目标图片可以没有内嵌 ICC，图片中已有但不同的 ICC 也只用于报告，不会阻止训练：

```bash
python3 train.py \
  --pair-dir dataset/pairs \
  --target-icc profiles/JapanColor2001Coated.icc \
  --val-ratio 0.1 \
  --model models/color_model_v2.npz \
  --report models/color_model_v2.report.json \
  --samples-per-image 40000 \
  --max-samples 3000000 \
  --ridge 1.0
```

该模式会：

- 检查指定文件确实是可用的 CMYK 输出 ICC；
- 使用它计算验证集的显示色差；
- 将它完整嵌入 `.npz` 模型和后续 CMYK 输出；
- 在报告中统计目标图内嵌 ICC 的 `matching / different / missing` 数量；
- 保持目标图原有 C、M、Y、K 数值不变。

`--target-icc` 的含义是“统一指定解释”，不是把其他 CMYK 空间转换到该 ICC。如果图片实际来自不同印刷空间，应先完成 CMYK→CMYK 色彩管理转换，再进行训练。

## ICC 基线 + 3D 残差 LUT（旧路径）

这套模型先用固定目标 ICC 计算标准 CMYK，再用 3D LUT 只预测人工目标相对标准结果的 `ΔC/ΔM/ΔY/ΔK`：

```text
output CMYK = ICC(input sRGB) + confidence(RGB) × residual_LUT(RGB)
```

训练命令与原训练器的数据配对方式相同，且必须明确提供固定 CMYK ICC：

```bash
python3 train_residual_lut.py \
  --pair-dir dataset/pairs \
  --target-icc profiles/JapanColor2001Coated.icc \
  --val-ratio 0.1 \
  --model models/residual_lut_v1.npz \
  --report models/residual_lut_v1.report.json \
  --grid-size 17 \
  --samples-per-image 40000 \
  --max-samples 3000000 \
  --eval-samples-per-image 10000 \
  --max-eval-samples 500000 \
  --confidence-samples 32 \
  --smoothness 0.12 \
  --baseline-regularization 0.25 \
  --huber-delta 8 \
  --max-cmy-residual 20 \
  --max-k-residual 15 \
  --seed 42
```

训练分两遍流式读取图片：第一遍估计各 RGB 网格的残差，第二遍使用 Huber 权重降低异常图片对和错位像素的影响。每个样本按三线性权重累计到相邻 8 个 LUT 节点，训练内存由网格尺寸决定，不随照片数量增长。最后会执行相邻节点平滑、低覆盖节点向零残差回归，并保存覆盖置信度。推理遇到训练覆盖不足的颜色时会自动退回 ICC 基线。

默认安全限制为 C/M/Y 最多修正 ±20、K 最多修正 ±15（均为 0～255 CMYK 数值）。这会截断更大的人工改色，适合防止训练集外颜色被过度迁移。

若目标是尽量拟合人工 CMYK，加 `--fit-human`。它会把残差上限放到满量程（±255），减弱向 ICC 零残差的回拉，并放宽 Huber 阈值，让大且一致的人工改动能写入 LUT。覆盖不足的颜色仍按置信度退回 ICC。已有按安全限制训练的 `.npz` 无法补回被截断的残差，必须重新训练：

```bash
python3 train_residual_lut.py \
  --pair-dir dataset/pairs \
  --target-icc profiles/PSOcoated_v3.icc \
  --fit-human \
  --val-ratio 0.1 \
  --model models/residual_lut_human.npz \
  --report models/residual_lut_human.report.json \
  --grid-size 17 \
  --samples-per-image 40000 \
  --max-samples 3000000 \
  --seed 42
```

仍可用 `--max-cmy-residual`、`--max-k-residual` 等参数覆盖该预设。图片对必须像素对齐，否则大幅残差会被当成「这种 RGB 都应这样改」写进全局表。

报告会同时给出：

- 纯 ICC 基线的 CMYK MAE、PSNR 和 ΔE76；
- ICC + LUT 的同组指标；
- 相对基线的平均 ΔE 改善百分比；
- 验证集中效果最差的图片对；
- LUT 节点覆盖数与平均置信度。

推荐先使用默认 `17³` 网格。数据较少可改为 `9`，只有当颜色覆盖充分且验证集证明有效时再尝试 `33`。目标图必须为像素对齐的 CMYK 图；固定 ICC 模式解释其现有 CMYK 数值，不会偷偷转换或改写目标像素。

## 人像第二阶段

全局 LUT 不能把「人」和背景分开处理。若需要先做全图印刷调色，再抠出人像（皮肤、头发、服装）单独往 target 的调色方向靠，先训练全局模型，再训练人像残差：

```text
输出 = 全局(ICC + LUT) + 人像遮罩 × 人像LUT(RGB)
```

人像 LUT 学的是 `target CMYK − 全局输出`，只在人像轮廓内采样，并按 RGB 立方体分层，避免大面积服装淹没头发和皮肤。推理时背景仍走全局结果。需要 mediapipe 做整身分割：

```bash
python3 -m pip install mediapipe

python3 train_portrait_skin.py \
  --model models/residual_lut_human.npz \
  --pair-dir dataset/pairs \
  --region person \
  --fit-human \
  --output models/residual_lut_human_portrait.npz \
  --report models/residual_lut_human_portrait.portrait.report.json \
  --val-ratio 0.1
```

`--region person` 是默认值。若只要皮肤、不改头发和服装，可改为 `--region skin`。人像阶段默认 `--fit-human`：残差满量程、较弱平滑、一致性降权更松，让第二层尽量跟上人工对人像的改法。已有按保守参数训练的人像 `.npz` 不会自动变激进，需要重训。若验证集上变差的图明显变多，再用 `--no-fit-human` 或把 `--agreement-sigma` 降到 `8`。

### 轮廓采样实验

若只想学发丝/衣缘的人工方向，不要把整图或人像内部写进第二层，用 `--region contour`。只在人像遮罩约 50% 的那一圈采样（`target − 全局`），推理时也只把这层加在轮廓环上，脸和衣服内部、背景仍走全局 LUT。不跑全图像素采样。轮廓实验默认关掉启发式 `--edge-lift`，避免和学习到的轮廓残差叠在一起：

```bash
python3 train_portrait_skin.py \
  --model models/residual_lut_human.npz \
  --pair-dir dataset/pairs \
  --region contour \
  --fit-human \
  --output models/residual_lut_human_contour.npz \
  --report models/residual_lut_human_contour.portrait.report.json \
  --val-ratio 0.1

python3 visualize_portrait_mask.py \
  --input-dir dataset/pairs \
  --output-dir mask_previews_contour \
  --region contour
```

预览里青色应是一圈轮廓，不应铺满整个人。训练阈值默认 `0.5`（环更窄）；仍偏宽可把 `--mask-threshold` 提到 `0.7`。此实验仍需要已经训好的全局模型，只是第二层不再从全图/全身采样。

训练前先看遮罩是否套住头发和服装。四联预览从左到右是原图、青色叠加、棋盘格抠图、热力遮罩（黄线是训练阈值 0.45 的轮廓）：

```bash
python3 visualize_portrait_mask.py \
  --input /path/to/photo.jpg \
  --output-dir mask_previews \
  --region person

python3 visualize_portrait_mask.py \
  --input-dir dataset/pairs \
  --output-dir mask_previews \
  --region person
```

看叠加和抠出图：头发、衣服被青色盖住就对了；轮廓切进脸、漏掉袖子或把背景算进人，说明分割不准。目录模式会额外写出 `mask_preview_report.csv`。`test.py` 的 `--save-portrait-mask` 只存灰阶遮罩，不方便审边缘。

人在画面里太小时，整图缩到 768 后主体往往只剩几十像素，分割会失败。此时会自动再跑 2×2 切块分割，并用脸框放大出上半身裁切补一次。若仍然不足 32 个有效像素：训练时该图跳过人像阶段、只贡献全局 LUT；推理时人像 LUT 不生效，整张图走全局结果。远景小人无法可靠抠出时，这是预期回退，不是漏跑。

`test.py` / `server.py` 会自动启用写入 `.npz` 的人像阶段。训练和推理必须都能做人像分割。

推理时还会沿人像轮廓减一点 K（以及少量 C），用来提亮发丝和衣缘上发闷的混合像素。默认峰值约 K `0.05`、C `0.02`，不需要重训；已有人像模型也会生效。轮廓发白就减小，仍发暗就加大：

```bash
python3 test.py --model models/residual_lut_human_portrait.npz \
  --input in.jpg --output result.tif --edge-lift 0.08
```

`--edge-lift 0` 可关掉。审核服务同样支持 `--edge-lift`。没有人像 LUT 时，只要装了 mediapipe，全局模型也会用整身轮廓做这一圈提亮。

人像阴影、舞台这类暗场，深色衣服和背景不在轮廓上，`edge-lift` 帮不上。若要把暗部再减墨提亮，推理可按原图亮度减 K（峰值示例 K `0.06`、C/M/Y 各 `0.035`）。新模型默认关闭，以免抵消调图师的 S 曲线压暗：

```bash
python3 test.py --model models/residual_lut_human_portrait.npz \
  --input in.jpg --output result.tif --shadow-lift 0.08
```

审核服务同样支持 `--shadow-lift`。不需要时省略或设为 `0`。

## 推理与测试

```bash
python3 test.py \
  --model models/adaptive_portrait.pt \
  --input "/path/to/test_input.jpg" \
  --output result.tif \
  --target "/path/to/test_target.jpg"
```

旧残差 LUT 把 `--model` 换成 `.npz` 即可。建议用 TIFF 保存中间结果，避免 CMYK JPEG 的二次压缩误差；需要交付 JPEG 时把输出后缀改为 `.jpg`。

`test.py` 和 `server.py` 按扩展名识别 `.pt` 自适应模型、残差 LUT `.npz` 与旧多项式 `.npz`。残差 LUT 自带覆盖置信度和 ICC 回退，因此服务端的 `--max-hue-shift` 只作用于旧多项式模型。

## 批量处理与审核系统

仓库包含一个无需前端构建工具的本地 Web 系统，支持：

- 批量上传 JPG、PNG、TIFF；
- 后台并行调用模型，输出带目标 ICC 的 CMYK TIFF（v3 默认压黑透亮 + 暖肤；v1/v2 仍默认去金黄去灰）；
- 为浏览器生成该输出的 sRGB 预览；
- 原图/调色结果拖动对比；
- 为每张结果上传像素对齐的 RGB/CMYK 目标图，并切换“原图/模型”或“目标图/模型”拖动对比；
- 通过、驳回、审核备注和状态筛选；
- 删除整个调色批次（含原图、结果和审核记录）；
- 打包下载所有已通过结果和审核清单。

启动：

```bash
python3 server.py \
  --model models/adaptive_portrait.pt \
  --data web_data \
  --host 127.0.0.1 \
  --port 8765 \
  --max-upload-mb 512
```

然后访问 [http://127.0.0.1:8765](http://127.0.0.1:8765)。处理记录、审核状态和输出文件保存在 `web_data/`，该目录默认不提交到 Git。

可通过 `--workers 2` 调整并行任务数。大尺寸图像会占用较多内存，建议从 1～2 个并行任务开始。

对旧多项式模型，`--max-hue-shift 15` 会限制模型相对标准 ICC 转换的色相旋转，避免训练集外的蓝色服装被调成绿色。值越小保护越强；推荐从 12～18 度试起。设为 `0` 可关闭保护。残差 LUT 模型使用自身的覆盖置信度、残差限幅和 ICC 回退，不使用该参数。

### 系统目录

```text
server.py                 Python 标准库 Web/API 服务
web/index.html            上传与审核页面
web/app.js                批量上传、轮询和审核交互
web/styles.css            响应式界面样式
5D2A8056_model.npz        默认 RGB→CMYK 模型与目标 ICC
web_data/<batch-id>/      本地批次数据（运行时生成）
```

服务默认只监听 `127.0.0.1`。若要部署到局域网或公网，应在前面增加带身份认证、HTTPS 和上传限制的反向代理。

### 目标图上传出现 `Failed to fetch`

更新代码后必须重启 `server.py`。输入图和目标图都会按 1 MB 分块落盘，目标图在 ICC 转换前缩小审核预览，可显著降低大尺寸图片上传时的内存峰值。默认单文件上限为 512 MB，可通过 `--max-upload-mb` 调整。

如果前面使用 Nginx，还需要在站点配置的 `server` 或 `location` 中设置：

```nginx
client_max_body_size 512m;
proxy_read_timeout 300s;
proxy_send_timeout 300s;
```

然后检查并重载：

```bash
sudo nginx -t
sudo nginx -s reload
```

若仍然失败，先确认 Python 服务没有因内存不足退出，并检查服务终端日志。页面现在会把网络中断提示为“无法连接服务器”，HTTP 413/400 等服务端错误则会显示相应状态或具体校验原因。

## 数据要求

- 输入和目标必须尺寸一致并像素级对齐。
- 目标必须是 CMYK 文件；未提供 `--target-icc` 时必须内嵌且统一 ICC，固定 ICC 模式允许目标不带配置文件。
- 单张图只能验证对该样例的拟合能力。实际训练应覆盖不同曝光、场景和主要颜色，并按整张图片划分训练集/验证集。
- 当前脚本是可解释的全局颜色映射，不改变锐度、纹理和几何结构。

## 批量比较图片自带 ICC 与指定 ICC

如果一批图片内嵌了不同的 ICC，希望在不改变原始像素值的情况下，比较“按图片自带 ICC
显示”和“统一按指定 ICC 显示”的差异，可以运行：

```bash
python3 compare_icc.py \
  --input-dir /path/to/cmyk-images \
  --assigned-icc profiles/JapanColor2001Coated.icc \
  --output-dir icc-comparison
```

程序递归扫描 JPG、TIFF 和 PNG，为每张有效图片生成三联 JPEG：自带 ICC 渲染、指定 ICC
渲染和 ΔE76 热力图。同时生成按平均色差排序的 `icc_comparison_report.json` 和可用 Excel
打开的 `icc_comparison_report.csv`。默认使用相对色度意图和黑点补偿，并把长边缩到 1600
像素后生成预览和计算色差；可通过 `--max-dimension` 调整。

这里的指定 ICC 是用来重新解释不变的 RGB/CMYK 像素值，并非完成源空间到目标空间的颜色
转换。指定 ICC、内嵌 ICC 的色彩空间必须与图片模式一致；缺少 ICC 或配置不匹配的图片会记录
在报告中而不会中断整批处理。
