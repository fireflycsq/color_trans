# RGB → 印刷 CMYK 调色模型

这个示例从像素级对齐的 RGB 输入与 CMYK 目标图中，学习一个带三阶交叉项的颜色映射，并把目标图的印刷 ICC 配置嵌入模型和输出文件。

## 训练

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

## 推理与测试

```bash
python3 test.py \
  --model model.npz \
  --input "/path/to/test_input.jpg" \
  --output result.tif \
  --target "/path/to/test_target.jpg"
```

建议用 TIFF 保存中间结果，避免 CMYK JPEG 的二次压缩误差；需要交付 JPEG 时把输出后缀改为 `.jpg`。

## 批量处理与审核系统

仓库包含一个无需前端构建工具的本地 Web 系统，支持：

- 批量上传 JPG、PNG、TIFF；
- 后台并行调用模型，输出带目标 ICC 的 CMYK TIFF；
- 为浏览器生成目标 ICC 下的 sRGB 预览；
- 原图/调色结果拖动对比；
- 为每张结果上传像素对齐的 RGB/CMYK 目标图，并切换“原图/模型”或“目标图/模型”拖动对比；
- 通过、驳回、审核备注和状态筛选；
- 打包下载所有已通过结果和审核清单。

启动：

```bash
python3 server.py \
  --model 5D2A8056_model.npz \
  --data web_data \
  --host 127.0.0.1 \
  --port 8765 \
  --max-hue-shift 15
```

然后访问 [http://127.0.0.1:8765](http://127.0.0.1:8765)。处理记录、审核状态和输出文件保存在 `web_data/`，该目录默认不提交到 Git。

可通过 `--workers 2` 调整并行任务数。大尺寸图像会占用较多内存，建议从 1～2 个并行任务开始。

`--max-hue-shift 15` 会限制模型相对标准 ICC 转换的色相旋转，避免训练集外的蓝色服装被调成绿色。值越小保护越强；推荐从 12～18 度试起。设为 `0` 可关闭保护、恢复原始模型输出。

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

## 数据要求

- 输入和目标必须尺寸一致并像素级对齐。
- 目标必须是带 ICC profile 的 CMYK 文件。
- 单张图只能验证对该样例的拟合能力。实际训练应覆盖不同曝光、场景和主要颜色，并按整张图片划分训练集/验证集。
- 当前脚本是可解释的全局颜色映射，不改变锐度、纹理和几何结构。
