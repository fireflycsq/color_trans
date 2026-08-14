# RGB → 印刷 CMYK 调色模型

这个示例从像素级对齐的 RGB 输入与 CMYK 目标图中，学习一个带三阶交叉项的颜色映射，并把目标图的印刷 ICC 配置嵌入模型和输出文件。

## 训练

```bash
python3 train.py \
  --input "/path/to/example_input.jpg" \
  --target "/path/to/example_target.jpg" \
  --model model.npz
```

## 推理与测试

```bash
python3 test.py \
  --model model.npz \
  --input "/path/to/test_input.jpg" \
  --output result.tif \
  --target "/path/to/test_target.jpg"
```

建议用 TIFF 保存中间结果，避免 CMYK JPEG 的二次压缩误差；需要交付 JPEG 时把输出后缀改为 `.jpg`。

## 数据要求

- 输入和目标必须尺寸一致并像素级对齐。
- 目标必须是带 ICC profile 的 CMYK 文件。
- 单张图只能验证对该样例的拟合能力。实际训练应覆盖不同曝光、场景和主要颜色，并按整张图片划分训练集/验证集。
- 当前脚本是可解释的全局颜色映射，不改变锐度、纹理和几何结构。
