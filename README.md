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

## 批量处理与审核系统

仓库包含一个无需前端构建工具的本地 Web 系统，支持：

- 批量上传 JPG、PNG、TIFF；
- 后台并行调用模型，输出带目标 ICC 的 CMYK TIFF；
- 为浏览器生成目标 ICC 下的 sRGB 预览；
- 原图/调色结果拖动对比；
- 通过、驳回、审核备注和状态筛选；
- 打包下载所有已通过结果和审核清单。

启动：

```bash
python3 server.py \
  --model 5D2A8056_model.npz \
  --data web_data \
  --host 127.0.0.1 \
  --port 8765
```

然后访问 [http://127.0.0.1:8765](http://127.0.0.1:8765)。处理记录、审核状态和输出文件保存在 `web_data/`，该目录默认不提交到 Git。

可通过 `--workers 2` 调整并行任务数。大尺寸图像会占用较多内存，建议从 1～2 个并行任务开始。

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
