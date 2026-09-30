# 人物抠图 (Portrait Cutout)

把照片里的**人物**抠出来，去掉背景，输出透明背景 PNG。纯本地运行，图片不上传，不联网（除了首次下载模型）。

- 基于 `u2net_human_seg` 人体分割模型（U²-Net, Apache-2.0）
- 只需 CPU，一张 810×1440 的照片约 **1.2 秒**
- 后处理包含**边缘内缩去描边**、**alpha 重映射**、**去渗色**，换背景不发灰边
- 支持单张 / 多张 / 整个文件夹批量
- Windows 拖拽即用，也支持命令行

## 快速开始

### 方式一：Windows 免安装（推荐给不想折腾的人）

到 [Releases](../../releases) 下载 `koutu.exe`，双击或把照片拖到 exe 上即可。不需要装 Python。

> 首次运行会下载 168MB 的模型，之后就不再需要网络。

### 方式二：从源码运行

```bash
pip install -r requirements.txt
python remove_bg.py 照片.jpg
```

首次运行会自动下载模型（约 168MB）。如果国内直连 GitHub 太慢，设置环境变量改用镜像：

```bash
set KOUTU_MIRROR=1
python remove_bg.py 照片.jpg
```

或者直接指定下载地址：

```bash
set KOUTU_MODEL_URL=https://hf-mirror.com/tomjackson2023/rembg/resolve/main/u2net_human_seg.onnx
```

### 方式三：Windows 拖拽脚本

把照片（可多选，或整个文件夹）拖到 `抠图.bat` 上松手。

## 用法

```bash
python remove_bg.py 照片.jpg                # 输出 照片_抠图.png
python remove_bg.py 照片.jpg 结果.png        # 指定输出名
python remove_bg.py 照片1.jpg 照片2.jpg      # 一次多张
python remove_bg.py 某个文件夹               # 批量处理文件夹
```

批量处理时模型只加载一次，比逐张跑明显更快。重复运行会自动跳过已生成的 `*_抠图.png`。

路径带中文或空格都可以，用引号包起来：`python remove_bg.py "我的 照片.jpg"`

## 关于透明背景

输出的 PNG 背景是**透明**的。Windows 自带的"照片"应用会把它显示成白底或黑底，这是正常现象，不是没抠干净。

想确认效果，用画图、PS、或任何支持透明通道的软件打开，或者直接粘到 PPT 里换背景色。

## 模型

`u2net_human_seg.onnx` 约 168MB，**不随仓库分发**（超过 GitHub 单文件 100MB 限制），首次运行时自动下载：

- 官方源：`danielgatis/rembg` 的 GitHub Release
- 镜像：`hf-mirror.com`（国内网络建议）

模型来源是 [U²-Net](https://github.com/xuebinqin/U-2-Net)，采用 Apache-2.0 许可。

## 效果边界

这个模型做的是**人体分割**，输出偏二值的蒙版。对全身照、半身照、证件照效果好；遇到下面这些情况边缘会不够精细：

- 大量飞散的碎发、蓬松长发
- 半透明材质（薄纱、玻璃）
- 人物与背景颜色极其接近

需要更精细的发丝边缘，可以换用 BiRefNet 系模型或配合 alpha matting 精修。

## 依赖

```
numpy
pillow
onnxruntime
```

## 许可

代码采用 MIT 许可，见 [LICENSE](LICENSE)。模型权重遵循其原始许可（Apache-2.0）。
