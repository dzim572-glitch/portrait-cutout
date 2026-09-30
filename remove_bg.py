# -*- coding: utf-8 -*-
"""人物抠图：把照片背景去掉，只留人物，输出透明背景 PNG。

用法:
    python remove_bg.py 照片.jpg                  # 自动输出 照片_抠图.png
    python remove_bg.py 照片.jpg 结果.png          # 指定输出文件名
    python remove_bg.py 照片1.jpg 照片2.jpg ...    # 一次多张
    python remove_bg.py 某个文件夹                 # 批量处理文件夹里所有图片

依赖: pillow numpy onnxruntime
模型: u2net_human_seg.onnx，放在脚本同目录或同目录的 models/ 下
"""
import os
import sys

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageFilter

SIZE = 320  # u2net 固定输入尺寸
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# 边缘处理参数
ERODE_PX = 1        # 内缩像素，用来清掉背景色描边
LOW, SPAN = 0.30, 0.50   # alpha 重映射: clip((a-LOW)/SPAN)
BG_COLOR = np.array([215.0, 205.0, 200.0], dtype=np.float32)  # 估计的背景色，用于去渗色

SUFFIX = "_抠图"
EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")

MODEL_NAME = "u2net_human_seg.onnx"
# 模型不随代码分发（168MB），缺了会按顺序尝试这些地址。
# 国内直连 GitHub release 经常很慢，所以备了镜像。
MODEL_URLS = (
    "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net_human_seg.onnx",
    "https://hf-mirror.com/tomjackson2023/rembg/resolve/main/u2net_human_seg.onnx",
)

_sess = None  # 模型只加载一次，批量处理时省掉重复开销


def _say(msg):
    """控制台编码千奇百怪，输出失败也不能让程序挂掉。"""
    try:
        print(msg)
    except UnicodeEncodeError:
        enc = sys.stdout.encoding or "utf-8"
        print(msg.encode(enc, "replace").decode(enc, "replace"))


def _search_dirs():
    """按优先级列出可能存放模型的位置。

    打包成 exe 后 onnx 会被解到 sys._MEIPASS，但每次启动都要解 176MB
    很慢，所以 exe 同目录也要找一遍——把模型放 exe 旁边就能跳过解包。
    """
    dirs = []
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        dirs.append(meipass)
    if getattr(sys, "frozen", False):
        dirs.append(os.path.dirname(os.path.abspath(sys.executable)))
    dirs.append(os.path.dirname(os.path.abspath(__file__)))
    for d in list(dirs):
        dirs.append(os.path.join(d, "models"))
    dirs += [os.environ.get("TEMP", ""), os.environ.get("TMP", "")]
    return [d for d in dirs if d]


def find_model():
    """找到就返回路径，找不到返回 None。"""
    for d in _search_dirs():
        p = os.path.join(d, "u2net_human_seg.onnx")
        if os.path.isfile(p) and os.path.getsize(p) > 1024 * 1024:
            return p
    return None


def _download_dir():
    """模型存哪：优先脚本/exe 同目录（下次不用重下），不行就退回临时目录。"""
    if getattr(sys, "frozen", False):
        base = os.path.dirname(os.path.abspath(sys.executable))
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    for d in (os.path.join(base, "models"), base):
        try:
            os.makedirs(d, exist_ok=True)
            probe = os.path.join(d, ".w_test")
            open(probe, "w").close()
            os.remove(probe)
            return d
        except OSError:
            continue
    return os.environ.get("TEMP", ".")


def ensure_model():
    """模型不在就自动下载。167MB，只需下一次。"""
    p = find_model()
    if p:
        return p

    import socket
    import urllib.request

    dest_dir = _download_dir()
    dest = os.path.join(dest_dir, MODEL_NAME)

    urls = list(MODEL_URLS)
    env_url = os.environ.get("KOUTU_MODEL_URL")
    if env_url:
        urls = [env_url]              # 想强制走某个地址就设这个环境变量
    elif os.environ.get("KOUTU_MIRROR"):
        urls = urls[::-1]             # 国内网络直连 GitHub 太慢时用这个

    # 连接卡住就报错，好让循环换下一个地址，而不是无限等
    socket.setdefaulttimeout(30)

    def hook(blocks, bs, total):
        if total <= 0:
            return
        done = min(blocks * bs, total)
        pct = done * 100 // total
        bar = "#" * (pct // 4)
        sys.stdout.write("\r  [%-25s] %3d%%  %5.1f/%.1f MB"
                         % (bar, pct, done / 1048576, total / 1048576))
        sys.stdout.flush()

    tmp = dest + ".part"
    last_err = None
    for i, url in enumerate(urls, 1):
        _say("下载模型 (%d/%d)，约 168MB，只需一次" % (i, len(urls)))
        _say("  " + url)
        _say("  保存到: " + dest)
        try:
            urllib.request.urlretrieve(url, tmp, hook)
            sys.stdout.write("\n")
            os.replace(tmp, dest)
            _say("下载完成。")
            return dest
        except Exception as e:
            sys.stdout.write("\n")
            last_err = e
            _say("  这个地址失败了（%s），换下一个" % type(e).__name__)
            try:
                os.remove(tmp)
            except OSError:
                pass

    raise SystemExit(
        "模型下载失败: %s\n请手动下载 %s 后放到: %s"
        % (last_err, MODEL_NAME, dest_dir)
    )


def session():
    global _sess
    if _sess is None:
        _sess = ort.InferenceSession(
            ensure_model(), providers=["CPUExecutionProvider"]
        )
    return _sess


def predict_mask(rgb):
    """跑模型，返回与原图同尺寸的 0~1 前景概率图。"""
    small = rgb.resize((SIZE, SIZE), Image.LANCZOS)
    x = np.asarray(small, dtype=np.float32) / 255.0
    x = (x - MEAN) / STD
    x = x.transpose(2, 0, 1)[None]

    sess = session()
    out = sess.run(None, {sess.get_inputs()[0].name: x})[0]
    pred = out[0, 0]
    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)

    m = Image.fromarray((pred * 255).astype(np.uint8), "L").resize(rgb.size, Image.LANCZOS)
    return np.asarray(m, dtype=np.float32) / 255.0


def erode(m, px):
    if px <= 0:
        return m
    im = Image.fromarray((m * 255).astype(np.uint8), "L").filter(ImageFilter.MinFilter(px * 2 + 1))
    return np.asarray(im, dtype=np.float32) / 255.0


def despill(rgb, alpha):
    """半透明边缘像素上残留的浅色背景，按 F=(C-(1-a)B)/a 反解拉回前景色。"""
    f = rgb.astype(np.float32)
    a = alpha[..., None]
    rec = np.clip((f - (1.0 - a) * BG_COLOR) / np.maximum(a, 0.35), 0, 255)
    w = np.clip((0.95 - alpha) / 0.35, 0, 1)[..., None]
    return np.clip(f * (1 - w) + rec * w, 0, 255)


def cut_one(src_path, out_path):
    src = Image.open(src_path).convert("RGB")
    rgb = np.asarray(src, dtype=np.uint8)

    alpha = np.clip((erode(predict_mask(src), ERODE_PX) - LOW) / SPAN, 0.0, 1.0)
    fg = despill(rgb, alpha)

    rgba = np.dstack([fg.round().astype(np.uint8), (alpha * 255).round().astype(np.uint8)])
    Image.fromarray(rgba, "RGBA").save(out_path)

    a = rgba[..., 3]
    _say("saved: %s  size=%s  透明=%d  实心=%d  半透明边缘=%d"
         % (out_path, (src.width, src.height), (a == 0).sum(), (a == 255).sum(),
            ((a > 0) & (a < 255)).sum()))


def default_out(src_path):
    """照片.jpg -> 照片_抠图.png，已带后缀就不重复加。"""
    base = os.path.splitext(src_path)[0]
    if base.endswith(SUFFIX):
        base = base[: -len(SUFFIX)]
    return base + SUFFIX + ".png"


def collect(paths):
    """展开文件夹，返回待处理的图片列表。"""
    files = []
    for p in paths:
        if os.path.isdir(p):
            found = sorted(
                os.path.join(p, n) for n in os.listdir(p)
                if n.lower().endswith(EXTS) and SUFFIX not in n
            )
            if not found:
                _say("[跳过] 文件夹里没有图片: %s" % p)
            files.extend(found)
        elif os.path.isfile(p):
            files.append(p)
        else:
            _say("[跳过] 找不到: %s" % p)
    return files


def main(argv):
    if not argv:
        raise SystemExit(__doc__)

    # 两参数且第二个是 .png：认作「输入 输出」形式（可覆盖已存在的文件）
    two_arg_out = (
        len(argv) == 2
        and argv[1].lower().endswith(".png")
        and not os.path.isdir(argv[0])
    )

    if two_arg_out:
        cut_one(argv[0], argv[1])
        return

    files = collect(argv)
    if not files:
        raise SystemExit("没有可处理的图片")

    _say("共 %d 张，开始处理..." % len(files))
    for i, f in enumerate(files, 1):
        _say("[%d/%d] %s" % (i, len(files), os.path.basename(f)))
        try:
            cut_one(f, default_out(f))
        except Exception as e:
            _say("  [失败] %s: %s" % (os.path.basename(f), e))
    _say("全部完成。")


if __name__ == "__main__":
    main(sys.argv[1:])
