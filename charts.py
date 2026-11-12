# -*- coding: utf-8 -*-
"""声明式图表渲染（纯 Pillow，禁止 matplotlib）。

外部只通过两个函数使用本模块：

``render(figspec, width=1500, height=880, scale=2) -> bytes``
    把第 4 节定义的 figspec 画成 PNG 字节流。默认输出 3000×1760
    （逻辑坐标 1500×880，2× 超采样后直接作为输出分辨率）。

``render_all(result) -> [{"name": str, "png": bytes, "caption": str}]``
    按 ``result["figure_specs"]`` 的顺序渲染全部图，文件名 stem 经
    :func:`util.safe_stem` 规范化。

绘图风格约定（学术印刷导向）：
- 中文字体优先 ``C:\\Windows\\Fonts\\msyh.ttc``，回退 msyhbd / 黑体 / DejaVu / 默认字体；
- 逻辑字号：标题 28、轴标签 18、刻度 14、图例 15、注脚 13；
- 配色克制：蓝/橙/绿/紫/红/青，解析解等重要曲线用加粗虚线区分；
- 自动 1/2/5×10^k 刻度、浅灰网格、四边框、底部参数注脚；
- 支持 ``ylog``（以 1e-16 为下限的对数轴）、``xlim/ylim``、``hlines/vlines``、
  series style ``line/dashed/dots/scatter``。
"""
from __future__ import annotations

import io
import math
import os
from functools import lru_cache
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import util

__all__ = ["render", "render_all", "PALETTE", "FONT_CANDIDATES"]

# ---------------------------------------------------------------------------
# 学术配色（按序列顺序循环使用）
# ---------------------------------------------------------------------------
PALETTE = (
    "#2E75B6",  # 蓝
    "#D98324",  # 橙
    "#2E8B57",  # 绿
    "#6B5B95",  # 紫
    "#C0392B",  # 红
    "#0F8B8D",  # 青
)

# 字体候选：按优先级排列，全部缺失时退回 Pillow 默认字体。
FONT_CANDIDATES = (
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
    r"C:\Windows\Fonts\DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
)
_BOLD_CANDIDATES = (
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
)
_LOG_FLOOR = 1e-16

# 逻辑字号（会乘以 scale 再交给 Pillow）
FS_TITLE = 28
FS_LABEL = 18
FS_TICK = 14
FS_LEGEND = 15
FS_NOTE = 14

_COL_TITLE = (34, 34, 34)
_COL_AXIS = (45, 45, 45)
_COL_TICK = (70, 70, 70)
_COL_GRID = (230, 230, 230)
_COL_NOTE = (85, 85, 85)
_COL_REFLINE = (125, 125, 125)

_ANALYTIC_HINTS = ("解析", "理论", "analytic", "theory", "exact", "精确")


# ---------------------------------------------------------------------------
# 字体与颜色
# ---------------------------------------------------------------------------
@lru_cache(maxsize=64)
def _font(size: int, bold: bool = False):
    """加载字体；size 为设备像素。缺字体时逐级回退，最终用 Pillow 默认字体。"""
    candidates = _BOLD_CANDIDATES + FONT_CANDIDATES if bold else FONT_CANDIDATES
    seen = set()
    for path in candidates:
        if path in seen or not os.path.exists(path):
            continue
        seen.add(path)
        try:
            return ImageFont.truetype(path, max(1, int(size)))
        except OSError:
            continue
    try:  # Pillow ≥ 10.1 的 load_default 支持 size
        return ImageFont.load_default(size=max(1, int(size)))
    except TypeError:  # pragma: no cover - 老版本 Pillow
        return ImageFont.load_default()


# ---------------------------------------------------------------------------
# 字形兜底：Microsoft YaHei 缺少部分数学符号（⟨⟩≪⇒⁻¹…），统一替换，
# 避免图上出现“方框/豆腐块”。只在字体确实没有该字形时才替换。
# ---------------------------------------------------------------------------
# 单字符级别的替代写法（目标字体通常具备）。
_FALLBACK = {
    "\u27e8": "\u3008",   # ⟨ -> 〈
    "\u27e9": "\u3009",   # ⟩ -> 〉
    "\u27ea": "\u3008",   # ⟪
    "\u27eb": "\u3009",   # ⟫
    "\u226a": "<<",       # ≪
    "\u226b": ">>",       # ≫
    "\u21d2": "=>",       # ⇒
    "\u21d0": "<=",       # ⇐
    "\u21d4": "<=>",      # ⇔
    "\u2248": "~",        # ≈
    "\u221d": "propto",   # ∝
    "\u22a5": "perp",     # ⊥
    "\u2220": "angle",    # ∠
    "\u27f6": "->",       # ⟶
    "\u2192": "->",       # →
    "\u2190": "<-",       # ←
    "\u2194": "<->",      # ↔
    "\u2026": "...",      # …
    "\u00b7": "\u00b7",   # ·（YaHei 有，占位）
    "\u210f": "hbar",     # ℏ
    "\u2663": "\u2663",
}

# 上下标字符 -> ASCII 对应（整段替换为 ^...）。
_SUP_CHARS = set("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺")
_SUB_CHARS = set("₀₁₂₃₄₅₆₇₈₉")
_SUP_TO_ASCII = {"⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
                 "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
                 "⁻": "-", "⁺": "+"}
_SUB_TO_ASCII = {"₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4",
                 "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9"}

# 每个字体（路径 + 字号）的 .notdef 字形掩码，用于判断缺字。
_TOFU_CACHE: dict = {}
_GLYPH_CACHE: dict = {}


def _font_key(font) -> tuple:
    path = getattr(font, "path", None) or ""
    size = int(getattr(font, "size", 0) or 0)
    return (path, size)


def _mask_bytes(font, text: str):
    """返回字形掩码的字节表示（失败返回 None）。"""
    try:
        mask = font.getmask(text)
        return bytes(mask)
    except (AttributeError, ValueError, OSError):
        return None


def _has_glyph(font, ch: str) -> bool:
    """判断字体是否真的含有 ch 的字形（与 .notdef 掩码比较）。"""
    key = (_font_key(font), ch)
    cached = _GLYPH_CACHE.get(key)
    if cached is not None:
        return cached
    if ch.isspace() or ch == "":
        _GLYPH_CACHE[key] = True
        return True
    tofu = _TOFU_CACHE.get(_font_key(font))
    if tofu is None:
        tofu = _mask_bytes(font, "\ue000")  # 私用区，正常字体都没有字形
        _TOFU_CACHE[_font_key(font)] = tofu
    mask = _mask_bytes(font, ch)
    if mask is None:
        result = False
    elif tofu is not None and mask == tofu:
        result = False
    else:
        try:
            result = font.getmask(ch).getbbox() is not None
        except (AttributeError, ValueError, OSError):
            result = bool(mask)
    _GLYPH_CACHE[key] = bool(result)
    return bool(result)


def _sup_style(font) -> bool:
    """当前字体是否完整支持上下标字符（否则整轴统一用 ^ 写法，避免风格混杂）。"""
    key = _font_key(font)
    cached = _GLYPH_CACHE.get((key, "__sup__"))
    if cached is None:
        cached = all(_has_glyph(font, c) for c in "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
        _GLYPH_CACHE[(key, "__sup__")] = cached
    return bool(cached)


def _superscript_label(exponent: int, font) -> str:
    """生成对数轴刻度标签；字体缺上标字形时退回 ``10^-12`` 形式。"""
    if _sup_style(font):
        return f"10{_superscript(exponent)}"
    return f"10^{exponent}"


def _glyph_safe(text, font) -> str:
    """把文本中当前字体缺失的字符替换成安全写法。

    - 连续上/下标字符：若整段可用则保留，否则转写成 ``^...``；
    - 单字符：查 :data:`_FALLBACK`，没有映射则用 ``?``；
    - 其余字体支持的字符原样保留。
    """
    text = str(text)
    if not text:
        return ""
    out = []
    index = 0
    length = len(text)
    while index < length:
        ch = text[index]
        if ch in _SUP_CHARS or ch in _SUB_CHARS:
            table = _SUP_TO_ASCII if ch in _SUP_CHARS else _SUB_TO_ASCII
            chars = _SUP_CHARS if ch in _SUP_CHARS else _SUB_CHARS
            end = index
            while end < length and text[end] in chars:
                end += 1
            run = text[index:end]
            if all(_has_glyph(font, c) for c in run):
                out.append(run)
            else:
                marker = "^" if ch in _SUP_CHARS else "_"
                out.append(marker + "".join(table.get(c, "") for c in run))
            index = end
            continue
        if ch in "\n\t ":
            out.append(ch)
        elif _has_glyph(font, ch):
            out.append(ch)
        else:
            out.append(_FALLBACK.get(ch, "?"))
        index += 1
    return "".join(out)


def _text(draw, xy, text, font, fill, anchor=None):
    """统一文本入口：先做字形兜底再绘制。"""
    safe = _glyph_safe(text, font)
    if not safe:
        return
    if anchor is None:
        draw.text(xy, safe, font=font, fill=fill)
    else:
        draw.text(xy, safe, font=font, fill=fill, anchor=anchor)


def _rgb(color: str) -> tuple:
    """'#RRGGBB' -> (r, g, b)。"""
    c = str(color).lstrip("#")
    if len(c) != 6:
        return (46, 117, 182)
    return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))


def _rgba(color: str, alpha: int = 255) -> tuple:
    r, g, b = _rgb(color)
    return (r, g, b, int(alpha))


def _series_color(index: int) -> str:
    return PALETTE[index % len(PALETTE)]


# ---------------------------------------------------------------------------
# 数值工具
# ---------------------------------------------------------------------------
def _to_float_list(values) -> list:
    """把 list / ndarray / 带 'nan' 字符串的数据转成 float 列表（保留 nan）。"""
    if values is None:
        return []
    if isinstance(values, (str, bytes)):
        return []
    try:
        arr = np.asarray(values, dtype=float).ravel()
        return [float(v) for v in arr.tolist()]
    except (TypeError, ValueError):
        out = []
        try:
            iterator = list(values)
        except TypeError:
            return []
        for item in iterator:
            try:
                out.append(float(item))
            except (TypeError, ValueError):
                out.append(float("nan"))
        return out


def _finite(v: float) -> bool:
    return isinstance(v, (int, float)) and math.isfinite(float(v))


def _nice_num(value: float, do_round: bool) -> float:
    """返回 1/2/5×10^k 形式的“漂亮”步长/跨度。"""
    if not _finite(value) or value <= 0:
        return 1.0
    exponent = math.floor(math.log10(value))
    frac = value / (10.0 ** exponent)
    if do_round:
        nice = 1.0 if frac < 1.5 else 2.0 if frac < 3.0 else 5.0 if frac < 7.0 else 10.0
    else:
        nice = 1.0 if frac <= 1.0 else 2.0 if frac <= 2.0 else 5.0 if frac <= 5.0 else 10.0
    return nice * (10.0 ** exponent)


def _nice_ticks(lo: float, hi: float, target: int = 6):
    """线性轴的自动刻度：返回 (tick 列表, 步长)。"""
    if not (_finite(lo) and _finite(hi)) or hi <= lo:
        return [lo, hi], max(1.0, abs(hi - lo))
    span = hi - lo
    step = _nice_num(span / max(1, target - 1), True)
    if step <= 0 or not _finite(step):
        step = span / max(1, target - 1)
    start = math.ceil(lo / step - 1e-9) * step
    end = math.floor(hi / step + 1e-9) * step
    ticks = []
    v = start
    guard = 0
    while v <= end + step * 1e-6 and guard < 200:
        if abs(v) < step * 1e-9:
            v = 0.0
        ticks.append(round(float(v), 12))
        v += step
        guard += 1
    if len(ticks) < 2:
        ticks = [float(lo), float(hi)]
    return ticks, step


_SUP = {"-": "⁻", "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴",
        "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹"}


def _superscript(number: int) -> str:
    return "".join(_SUP.get(ch, ch) for ch in str(number))


def _log_ticks(lo: float, hi: float):
    """对数轴刻度：按十进制数量级取 10^k；跨度过大时隔 2/3 个数量级。

    返回 ``(tick 列表, 指数列表)``；标签由调用方结合字体可用字形生成。
    """
    lo = max(_LOG_FLOOR, float(lo))
    hi = max(float(hi), lo * 10.0)
    e0 = int(math.floor(math.log10(lo) - 1e-9))
    e1 = int(math.ceil(math.log10(hi) + 1e-9))
    span = max(1, e1 - e0)
    stride = 1 if span <= 9 else 2 if span <= 16 else 3
    exps = list(range(e0, e1 + 1, stride))
    if e1 not in exps:
        exps.append(e1)
    return [10.0 ** e for e in exps], exps


def _fmt_tick(value: float, step: float = 0.0) -> str:
    """线性轴刻度标签：按步长决定小数位，过大/过小时用科学计数。"""
    if not _finite(value):
        return "nan"
    if abs(value) < 1e-300:
        return "0"
    av = abs(value)
    if av >= 1e5 or av < 1e-4:
        return f"{value:.3g}"
    if step and _finite(step) and step > 0:
        digits = max(0, -int(math.floor(math.log10(step))))
        digits = min(digits, 6)
        text = f"{value:.{digits}f}"
        if text.startswith("-0") and float(text) == 0.0:
            text = text[1:]
        return text
    return f"{value:.6g}"


def _wrap_text(draw, text: str, font, max_width: float) -> list:
    """按像素宽度折行（中英混排逐字符，简单可靠）。"""
    text = _glyph_safe(text, font)
    if not text:
        return []
    lines = []
    for raw in text.splitlines() or [text]:
        current = ""
        for ch in raw:
            trial = current + ch
            if draw.textlength(trial, font=font) <= max_width or not current:
                current = trial
            else:
                lines.append(current)
                current = ch
        lines.append(current)
    return lines


# ---------------------------------------------------------------------------
# 主渲染流程
# ---------------------------------------------------------------------------
def render(figspec: dict, width: int = 1500, height: int = 880, scale: int = 2) -> bytes:
    """把声明式 figspec 渲染为 PNG 字节流（默认 3000×1760）。"""
    if not isinstance(figspec, dict):
        raise TypeError("figspec 必须是 dict")
    width = max(320, int(width))
    height = max(240, int(height))
    scale = max(1, int(scale))
    canvas = _draw_figure(figspec, width, height, scale)
    buf = io.BytesIO()
    canvas.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def render_all(result: dict) -> list:
    """按 ``result["figure_specs"]`` 顺序渲染全部图。"""
    specs = (result or {}).get("figure_specs") or []
    figures = []
    used = {}
    for index, figspec in enumerate(specs):
        if not isinstance(figspec, dict):
            continue
        raw_name = figspec.get("name") or f"fig_{index + 1}"
        name = util.safe_stem(raw_name, fallback=f"fig_{index + 1}")
        if name in used:
            used[name] += 1
            name = f"{name}_{used[name]}"
        else:
            used[name] = 1
        png = render(figspec)
        caption = str(figspec.get("caption") or "")
        if figspec.get("ylog"):
            extra = "纵轴为对数刻度（下限 1e-16），非正值不参与绘制。"
            caption = f"{caption}（{extra}）" if caption else extra
        figures.append({"name": name, "png": png, "caption": caption})
    return figures


def _draw_figure(figspec: dict, width: int, height: int, scale: int) -> Image.Image:
    """完整绘制一张图，返回 RGB 图像。"""
    s = int(scale)

    def px(logical: float) -> int:
        return int(round(float(logical) * s))

    fig_w, fig_h = width * s, height * s
    canvas = Image.new("RGBA", (fig_w, fig_h), (255, 255, 255, 255))

    ylog = bool(figspec.get("ylog"))
    title = str(figspec.get("title") or "")
    xlabel = str(figspec.get("xlabel") or "x")
    ylabel = str(figspec.get("ylabel") or "y")
    series_in = [item for item in (figspec.get("series") or []) if isinstance(item, dict)]
    hlines = [item for item in (figspec.get("hlines") or []) if isinstance(item, dict)]
    vlines = [item for item in (figspec.get("vlines") or []) if isinstance(item, dict)]
    notes = [str(n) for n in (figspec.get("notes") or []) if str(n).strip()]

    # ------------------------------------------------------------ 数据准备
    prepared = []
    for index, item in enumerate(series_in):
        xs = _to_float_list(item.get("x"))
        ys = _to_float_list(item.get("y"))
        count = min(len(xs), len(ys))
        prepared.append({
            "label": str(item.get("label") or f"序列 {index + 1}"),
            "x": xs[:count],
            "y": ys[:count],
            "style": str(item.get("style") or "line").lower(),
            "color": _series_color(index),
        })

    def collect_x():
        vals = []
        for series in prepared:
            vals.extend(v for v in series["x"] if _finite(v))
        vals.extend(v.get("x") for v in vlines if _finite(v.get("x")))
        return vals

    def collect_y():
        vals = []
        for series in prepared:
            vals.extend(v for v in series["y"] if _finite(v))
        vals.extend(v.get("y") for v in hlines if _finite(v.get("y")))
        return vals

    xs_all, ys_all = collect_x(), collect_y()

    xlim = _pair(figspec.get("xlim"))
    ylim = _pair(figspec.get("ylim"))
    if xlim is None:
        x0, x1 = _auto_range(xs_all)
    else:
        x0, x1 = xlim
    if ylog:
        positives = [v for v in ys_all if _finite(v) and v > 0]
        if ylim is None:
            y0, y1 = _auto_log_range(positives)
        else:
            y0 = max(_LOG_FLOOR, float(ylim[0]))
            y1 = max(float(ylim[1]), y0 * 10.0)
        if y1 <= y0:
            y0, y1 = _LOG_FLOOR, 1.0
    else:
        if ylim is None:
            y0, y1 = _auto_range(ys_all)
        else:
            y0, y1 = ylim
    if not (x1 > x0):
        x0, x1 = x0 - 0.5, x0 + 0.5
    if not (y1 > y0):
        y0, y1 = y0 - 0.5, y0 + 0.5

    # ------------------------------------------------------------ 版面测量
    left = 118
    right_pad = 44
    top = 102
    base_bottom = 86
    plot_x0, plot_x1 = px(left), fig_w - px(right_pad)
    plot_w = max(px(160), plot_x1 - plot_x0)

    note_font = _font(px(FS_NOTE), bold=False)
    measure = ImageDraw.Draw(canvas)
    note_text_width = fig_w - plot_x0 - px(right_pad)
    note_lines = []
    for note in notes:
        wrapped = _wrap_text(measure, f"· {note}", note_font, note_text_width - px(4))
        note_lines.extend(wrapped or [note])
    note_line_h = px(22)
    notes_h = len(note_lines) * note_line_h + px(8) if note_lines else 0
    plot_y0 = px(top)
    plot_y1 = fig_h - px(base_bottom) - notes_h
    plot_h = max(px(180), plot_y1 - plot_y0)
    plot_y1 = plot_y0 + plot_h

    # 坐标映射（先按逻辑范围，最后再乘到设备像素）
    def map_x(value: float) -> float:
        return (value - x0) / (x1 - x0) * (plot_w - 1)

    def map_y(value: float) -> float:
        if ylog:
            lo, hi = math.log10(y0), math.log10(y1)
            lv = math.log10(max(float(value), _LOG_FLOOR))
            frac = (lv - lo) / (hi - lo) if hi > lo else 0.0
        else:
            frac = (value - y0) / (y1 - y0)
        return (plot_h - 1) - frac * (plot_h - 1)

    # ------------------------------------------------------------ 网格与曲线
    layer = Image.new("RGBA", (plot_w, plot_h), (0, 0, 0, 0))
    layer_draw = ImageDraw.Draw(layer)

    tick_font = _font(px(FS_TICK))
    if ylog:
        y_ticks, exponents = _log_ticks(y0, y1)
        y_labels = [_glyph_safe(_superscript_label(e, tick_font), tick_font)
                    for e in exponents]
        y_step = 0.0
    else:
        y_ticks, y_step = _nice_ticks(y0, y1, target=6)
        y_labels = [_glyph_safe(_fmt_tick(v, y_step), tick_font) for v in y_ticks]
    x_ticks, x_step = _nice_ticks(x0, x1, target=max(4, min(8, int(plot_w / max(1, px(170))))))
    x_labels = [_glyph_safe(_fmt_tick(v, x_step), tick_font) for v in x_ticks]

    line_w = max(1, px(2.0))
    for tick in y_ticks:
        yy = map_y(tick)
        if -px(2) <= yy <= plot_h + px(2):
            _safe_line(layer_draw, [(0, yy), (plot_w - 1, yy)],
                       fill=(*_COL_GRID, 255), width=max(1, px(1.0)))
    for tick in x_ticks:
        xx = map_x(tick)
        if -px(2) <= xx <= plot_w + px(2):
            _safe_line(layer_draw, [(xx, 0), (xx, plot_h - 1)],
                       fill=(*_COL_GRID, 255), width=max(1, px(1.0)))

    legend_items = _draw_series(layer_draw, prepared, map_x, map_y, plot_w, plot_h, s, ylog)
    _draw_reflines(layer_draw, hlines, vlines, map_x, map_y, plot_w, plot_h, s, ylog)
    canvas.alpha_composite(layer, (plot_x0, plot_y0))

    draw = ImageDraw.Draw(canvas)

    # ------------------------------------------------------------ 边框与刻度
    axis_w = max(1, px(1.4))
    draw.rectangle([plot_x0, plot_y0, plot_x0 + plot_w - 1, plot_y0 + plot_h - 1],
                   outline=(*_COL_AXIS, 255), width=axis_w)
    tick_len = px(7)
    y_label_left = plot_x0 - tick_len - px(7)
    for tick, label in zip(y_ticks, y_labels):
        yy = plot_y0 + map_y(tick)
        if plot_y0 - 1 <= yy <= plot_y0 + plot_h:
            _safe_line(draw, [(plot_x0 - tick_len, yy), (plot_x0, yy)],
                       fill=(*_COL_AXIS, 255), width=axis_w)
            _text(draw, (plot_x0 - tick_len - px(7), yy), label,
                  tick_font, _COL_TICK, anchor="rm")
            if label:
                width = draw.textlength(label, font=tick_font)
                y_label_left = min(y_label_left, plot_x0 - tick_len - px(7) - width)
    for tick, label in zip(x_ticks, x_labels):
        xx = plot_x0 + map_x(tick)
        if plot_x0 <= xx <= plot_x0 + plot_w:
            _safe_line(draw, [(xx, plot_y0 + plot_h), (xx, plot_y0 + plot_h + tick_len)],
                       fill=(*_COL_AXIS, 255), width=axis_w)
            _text(draw, (xx, plot_y0 + plot_h + tick_len + px(7)), label,
                  tick_font, _COL_TICK, anchor="ma")

    # 轴标签
    label_font = _font(px(FS_LABEL), bold=False)
    _text(draw, (plot_x0 + plot_w / 2, plot_y0 + plot_h + px(50)), xlabel,
          label_font, _COL_AXIS, anchor="ma")
    safe_ylabel = _glyph_safe(ylabel, label_font)
    if safe_ylabel:
        try:
            yb = label_font.getbbox(safe_ylabel)
            half_h = max(px(6), (yb[3] - yb[1]) / 2)
        except (AttributeError, ValueError):  # pragma: no cover
            half_h = px(9)
        ylabel_x = max(px(10) + half_h, y_label_left - px(8) - half_h)
        _draw_rotated_text(canvas, safe_ylabel, label_font, (*_COL_AXIS, 255),
                           (ylabel_x, plot_y0 + plot_h / 2))

    # 标题（过长时自动降字号）
    if title:
        size = FS_TITLE
        title_font = _font(px(size), bold=True)
        safe_title = _glyph_safe(title, title_font)
        while (measure.textlength(safe_title, font=title_font) > fig_w - px(120)
               and size > 18):
            size -= 2
            title_font = _font(px(size), bold=True)
            safe_title = _glyph_safe(title, title_font)
        _text(draw, (fig_w / 2, px(48)), title, title_font, _COL_TITLE, anchor="mm")

    # 图例（绘制在线层之上、边框之内）
    if legend_items:
        _draw_legend(canvas, legend_items, plot_x0, plot_y0, plot_w, plot_h, s)

    # 底部参数注脚
    if note_lines:
        y_note = fig_h - notes_h + px(4)
        for line in note_lines:
            _text(draw, (plot_x0, y_note), line, note_font, _COL_NOTE, anchor="la")
            y_note += note_line_h

    return canvas.convert("RGB")


# ---------------------------------------------------------------------------
# 曲线、参照线、图例
# ---------------------------------------------------------------------------
def _pair(value):
    """把 [lo, hi] / (lo, hi) 规范为 (float, float)；非法返回 None。"""
    if value is None:
        return None
    try:
        if len(value) != 2:
            return None
        lo, hi = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None
    if not (_finite(lo) and _finite(hi)):
        return None
    return (lo, hi)


def _auto_range(values: Sequence[float]) -> tuple:
    vals = [float(v) for v in values if _finite(v)]
    if not vals:
        return (0.0, 1.0)
    lo, hi = min(vals), max(vals)
    if hi <= lo:
        delta = max(1.0, abs(lo) * 0.1)
        return (lo - delta, hi + delta)
    pad = (hi - lo) * 0.03
    return (lo - pad, hi + pad)


def _auto_log_range(values: Sequence[float]) -> tuple:
    vals = [float(v) for v in values if _finite(v) and v > 0]
    if not vals:
        return (_LOG_FLOOR, 1.0)
    lo = max(_LOG_FLOOR, min(vals))
    hi = max(vals)
    e0 = math.floor(math.log10(lo))
    e1 = math.ceil(math.log10(hi))
    if e1 <= e0:
        e1 = e0 + 1
    lo = max(_LOG_FLOOR, 10.0 ** e0)
    hi = 10.0 ** e1
    return (lo, hi)


def _safe_line(draw, points, fill, width):
    """裁剪到安全整数范围后画线，避免极端坐标导致 Pillow 溢出。"""
    safe = []
    for x, y in points:
        if not (_finite(x) and _finite(y)):
            return
        safe.append((max(-1e6, min(1e6, float(x))), max(-1e6, min(1e6, float(y)))))
    draw.line(safe, fill=fill, width=max(1, int(width)), joint="curve")


def _safe_point(x, y):
    if not (_finite(x) and _finite(y)):
        return None
    return (max(-1e6, min(1e6, float(x))), max(-1e6, min(1e6, float(y))))


def _draw_series(draw, prepared, map_x, map_y, plot_w, plot_h, s, ylog=False):
    """绘制所有序列；返回图例条目列表。"""
    legend = []
    for series in prepared:
        style = series["style"]
        label = series["label"]
        color = series["color"]
        is_analytic = any(hint in label.lower() for hint in _ANALYTIC_HINTS)
        if style == "dashed" or (style == "line" and is_analytic):
            style = "dashed"
        width_logical = 2.0
        if style == "dashed":
            width_logical = 2.8 if is_analytic else 2.2
        width = max(1, int(round(width_logical * s)))

        xs, ys = series["x"], series["y"]
        points, segments = [], []
        for x, y in zip(xs, ys):
            valid = _finite(x) and _finite(y)
            if valid and ylog and y <= 0:
                valid = False
            if not valid:
                if len(points) >= 2:
                    segments.append(points)
                points = []
                continue
            point = _safe_point(map_x(x), map_y(y))
            if point is None:
                if len(points) >= 2:
                    segments.append(points)
                points = []
                continue
            points.append(point)
        if len(points) >= 2:
            segments.append(points)

        if style in ("line", "dashed"):
            for seg in segments:
                if style == "dashed":
                    _draw_dashed(draw, seg, color, width, dash=max(2, 9 * s), gap=max(2, 6 * s))
                else:
                    _safe_line(draw, seg, fill=_rgba(color), width=width)
        if style == "dots":
            for _x, _y in _valid_points(xs, ys, map_x, map_y, ylog):
                radius = max(2, int(round(3.2 * s)))
                draw.ellipse([_x - radius, _y - radius, _x + radius, _y + radius],
                             fill=_rgba(color))
        elif style == "scatter":
            for _x, _y in _valid_points(xs, ys, map_x, map_y, ylog):
                radius = max(3, int(round(5.0 * s)))
                draw.ellipse([_x - radius, _y - radius, _x + radius, _y + radius],
                             fill=_rgba(color, 110), outline=_rgba(color), width=max(1, s))
        if style == "line" and len(segments) == 1 and len(segments[0]) == 1:
            # 单点序列：用圆点标记
            _x, _y = segments[0][0]
            radius = max(2, int(round(4.0 * s)))
            draw.ellipse([_x - radius, _y - radius, _x + radius, _y + radius],
                         fill=_rgba(color))
        legend.append({"label": label, "style": style, "color": color})
    return legend


def _valid_points(xs, ys, map_x, map_y, ylog):
    """返回设备坐标下有效的散点（用于 dots/scatter）。"""
    for x, y in zip(xs, ys):
        if not (_finite(x) and _finite(y)):
            continue
        if ylog and y <= 0:
            continue
        point = _safe_point(map_x(x), map_y(y))
        if point is not None:
            yield point


def _draw_dashed(draw, points, color, width, dash, gap):
    """沿折线画虚线（按设备像素累计弧长）。"""
    if len(points) < 2:
        return
    pattern = [max(1, int(dash)), max(1, int(gap))]
    index, remaining = 0, pattern[0]
    current = points[0]
    for target in points[1:]:
        while True:
            dx, dy = target[0] - current[0], target[1] - current[1]
            distance = math.hypot(dx, dy)
            if distance < 1e-9:
                break
            if distance <= remaining:
                if index == 0:
                    _safe_line(draw, [current, target], fill=_rgba(color), width=width)
                remaining -= distance
                current = target
                break
            ratio = remaining / distance
            nxt = (current[0] + dx * ratio, current[1] + dy * ratio)
            if index == 0:
                _safe_line(draw, [current, nxt], fill=_rgba(color), width=width)
            current = nxt
            index = 1 - index
            remaining = pattern[index]


def _draw_reflines(draw, hlines, vlines, map_x, map_y, plot_w, plot_h, s, ylog):
    """绘制 hlines / vlines 及其标签。"""
    label_font = _font(int(13 * s))
    for item in hlines:
        value = item.get("y")
        if not _finite(value):
            continue
        if ylog and float(value) <= 0:
            continue
        yy = map_y(float(value))
        if not (-1 <= yy <= plot_h):
            continue
        style = str(item.get("style") or "dashed").lower()
        width = max(1, int(round(1.6 * s)))
        if style == "dashed":
            _draw_dashed(draw, [(0, yy), (plot_w - 1, yy)], "#7D7D7D", width,
                         dash=max(2, 7 * s), gap=max(2, 5 * s))
        else:
            _safe_line(draw, [(0, yy), (plot_w - 1, yy)],
                       fill=(*_COL_REFLINE, 255), width=width)
        label = str(item.get("label") or "")
        if label:
            _label_with_bg(draw, (int(8 * s), yy - int(4 * s)), label,
                           label_font, anchor="ls")
    for item in vlines:
        value = item.get("x")
        if not _finite(value):
            continue
        xx = map_x(float(value))
        if not (-1 <= xx <= plot_w):
            continue
        width = max(1, int(round(1.6 * s)))
        _draw_dashed(draw, [(xx, 0), (xx, plot_h - 1)], "#7D7D7D", width,
                     dash=max(2, 7 * s), gap=max(2, 5 * s))
        label = str(item.get("label") or "")
        if label:
            _label_with_bg(draw, (xx + int(6 * s), int(6 * s)), label,
                           label_font, anchor="la")


def _label_with_bg(draw, xy, text, font, anchor="la"):
    """带白色底衬的小标签，避免压在曲线上难以辨认。"""
    text = _glyph_safe(text, font)
    if not text:
        return
    try:
        bbox = draw.textbbox(xy, text, font=font, anchor=anchor)
    except (ValueError, TypeError):
        bbox = draw.textbbox(xy, text, font=font)
    pad = 3
    draw.rectangle([bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad],
                   fill=(255, 255, 255, 215))
    draw.text(xy, text, font=font, fill=(90, 90, 90), anchor=anchor)


def _draw_legend(canvas, items, plot_x0, plot_y0, plot_w, plot_h, s):
    """在绘图区右上角绘制带底衬的图例（必要时分成两列）。"""
    font = _font(int(FS_LEGEND * s))
    padding = int(10 * s)
    row_h = int(28 * s)
    swatch = int(34 * s)
    gap = int(8 * s)
    measure = ImageDraw.Draw(canvas)
    labels = [_glyph_safe(item["label"], font) for item in items]
    widths = [swatch + gap + measure.textlength(label, font=font)
              for label in labels]
    max_text_w = max(widths) if widths else 0
    box_w = int(max_text_w) + 2 * padding
    one_col_h = len(items) * row_h + 2 * padding
    two_col = one_col_h > plot_h * 0.72 and len(items) > 3
    cols = 2 if two_col else 1
    per_col = int(math.ceil(len(items) / cols))
    box_h = per_col * row_h + 2 * padding
    if two_col:
        box_w = int(max_text_w) * 2 + 2 * padding + int(18 * s)
    box_w = min(box_w, plot_w - int(8 * s))
    box_h = min(box_h, plot_h - int(8 * s))
    x1 = plot_x0 + plot_w - int(12 * s)
    y0 = plot_y0 + int(12 * s)
    x0 = x1 - box_w
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle([x0, y0, x1, y0 + box_h], radius=int(6 * s),
                           fill=(255, 255, 255, 238),
                           outline=(185, 185, 185, 255), width=max(1, s // 2))
    for index, item in enumerate(items):
        col = index // per_col
        row = index % per_col
        item_x = x0 + padding + col * (int(max_text_w) + int(18 * s))
        item_y = y0 + padding + row * row_h
        cy = item_y + row_h // 2
        color = item["color"]
        style = item["style"]
        if style == "dashed":
            _draw_dashed(draw, [(item_x, cy), (item_x + swatch, cy)], color,
                         max(1, int(round(2.4 * s))),
                         dash=max(2, 7 * s), gap=max(2, 5 * s))
        elif style in ("scatter", "dots"):
            radius = max(2, int(round(3.6 * s)))
            draw.ellipse([item_x + swatch // 2 - radius, cy - radius,
                          item_x + swatch // 2 + radius, cy + radius],
                         fill=_rgba(color) if style == "dots" else _rgba(color, 120),
                         outline=_rgba(color))
        else:
            _safe_line(draw, [(item_x, cy), (item_x + swatch, cy)],
                       fill=_rgba(color), width=max(1, int(round(2.0 * s))))
        _text(draw, (item_x + swatch + gap, cy), labels[index], font,
              (45, 45, 45), anchor="lm")


def _draw_rotated_text(canvas, text, font, fill, center_xy):
    """在 RGBA 画布上绘制逆时针旋转 90° 的文字（纵轴标签）。"""
    text = _glyph_safe(text, font)
    if not text:
        return
    try:
        bbox = font.getbbox(text)
    except (AttributeError, ValueError):  # pragma: no cover
        return
    width = max(1, int(bbox[2] - bbox[0]) + 8)
    height = max(1, int(bbox[3] - bbox[1]) + 8)
    tmp = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    ImageDraw.Draw(tmp).text((4 - bbox[0], 4 - bbox[1]), text, font=font, fill=fill)
    rotated = tmp.rotate(90, expand=True)
    canvas.alpha_composite(
        rotated,
        (int(center_xy[0] - rotated.width / 2), int(center_xy[1] - rotated.height / 2)),
    )
