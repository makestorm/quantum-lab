# -*- coding: utf-8 -*-
"""QuantumLab 共用工具：JSON 序列化、时间戳、数值格式。"""
from __future__ import annotations

import datetime as _dt
import math
import os

import numpy as np


def to_jsonable(obj):
    """把 numpy / 复数 / 元组递归转换为可 json.dump 的结构。

    复数统一表示为 ``[real, imaginary]``（与 request 中初始态写法一致）。
    非有限浮点数转成字符串 ``"nan"`` / ``"inf"``，避免 JSON 非法。
    """
    if obj is None or isinstance(obj, (bool, np.bool_)):
        return None if obj is None else bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        if math.isnan(v):
            return "nan"
        if math.isinf(v):
            return "inf" if v > 0 else "-inf"
        return v
    if isinstance(obj, complex) or isinstance(obj, np.complexfloating):
        z = complex(obj)
        return [z.real, z.imag]
    if isinstance(obj, np.ndarray):
        return [to_jsonable(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, str):
        return obj
    return str(obj)


def now_iso() -> str:
    """本地时区（Asia/Shanghai）带偏移的 ISO 时间戳。"""
    tz = _dt.timezone(_dt.timedelta(hours=8))
    return _dt.datetime.now(tz).isoformat(timespec="seconds")


def now_stamp() -> str:
    """用于实验编号的时间片段：YYYYMMDD-HHMMSS。"""
    tz = _dt.timezone(_dt.timedelta(hours=8))
    return _dt.datetime.now(tz).strftime("%Y%m%d-%H%M%S")


def fmt_float(v, digits: int = 6) -> str:
    """紧凑格式化浮点数，用于报告与终端。"""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    if math.isnan(v):
        return "nan"
    if math.isinf(v):
        return "inf" if v > 0 else "-inf"
    if v == 0.0:
        return "0"
    if abs(v) >= 1e5 or abs(v) < 1e-4:
        return f"{v:.{digits}g}"
    return f"{v:.{digits}f}".rstrip("0").rstrip(".")


def complex_str(z, digits: int = 4) -> str:
    z = complex(z)
    re_, im_ = fmt_float(z.real, digits), fmt_float(abs(z.imag), digits)
    if abs(z.imag) < 1e-12:
        return re_
    return f"{re_} {'+' if z.imag >= 0 else '-'} {im_}j"


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def safe_stem(name: str, fallback: str = "fig") -> str:
    """仅保留 [a-z0-9_]，用于图片文件名。"""
    out = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in str(name).lower())
    return out.strip("_") or fallback
