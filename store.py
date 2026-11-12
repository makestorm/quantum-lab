# -*- coding: utf-8 -*-
"""实验记录库：目录结构、原子写、索引、历史查找与实验对比。

设计要点：

* 每个实验一个目录 ``ql-YYYYMMDD-HHMMSS-<4 位随机>``，目录内文件见 docs/CONTRACT.md 第 6 节。
* ``index.json`` 以 ``{"version": 1, "experiments": [...]}`` 保存，写入走
  “临时文件 + ``os.replace``”，避免中断留下半个 JSON。
* ``report`` / ``charts`` 等队友模块可能尚未就绪，这里延迟导入，便于单元测试注入假模块。
"""
from __future__ import annotations

import importlib
import json
import os
import re
import secrets
import shutil

from . import util

# 默认根目录：导入时的工作目录下 quantumlab-experiments（契约要求）
ROOT_DEFAULT = os.path.join(os.getcwd(), "quantumlab-experiments")

INDEX_VERSION = 1
INDEX_NAME = "index.json"
META_NAME = "meta.json"

_ID_RE = re.compile(r"^ql-\d{8}-\d{6}-[0-9a-f]{4}$")
RELATIONS = (None, "rerun", "derived", "sweep_child")


class StoreError(RuntimeError):
    """实验记录库层次上的错误（目录损坏、ID 不存在等）。"""


# --------------------------------------------------------------------- 基础
def root_path(root=None) -> str:
    """把用户给的根目录规范成绝对路径；空值回退到 :data:`ROOT_DEFAULT`。"""
    return os.path.abspath(root or ROOT_DEFAULT)


def experiment_dir(root, exp_id) -> str:
    """返回实验目录的绝对路径，并校验 ID 形态，阻止路径穿越。"""
    _check_id(exp_id)
    return os.path.join(root_path(root), exp_id)


def _check_id(exp_id) -> str:
    if not isinstance(exp_id, str) or not _ID_RE.match(exp_id):
        raise ValueError(
            f"非法实验 ID {exp_id!r}；正确形态如 ql-20261112-110359-a1b2"
        )
    return exp_id


def _load_report():
    """延迟导入报告模块，缺失时给出可读错误。"""
    try:
        return importlib.import_module("qllab.report")
    except Exception as exc:  # pragma: no cover - 依赖缺失的兜底
        raise StoreError(
            f"报告模块 qllab.report 不可用，无法写 data.csv / report.html：{exc}"
        ) from exc


def _write_json_atomic(path: str, payload) -> None:
    """先写临时文件再 ``os.replace``，保证 JSON 文件不会只写一半。"""
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    tmp = f"{path}.tmp-{os.getpid()}-{secrets.token_hex(2)}"
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(util.to_jsonable(payload), fh, ensure_ascii=False, indent=2)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def _read_json(path: str):
    # utf-8-sig：外部工具（如 Windows PowerShell）写出的 JSON 可能带 BOM
    with open(path, "r", encoding="utf-8-sig") as fh:
        return json.load(fh)


def _newest_first(entries):
    """按创建时间（再按 ID）倒序，保证排序幂等。"""
    return sorted(
        entries,
        key=lambda e: (str(e.get("created_at") or ""), str(e.get("id") or "")),
        reverse=True,
    )


def _entry_from_meta(meta_block, validation, figure_stems):
    """把 meta.json 的内容扩展为 index.json 条目（含 overall/dir/figures）。"""
    entry = dict(meta_block)
    entry["overall"] = (validation or {}).get("overall", "skipped")
    entry["dir"] = meta_block.get("id")
    entry["figures"] = list(figure_stems or [])
    return entry


def _scan_entries(root: str):
    """索引缺失/损坏时，从各实验目录重建条目。"""
    entries = []
    if not os.path.isdir(root):
        return entries
    for name in os.listdir(root):
        exp_dir = os.path.join(root, name)
        if not os.path.isdir(exp_dir) or not _ID_RE.match(name):
            continue
        meta_path = os.path.join(exp_dir, META_NAME)
        if not os.path.isfile(meta_path):
            continue
        try:
            meta_block = _read_json(meta_path)
        except (OSError, ValueError):
            continue
        if not isinstance(meta_block, dict):
            continue
        meta_block.setdefault("id", name)
        validation = {}
        try:
            validation = _read_json(os.path.join(exp_dir, "validation.json"))
        except (OSError, ValueError):
            pass
        figures = []
        fig_dir = os.path.join(exp_dir, "figures")
        if os.path.isdir(fig_dir):
            figures = sorted(
                os.path.splitext(f)[0] for f in os.listdir(fig_dir)
                if f.lower().endswith(".png")
            )
        entries.append(_entry_from_meta(meta_block, validation, figures))
    return entries


def _index_path(root: str) -> str:
    return os.path.join(root, INDEX_NAME)


def _save_index(root: str, entries) -> None:
    _write_json_atomic(
        _index_path(root),
        {"version": INDEX_VERSION, "experiments": _newest_first(entries)},
    )


# ------------------------------------------------------------------- 索引 API
def load_index(root=None):
    """读取全部实验条目（倒序）；索引缺失或损坏时从实验目录重建。"""
    root = root_path(root)
    path = _index_path(root)
    entries = None
    if os.path.isfile(path):
        try:
            payload = _read_json(path)
            if isinstance(payload, dict) and isinstance(payload.get("experiments"), list):
                entries = [e for e in payload["experiments"] if isinstance(e, dict)
                           and isinstance(e.get("id"), str)]
        except (OSError, ValueError):
            entries = None
    if entries is None:
        entries = _scan_entries(root)
        if entries:
            try:  # 尽力修复索引；失败不影响读取
                _save_index(root, entries)
            except OSError:
                pass
    return _newest_first(entries)


def _add_to_index(root: str, entry) -> None:
    entries = [e for e in load_index(root) if e.get("id") != entry.get("id")]
    entries.append(entry)
    _save_index(root, entries)


# ------------------------------------------------------------------- 写入 API
def _new_id(root: str) -> str:
    """生成唯一实验 ID（同一秒内重复创建也能区分）。"""
    for _ in range(20):
        exp_id = f"ql-{util.now_stamp()}-{secrets.token_hex(2)}"
        if not os.path.exists(os.path.join(root, exp_id)):
            return exp_id
    raise StoreError("无法生成唯一实验 ID，请稍后重试")


def _build_meta(exp_id: str, req: dict, meta) -> dict:
    extra = dict(meta or {})
    relation = extra.get("relation")
    if relation not in RELATIONS:
        raise ValueError(f"relation 必须是 {RELATIONS} 之一，收到 {relation!r}")
    block = {
        "id": exp_id,
        "created_at": util.now_iso(),
        "model": req.get("model"),
        "params": util.to_jsonable(req.get("params") or {}),
        "parent_id": extra.get("parent_id"),
        "relation": relation,
        "duration_ms": extra.get("duration_ms"),
        "solver": extra.get("solver") or (req.get("solver") or {}).get("method"),
        "question": req.get("question") or "",
    }
    if block["parent_id"] is not None:
        _check_id(block["parent_id"])
    return block


def _write_figures(fig_dir: str, figures):
    """把 render_all 返回的 PNG bytes 落盘，返回文件名 stem 列表。"""
    stems = []
    for i, fig in enumerate(figures or []):
        if not isinstance(fig, dict):
            continue
        png = fig.get("png")
        if isinstance(png, bytearray):
            png = bytes(png)
        if not isinstance(png, (bytes, memoryview)) or not png:
            continue
        name = util.safe_stem(fig.get("name") or f"figure_{i + 1}",
                              fallback=f"figure_{i + 1}")
        stem, k = name, 2
        while os.path.exists(os.path.join(fig_dir, f"{stem}.png")):
            stem = f"{name}_{k}"
            k += 1
        with open(os.path.join(fig_dir, f"{stem}.png"), "wb") as fh:
            fh.write(png)
        stems.append(stem)
    return stems


def create(root, req, result, validation, figures=None, meta=None) -> str:
    """写入一个完整实验目录并更新索引，返回实验 ID。

    参数
    ----
    root : str | None
        实验根目录，``None`` 时用 :data:`ROOT_DEFAULT`。
    req, result, validation : dict
        canonical request、求解结果、验证报告。
    figures : list | None
        ``charts.render_all`` 的返回值：``[{"name", "png", "caption"}, ...]``。
    meta : dict | None
        调用方补充的元信息：``parent_id`` / ``relation`` / ``duration_ms`` /
        ``solver`` / ``question``。
    """
    root = root_path(root)
    os.makedirs(root, exist_ok=True)
    exp_id = _new_id(root)
    exp_dir = os.path.join(root, exp_id)
    fig_dir = os.path.join(exp_dir, "figures")
    os.makedirs(fig_dir, exist_ok=False)

    formats = ((req.get("output") or {}).get("formats") or [])
    meta_block = _build_meta(exp_id, req, meta)
    try:
        _write_json_atomic(os.path.join(exp_dir, "request.json"), req)
        _write_json_atomic(os.path.join(exp_dir, "result.json"), result)
        _write_json_atomic(os.path.join(exp_dir, "validation.json"), validation)
        _write_json_atomic(os.path.join(exp_dir, META_NAME), meta_block)

        report = _load_report()
        report.write_csv(os.path.join(exp_dir, "data.csv"), result)
        stems = _write_figures(fig_dir, figures)

        if "html" in formats:
            bundle = report.bundle(req, result, validation, figures or [], meta_block)
            report.write_html(os.path.join(exp_dir, "report.html"), bundle)

        _add_to_index(root, _entry_from_meta(meta_block, validation, stems))
    except Exception:
        shutil.rmtree(exp_dir, ignore_errors=True)
        raise
    return exp_id


# ------------------------------------------------------------------- 查询 API
def _param_value_match(actual, expected) -> bool:
    """参数匹配：数值用 1e-9 绝对容差，其余严格相等。"""
    if isinstance(expected, bool) or isinstance(actual, bool):
        return actual == expected
    if isinstance(expected, (int, float)):
        try:
            return abs(float(actual) - float(expected)) <= 1e-9
        except (TypeError, ValueError):
            return False
    return actual == expected


def find(root, query=None):
    """按模型 / 参数 / 文本 / overall / 关系过滤，按时间倒序返回条目。"""
    q = dict(query or {})
    entries = load_index(root)
    model = q.get("model")
    if isinstance(model, (list, tuple, set)):
        model = set(model)
    params = q.get("params") or {}
    text = q.get("text")
    overall = q.get("overall")
    relation = q.get("relation")
    parent_id = q.get("parent_id")
    limit = q.get("limit")

    out = []
    for entry in entries:
        if model is not None:
            if isinstance(model, set):
                if entry.get("model") not in model:
                    continue
            elif entry.get("model") != model:
                continue
        if params:
            actual = entry.get("params") or {}
            if any(k not in actual or not _param_value_match(actual.get(k), v)
                   for k, v in params.items()):
                continue
        if text:
            haystack = str(entry.get("question") or "")
            if str(text).lower() not in haystack.lower():
                continue
        if overall is not None and entry.get("overall") != overall:
            continue
        if relation is not None and entry.get("relation") != relation:
            continue
        if parent_id is not None and entry.get("parent_id") != parent_id:
            continue
        out.append(entry)
        if limit is not None and len(out) >= int(limit):
            break
    return out


def get(root, exp_id):
    """读取一个实验：meta / request / result / validation / figures 路径。"""
    exp_dir = experiment_dir(root, exp_id)
    if not os.path.isdir(exp_dir):
        raise KeyError(f"实验不存在：{exp_id}")
    meta_block = _read_json(os.path.join(exp_dir, META_NAME))
    request = _read_json(os.path.join(exp_dir, "request.json"))
    result = _read_json(os.path.join(exp_dir, "result.json"))
    validation = _read_json(os.path.join(exp_dir, "validation.json"))
    figures = {}
    fig_dir = os.path.join(exp_dir, "figures")
    if os.path.isdir(fig_dir):
        for name in sorted(os.listdir(fig_dir)):
            if name.lower().endswith(".png"):
                figures[os.path.splitext(name)[0]] = os.path.join(fig_dir, name)
    report_path = os.path.join(exp_dir, "report.html")
    return {
        "meta": meta_block,
        "request": request,
        "result": result,
        "validation": validation,
        "figures": figures,
        "dir": exp_dir,
        "report": report_path if os.path.exists(report_path) else None,
    }


# ------------------------------------------------------------------- 对比 API
def _scalar_union(rows):
    names = []
    for row in rows:
        for name in (row.get("scalars") or {}):
            if name not in names:
                names.append(name)
    return names


def _value_equal(a, b) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        scale = max(1.0, abs(float(a)), abs(float(b)))
        return abs(float(a) - float(b)) <= 1e-12 * scale
    return a == b


def compare_entries(root, ids):
    """对比若干实验：参数差异、关键 scalars 差异与总体验证状态。"""
    ids = list(ids)
    if len(ids) < 1:
        raise ValueError("compare 至少需要一个实验 ID")
    rows = []
    for exp_id in ids:
        exp = get(root, exp_id)
        result = exp.get("result") or {}
        validation = exp.get("validation") or {}
        rows.append({
            "id": exp_id,
            "dir": exp.get("dir"),
            "meta": exp.get("meta") or {},
            "model": (exp.get("meta") or {}).get("model"),
            "created_at": (exp.get("meta") or {}).get("created_at"),
            "question": (exp.get("meta") or {}).get("question"),
            "params": (exp.get("meta") or {}).get("params") or {},
            "overall": validation.get("overall", "skipped"),
            "counts": validation.get("counts") or {},
            "failed": validation.get("failed") or [],
            "scalars": util.to_jsonable(result.get("scalars") or {}),
        })

    param_names = []
    for row in rows:
        for name in row.get("params") or {}:
            if name not in param_names:
                param_names.append(name)
    param_diff = {}
    for name in param_names:
        values = {row["id"]: (row.get("params") or {}).get(name) for row in rows}
        ordered = list(values.values())
        differs = any(not _value_equal(ordered[0], v) for v in ordered[1:])
        param_diff[name] = {"values": values, "differs": differs}

    scalar_diff = {}
    for name in _scalar_union(rows):
        values = {row["id"]: (row.get("scalars") or {}).get(name) for row in rows}
        ordered = list(values.values())
        differs = any(not _value_equal(ordered[0], v) for v in ordered[1:])
        scalar_diff[name] = {"values": values, "differs": differs}

    overall_diff = {row["id"]: row.get("overall") for row in rows}
    failed_diff = {row["id"]: row.get("failed") for row in rows}
    models = {row.get("model") for row in rows}
    return {
        "ids": ids,
        "rows": rows,
        "diff": {
            "params": param_diff,
            "scalars": scalar_diff,
            "overall": overall_diff,
            "failed": failed_diff,
            "models_differ": len(models) > 1,
            "any_failed": any(row.get("overall") not in ("pass", "skipped")
                              for row in rows),
        },
    }
