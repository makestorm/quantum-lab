# -*- coding: utf-8 -*-
"""QuantumLab 命令行入口。

子命令
------
``run / sweep / list / show / rerun / derive / compare / export / selfcheck``。

安全边界
--------
* 不调用 ``eval`` / ``exec``，不执行任何用户字符串；请求只经
  :func:`qllab.spec.normalize_request` 白名单规范化。
* ``derive --set`` 只允许白名单 dotted path（如 ``params.Omega_z=0``、
  ``time.n_steps=801``），值按 JSON/数字解析，不做表达式求值。
* 不使用 ``subprocess``。

退出码
------
0 成功；1 执行失败或验证存在 fail；2 参数/请求非法。
"""
from __future__ import annotations

import argparse
import copy
import importlib
import json
import math
import os
import re
import secrets
import shutil
import sys
import tempfile
import time

from . import spec, store, util

_STATUS_ZH = {"pass": "通过", "partial": "部分验证未通过", "skipped": "全部检查跳过"}


# ------------------------------------------------------------------ 基础工具
class _InputError(ValueError):
    """读取用户请求时的输入错误（对应退出码 2）。"""


class _MissingDependency(RuntimeError):
    """依赖模块（models/validate/charts/report）尚未就绪。"""


def _import_core(name):
    try:
        return importlib.import_module(name)
    except Exception as exc:  # pragma: no cover - 依赖缺失时的兜底
        raise _MissingDependency(
            f"依赖模块 {name} 不可用：{type(exc).__name__}: {exc}"
        ) from exc


def _json_dumps(payload) -> str:
    return json.dumps(util.to_jsonable(payload), ensure_ascii=False, indent=2)


def _print_json(payload) -> None:
    print(_json_dumps(payload))


def _short_path(path: str) -> str:
    """尽量用相对 cwd 的短路径打印，控制终端行宽。"""
    if not path:
        return ""
    try:
        rel = os.path.relpath(path)
    except ValueError:
        return path
    return rel if not rel.startswith("..") and len(rel) <= len(path) else path


def _fail(args, message, code=2):
    """统一失败输出：``--json`` 时 stdout 也给机器可读错误。"""
    if getattr(args, "json", False):
        _print_json({"ok": False, "code": code, "error": message})
    print(f"错误：{message}", file=sys.stderr)
    return code


def _exc_message(exc) -> str:
    if isinstance(exc, KeyError) and exc.args:
        return str(exc.args[0])
    return str(exc)


def _dedupe(items):
    """按首次出现顺序去重，保留原始顺序（幂等，便于反复调用）。"""
    out, seen = [], set()
    for item in items or ():
        key = item if isinstance(item, (str, int, float, bool, type(None))) else repr(item)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _fmt_value(value) -> str:
    if _is_number(value):
        return util.fmt_float(value, 8)
    if value is None:
        return "—"
    try:
        text = json.dumps(util.to_jsonable(value), ensure_ascii=False)
    except (TypeError, ValueError):
        text = str(value)
    return text if len(text) <= 48 else text[:45] + "…"


def _params_brief(params, width=88) -> str:
    if not isinstance(params, dict) or not params:
        return "（无参数）"
    parts = [f"{k}={_fmt_value(v)}" for k, v in params.items()]
    text = "; ".join(parts)
    return text if len(text) <= width else text[: width - 1] + "…"


def _figure_stems(exp_dir: str):
    fig_dir = os.path.join(exp_dir, "figures")
    if not os.path.isdir(fig_dir):
        return []
    return sorted(
        os.path.splitext(name)[0] for name in os.listdir(fig_dir)
        if name.lower().endswith(".png")
    )


# ---------------------------------------------------------------- 请求读取
def _load_raw(args):
    """从 --request / --spec / --stdin 读取原始请求 dict。"""
    try:
        if getattr(args, "request", None):
            # utf-8-sig：容忍 Windows 工具写出的 UTF-8 BOM
            with open(args.request, "r", encoding="utf-8-sig") as fh:
                raw = json.load(fh)
        elif getattr(args, "spec", None) is not None:
            raw = json.loads(args.spec)
        elif getattr(args, "stdin", False):
            raw = json.loads(sys.stdin.read())
        else:
            raise _InputError("需要通过 --request、--spec 或 --stdin 提供请求")
    except OSError as exc:
        raise _InputError(f"无法读取请求文件：{exc}") from exc
    except json.JSONDecodeError as exc:
        hint = ""
        source = args.spec if getattr(args, "spec", None) is not None else None
        if source is not None and re.search(r"\{\s*[A-Za-z_\u4e00-\u9fff]", str(source)):
            # PowerShell 把内联参数里的双引号吞掉，JSON 会变成 {model:...} 这种形态。
            hint = ("；提示：PowerShell 会剥掉内联参数中的双引号，请改用 "
                    "--request <文件>，或把 JSON 用引号包好后通过管道 --stdin 传入")
        raise _InputError(f"请求不是合法 JSON：{exc}{hint}") from exc
    if not isinstance(raw, dict):
        raise _InputError("请求必须是 JSON 对象（dict）")
    return raw


# ---------------------------------------------------------------- 执行流水线
def _execute_pipeline(root, req, relation=None, parent_id=None):
    """执行 求解 → 验证 → 图表 → 落盘，返回 (summary, result, validation, figures)。"""
    models = _import_core("qllab.models")
    validate = _import_core("qllab.validate")
    charts = _import_core("qllab.charts")

    started = time.perf_counter()
    result = models.run_model(req)
    if not isinstance(result, dict):
        raise RuntimeError("models.run_model 必须返回 dict")
    validation = validate.validate(req, result)
    if not isinstance(validation, dict):
        raise RuntimeError("validate.validate 必须返回 dict")

    # 确认块（describe_request）已经打印过请求规范化产生的提示，摘要里只补充新增提示，
    # 避免同一条提示在终端出现两次；JSON 摘要仍保留完整 warnings 便于追溯。
    request_warnings = _dedupe(req.get("warnings") or [])
    new_warnings = _dedupe([w for w in (result.get("warnings") or [])
                            if w not in set(request_warnings)])
    warnings = _dedupe(request_warnings + new_warnings)
    figures = []
    if bool((req.get("output") or {}).get("figures", True)):
        try:
            figures = charts.render_all(result) or []
        except Exception as exc:  # 图表失败不抹掉物理结果，但要显式提示
            new_warnings.append(f"图表渲染失败，已跳过：{type(exc).__name__}: {exc}")
            warnings = _dedupe(warnings + new_warnings[-1:])

    duration_ms = int(round((time.perf_counter() - started) * 1000))
    meta = {
        "parent_id": parent_id,
        "relation": relation,
        "duration_ms": duration_ms,
        "solver": (req.get("solver") or {}).get("method"),
        "question": req.get("question") or "",
    }
    exp_id = store.create(root, req, result, validation, figures, meta)
    exp_dir = store.experiment_dir(root, exp_id)

    formats = (req.get("output") or {}).get("formats") or []
    report_path = None
    if "html" in formats:
        candidate = os.path.join(exp_dir, "report.html")
        report_path = candidate if os.path.exists(candidate) else None

    summary = {
        "ok": True,
        "id": exp_id,
        "dir": exp_dir,
        "model": req.get("model"),
        "params": util.to_jsonable(req.get("params") or {}),
        "overall": validation.get("overall", "skipped"),
        "counts": validation.get("counts") or {},
        "checks": validation.get("checks") or [],
        "failed": validation.get("failed") or [],
        "scalars": util.to_jsonable(result.get("scalars") or {}),
        "figures": _figure_stems(exp_dir),
        "report": report_path,
        "result_path": os.path.join(exp_dir, "result.json"),
        "duration_ms": duration_ms,
        "relation": relation,
        "parent_id": parent_id,
        "warnings": warnings,
        "new_warnings": new_warnings,
    }
    return summary, result, validation, figures


def _summary_exit_code(summary) -> int:
    counts = summary.get("counts") or {}
    if summary.get("failed") or summary.get("overall") == "partial":
        return 1
    if int(counts.get("fail") or 0) > 0:
        return 1
    return 0


# ---------------------------------------------------------------- 终端摘要
def _failed_details(summary):
    checks = summary.get("checks") or []
    failed_ids = set(summary.get("failed") or [])
    details = []
    for check in checks:
        if not isinstance(check, dict):
            continue
        if check.get("status") != "fail" and check.get("id") not in failed_ids:
            continue
        label = check.get("name") or check.get("id")
        metric = check.get("metric") or {}
        metric_text = ""
        if isinstance(metric, dict) and metric.get("name"):
            metric_text = f"{metric.get('name')}={_fmt_value(metric.get('value'))}"
        tolerance = check.get("tolerance")
        tol_text = f"，容差 {_fmt_value(tolerance)}" if tolerance is not None else ""
        detail = f"{check.get('id')}（{label}）"
        if metric_text:
            detail += f"：{metric_text}"
        detail += tol_text
        details.append(detail)
    if not details:
        details = [str(item) for item in failed_ids]
    return details


def _print_summary(summary) -> None:
    overall = summary.get("overall")
    counts = summary.get("counts") or {}
    print("")
    print("【实验已记录】")
    print(f"实验 ID：{summary.get('id')}")
    print(f"实验目录：{_short_path(summary.get('dir'))}")
    print(f"模型：{summary.get('model')}")
    print(f"总体验证：{overall}（{_STATUS_ZH.get(overall, overall)}）")
    print(f"检查计数：pass {counts.get('pass', 0)} / fail {counts.get('fail', 0)}"
          f" / skipped {counts.get('skipped', 0)}")

    failed = summary.get("failed") or []
    if failed:
        print("失败检查：")
        for line in _failed_details(summary):
            print(f"  - {line}")
    else:
        print("失败检查：无")

    scalars = summary.get("scalars") or {}
    picked = [(k, v) for k, v in scalars.items() if _is_number(v)]
    if picked:
        print("关键数值：")
        for key, value in picked[:8]:
            print(f"  - {key} = {util.fmt_float(value, 8)}")
    else:
        print("关键数值：无标量输出")

    figures = summary.get("figures") or []
    if figures:
        print(f"图表（{len(figures)} 张）：{', '.join(figures)}")
    else:
        print("图表：无")
    report = summary.get("report")
    print(f"报告：{_short_path(report) if report else '未生成（output.formats 不含 html）'}")
    print(f"耗时：{summary.get('duration_ms')} ms")

    # 只补充确认块里没有出现过的提示（请求规范化提示已在上方确认块打印）。
    warnings = summary.get("new_warnings")
    if warnings is None:
        warnings = summary.get("warnings") or []
    if warnings:
        print("补充提示：")
        for item in warnings[:6]:
            text = str(item)
            print(f"  - {text if len(text) <= 88 else text[:87] + '…'}")


# ---------------------------------------------------------------- 执行类命令
def _perform_request(args, raw, relation=None, parent_id=None) -> int:
    """规范化 → 确认块 → 执行 → 摘要。run/sweep/rerun/derive 共用。"""
    try:
        req = spec.normalize_request(raw)
    except spec.SpecError as exc:
        return _fail(args, f"请求非法：{exc}", 2)

    if not getattr(args, "json", False):
        print(spec.describe_request(req))
        print("")
        print("【执行中】求解 → 验证 → 绘图 → 落盘 ……")

    try:
        summary, _result, _validation, _figures = _execute_pipeline(
            args.root, req, relation=relation, parent_id=parent_id)
    except _MissingDependency as exc:
        return _fail(args, str(exc), 1)
    except Exception as exc:
        return _fail(args, f"执行失败：{type(exc).__name__}: {exc}", 1)

    summary["confirm"] = spec.describe_request(req)
    if getattr(args, "json", False):
        _print_json(summary)
    else:
        _print_summary(summary)
    return _summary_exit_code(summary)


def _cmd_run(args) -> int:
    """run：执行一次实验。"""
    try:
        raw = _load_raw(args)
    except _InputError as exc:
        return _fail(args, str(exc), 2)
    return _perform_request(args, raw)


def _build_sweep_block(args):
    if not getattr(args, "param", None):
        return None
    block = {"param": args.param}
    values = getattr(args, "values", None)
    if values:
        parts = [p.strip() for p in str(values).split(",") if p.strip()]
        try:
            block["values"] = [float(p) for p in parts]
        except ValueError as exc:
            raise _InputError("--values 需要逗号分隔的数字，如 0,0.5,1") from exc
    elif args.start is not None and args.stop is not None:
        block["start"] = args.start
        block["stop"] = args.stop
        if args.num is not None:
            block["num"] = args.num
    else:
        raise _InputError("sweep --param 需要 --values 或 --start/--stop（可加 --num）")
    return block


def _cmd_sweep(args) -> int:
    """sweep：在请求中注入参数扫描后执行。"""
    try:
        raw = _load_raw(args)
        sweep_block = _build_sweep_block(args)
    except _InputError as exc:
        return _fail(args, str(exc), 2)
    if sweep_block is not None:
        raw["sweep"] = sweep_block
    return _perform_request(args, raw)


def _cmd_rerun(args) -> int:
    """rerun：完全相同的规范请求重跑，生成新 ID，保留父实验。"""
    try:
        parent = store.get(args.root, args.id)
    except ValueError as exc:
        return _fail(args, str(exc), 2)
    except KeyError as exc:
        return _fail(args, _exc_message(exc), 2)
    except OSError as exc:
        return _fail(args, f"读取父实验失败：{exc}", 1)
    if not getattr(args, "json", False):
        print(f"【重跑】父实验：{args.id}（父实验保留，将生成新实验 ID）")
        print("")
    return _perform_request(args, parent["request"], relation="rerun",
                            parent_id=args.id)


# ---------------------------------------------------------------- derive
def _allowed_set_paths(model):
    """返回 ``--set`` 白名单：dotted path -> 值类型。"""
    allowed = {}
    for name in spec.MODEL_PARAMS.get(model, ()):
        allowed[f"params.{name}"] = "int" if name == "n_levels" else "float"
    allowed["time.t_start"] = "float"
    allowed["time.t_end"] = "float"
    allowed["time.n_steps"] = "int"
    for key in spec.DEFAULT_TOL:
        allowed[f"tolerance.{key}"] = "float"
    allowed["initial_state.kind"] = "str"
    allowed["initial_state.n"] = "int"
    allowed["initial_state.alpha"] = "json"
    allowed["initial_state.components"] = "json"
    allowed["sweep.param"] = "str"
    allowed["sweep.values"] = "json"
    allowed["sweep.start"] = "float"
    allowed["sweep.stop"] = "float"
    allowed["sweep.num"] = "int"
    allowed["output.formats"] = "json"
    allowed["output.figures"] = "bool"
    allowed["question"] = "str"
    return allowed


def _parse_set_value(path, kind, text):
    """按白名单类型解析 ``--set`` 的值；不做表达式求值。"""
    raw = text.strip()
    if kind == "float":
        try:
            value = float(raw)
        except ValueError as exc:
            raise spec.SpecError(f"--set {path} 需要实数，收到 {text!r}") from exc
        if not math.isfinite(value):
            raise spec.SpecError(f"--set {path} 需要有限实数，收到 {text!r}")
        return value
    if kind == "int":
        try:
            return int(raw, 10)
        except ValueError as exc:
            raise spec.SpecError(f"--set {path} 需要整数，收到 {text!r}") from exc
    if kind == "json":
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise spec.SpecError(
                f"--set {path} 需要 JSON 值（如 [0,1]），收到 {text!r}"
            ) from exc
    if kind == "bool":
        low = raw.lower()
        if low in ("true", "1", "yes", "on"):
            return True
        if low in ("false", "0", "no", "off"):
            return False
        raise spec.SpecError(f"--set {path} 需要布尔值，收到 {text!r}")
    if kind == "str":
        if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "\"'":
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return raw[1:-1]
        return raw
    raise spec.SpecError(f"--set {path} 的类型不受支持：{kind}")


def _set_dotted(block, parts, value):
    current = block
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            child = {}
            current[part] = child
        current = child
    current[parts[-1]] = value


def _apply_sets(request, assignments):
    """在父 request 副本上套用 ``--set`` 白名单修改。"""
    model = request.get("model")
    if model not in spec.MODEL_PARAMS:
        raise spec.SpecError(f"父实验模型 {model!r} 无法派生")
    allowed = _allowed_set_paths(model)
    modified = copy.deepcopy(request)
    if not assignments:
        raise spec.SpecError("derive 至少需要一个 --set，例如 --set params.Omega_z=0")
    for item in assignments:
        if "=" not in item:
            raise spec.SpecError(f"--set 需要 path=value 形式，收到 {item!r}")
        path, raw_value = item.split("=", 1)
        path = path.strip()
        kind = allowed.get(path)
        if kind is None:
            raise spec.SpecError(
                f"--set 不允许字段 {path!r}；允许：{', '.join(sorted(allowed))}"
            )
        value = _parse_set_value(path, kind, raw_value)
        _set_dotted(modified, path.split("."), value)
    return modified


def _cmd_derive(args) -> int:
    """derive：基于父实验改参数另存新实验（relation=derived）。"""
    try:
        parent = store.get(args.root, args.id)
    except ValueError as exc:
        return _fail(args, str(exc), 2)
    except KeyError as exc:
        return _fail(args, _exc_message(exc), 2)
    except OSError as exc:
        return _fail(args, f"读取父实验失败：{exc}", 1)
    try:
        modified = _apply_sets(parent["request"], args.set or [])
    except spec.SpecError as exc:
        return _fail(args, f"派生参数非法：{exc}", 2)
    if not getattr(args, "json", False):
        print(f"【派生】父实验：{args.id}（父实验保留，将生成新实验 ID）")
        print("")
    return _perform_request(args, modified, relation="derived", parent_id=args.id)


# ---------------------------------------------------------------- 查询类命令
def _cmd_list(args) -> int:
    """list：列出历史实验（可过滤）。"""
    query = {"model": args.model, "text": args.text, "limit": args.limit}
    try:
        entries = store.find(args.root, query)
    except OSError as exc:
        return _fail(args, f"读取实验索引失败：{exc}", 1)
    if args.json:
        _print_json({"ok": True, "root": store.root_path(args.root),
                     "count": len(entries), "experiments": entries})
        return 0
    if not entries:
        print("（暂无实验记录）")
        return 0
    print(f"【实验列表】共 {len(entries)} 条（新的在前）")
    for index, entry in enumerate(entries, 1):
        relation = entry.get("relation") or "首次"
        parent = f" ← {entry.get('parent_id')}" if entry.get("parent_id") else ""
        print(f"{index:>2}. {entry.get('id')}  {entry.get('created_at')}")
        print(f"    模型：{entry.get('model')}  关系：{relation}{parent}"
              f"  验证：{entry.get('overall')}")
        print(f"    参数：{_params_brief(entry.get('params'))}")
    return 0


def _cmd_show(args) -> int:
    """show：展示一个实验的元信息、验证表与关键数值。"""
    try:
        exp = store.get(args.root, args.id)
    except ValueError as exc:
        return _fail(args, str(exc), 2)
    except KeyError as exc:
        return _fail(args, _exc_message(exc), 2)
    except OSError as exc:
        return _fail(args, f"读取实验失败：{exc}", 1)

    meta = exp.get("meta") or {}
    request = exp.get("request") or {}
    validation = exp.get("validation") or {}
    result = exp.get("result") or {}
    payload = {
        "ok": True,
        "meta": meta,
        "request": request,
        "validation": validation,
        "scalars": util.to_jsonable(result.get("scalars") or {}),
        "eigen_values": util.to_jsonable((result.get("eigen") or {}).get("values")),
        "summary": util.to_jsonable(result.get("summary") or {}),
        "figures": exp.get("figures") or {},
        "report": exp.get("report"),
        "dir": exp.get("dir"),
        "result_path": os.path.join(exp.get("dir") or "", "result.json"),
    }
    if args.json:
        _print_json(payload)
        return 0

    print(f"【实验 {meta.get('id', args.id)}】")
    print(f"创建时间：{meta.get('created_at')}  模型：{meta.get('model')}")
    relation = meta.get("relation") or "首次"
    parent = f"  父实验：{meta.get('parent_id')}" if meta.get("parent_id") else ""
    print(f"关系：{relation}{parent}  耗时：{meta.get('duration_ms')} ms")
    print(f"求解器：{meta.get('solver')}")
    print(f"目录：{_short_path(exp.get('dir'))}")
    print(f"参数：{_params_brief(request.get('params') or meta.get('params'))}")
    question = meta.get("question") or ""
    if question:
        text = question if len(question) <= 88 else question[:87] + "…"
        print(f"原始问题：{text}")

    overall = validation.get("overall")
    counts = validation.get("counts") or {}
    print("")
    print(f"总体验证：{overall}（{_STATUS_ZH.get(overall, overall)}）"
          f"  pass {counts.get('pass', 0)} / fail {counts.get('fail', 0)}"
          f" / skipped {counts.get('skipped', 0)}")
    checks = validation.get("checks") or []
    if checks:
        print("检查表：")
        for check in checks:
            status = str(check.get("status", "?"))
            mark = {"pass": "PASS", "fail": "FAIL", "skipped": "SKIP"}.get(status, "?")
            metric = check.get("metric") or {}
            metric_text = ""
            if isinstance(metric, dict) and metric.get("name"):
                metric_text = f"  {metric.get('name')}={_fmt_value(metric.get('value'))}"
            print(f"  [{mark}] {check.get('id')}  {check.get('name')}{metric_text}")

    scalars = result.get("scalars") or {}
    picked = [(k, v) for k, v in scalars.items() if _is_number(v)]
    if picked:
        print("关键数值：")
        for key, value in picked[:10]:
            print(f"  - {key} = {util.fmt_float(value, 8)}")
    summary_block = result.get("summary") or {}
    for bullet in (summary_block.get("bullets") or [])[:5]:
        print(f"  · {bullet}")
    failed = validation.get("failed") or []
    if failed:
        print(f"失败检查：{', '.join(str(x) for x in failed)}")
    figures = exp.get("figures") or {}
    if figures:
        print("图表：")
        for name, path in figures.items():
            print(f"  - {name}: {_short_path(path)}")
    report = exp.get("report")
    print(f"报告：{_short_path(report) if report else '未生成'}")
    return 0


def _print_compare(data) -> None:
    rows = data.get("rows") or []
    diff = data.get("diff") or {}
    print(f"【实验对比】共 {len(rows)} 个实验")
    for row in rows:
        failed = row.get("failed") or []
        tail = f"失败：{', '.join(str(x) for x in failed)}" if failed else "无失败检查"
        print(f"  - {row.get('id')}｜{row.get('model')}｜{row.get('overall')}｜{tail}")
        print(f"    参数：{_params_brief(row.get('params'))}")

    def _print_block(title, block, keys):
        print("")
        print(title)
        changed = [(name, block.get(name) or {}) for name in keys
                   if (block.get(name) or {}).get("differs")]
        if not changed:
            print("  无差异")
            return
        for name, item in changed:
            print(f"  {name}：")
            for row in rows:
                print(f"    {row.get('id')} = {_fmt_value((item.get('values') or {}).get(row.get('id')))}")

    _print_block("【参数差异】", diff.get("params") or {},
                 list((diff.get("params") or {}).keys()))
    _print_block("【关键数值差异】", diff.get("scalars") or {},
                 list((diff.get("scalars") or {}).keys()))

    print("")
    print("【验证状态】")
    for row in rows:
        failed = row.get("failed") or []
        extra = f"，失败项 {', '.join(str(x) for x in failed)}" if failed else ""
        print(f"  {row.get('id')}：{row.get('overall')}{extra}")


def _cmd_compare(args) -> int:
    """compare：对比多个实验的参数与关键数值。"""
    try:
        data = store.compare_entries(args.root, args.ids)
    except ValueError as exc:
        return _fail(args, str(exc), 2)
    except KeyError as exc:
        return _fail(args, _exc_message(exc), 2)
    except OSError as exc:
        return _fail(args, f"读取实验失败：{exc}", 1)
    payload = {"ok": True}
    payload.update(data)
    if args.json:
        _print_json(payload)
        return 0
    _print_compare(data)
    return 0


# ---------------------------------------------------------------- export
def _restore_observable_arrays(result):
    """把 result.json 里的 list 观察量还原为 numpy 数组，便于 report 写 CSV。"""
    try:
        import numpy as np
    except Exception:  # pragma: no cover - numpy 应该总是可用
        return result
    restored = dict(result)
    times = restored.get("times")
    if isinstance(times, list):
        try:
            restored["times"] = np.asarray(times, dtype=float)
        except (TypeError, ValueError):
            pass
    observables = restored.get("observables")
    if isinstance(observables, dict):
        converted = {}
        for key, value in observables.items():
            if isinstance(value, list):
                try:
                    converted[key] = np.asarray(value, dtype=float)
                except (TypeError, ValueError):
                    converted[key] = value
            else:
                converted[key] = value
        restored["observables"] = converted
    return restored


def _cmd_export(args) -> int:
    """export：重写 html / json / csv 报告产物。"""
    try:
        exp = store.get(args.root, args.id)
    except ValueError as exc:
        return _fail(args, str(exc), 2)
    except KeyError as exc:
        return _fail(args, _exc_message(exc), 2)
    except OSError as exc:
        return _fail(args, f"读取实验失败：{exc}", 1)

    try:
        report = _import_core("qllab.report")
    except _MissingDependency as exc:
        return _fail(args, str(exc), 1)

    formats = ["html", "json", "csv"] if args.format == "all" else [args.format]
    request = exp.get("request") or {}
    result = _restore_observable_arrays(exp.get("result") or {})
    validation = exp.get("validation") or {}
    meta = exp.get("meta") or {}
    exp_dir = exp.get("dir")
    figures = []
    for name, path in (exp.get("figures") or {}).items():
        try:
            with open(path, "rb") as fh:
                png = fh.read()
        except OSError:
            continue
        figures.append({"name": name, "png": png, "caption": ""})

    paths = {}
    try:
        bundle = None
        if "html" in formats or "json" in formats:
            bundle = report.bundle(request, result, validation, figures, meta)
        if "html" in formats:
            target = os.path.join(exp_dir, "report.html")
            report.write_html(target, bundle)
            paths["html"] = target
        if "json" in formats:
            target = os.path.join(exp_dir, "report.json")
            report.write_json(target, bundle)
            paths["json"] = target
        if "csv" in formats:
            target = os.path.join(exp_dir, "data.csv")
            report.write_csv(target, result)
            paths["csv"] = target
    except Exception as exc:
        return _fail(args, f"导出失败：{type(exc).__name__}: {exc}", 1)

    if args.json:
        _print_json({"ok": True, "id": args.id, "dir": exp_dir,
                     "formats": formats, "paths": paths})
        return 0
    print(f"【导出完成】实验 {args.id}")
    for fmt in formats:
        print(f"  {fmt}：{_short_path(paths.get(fmt, ''))}")
    return 0


# ---------------------------------------------------------------- selfcheck
def _observable_peak(result, name):
    """观察量时间序列的绝对值峰值；字段缺失返回 None。"""
    observables = (result or {}).get("observables") or {}
    if not isinstance(observables, dict):
        return None
    seq = observables.get(name)
    if seq is None:
        return None
    try:
        values = [abs(float(v)) for v in seq]
    except (TypeError, ValueError):
        try:
            return abs(float(seq))
        except (TypeError, ValueError):
            return None
    return max(values) if values else None


def _scalar_by_keywords(result, keywords):
    """在 scalars / metrics 中按关键字找第一个数值项，返回 (key, value)。"""
    for block_name in ("scalars", "metrics"):
        block = (result or {}).get(block_name) or {}
        if not isinstance(block, dict):
            continue
        for key, value in block.items():
            low = str(key).lower()
            if not any(word in low for word in keywords):
                continue
            if isinstance(value, bool):
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                return str(key), number
    return None, None


def _fmt(value) -> str:
    if value is None:
        return "未找到"
    try:
        return util.fmt_float(value, 10)
    except (TypeError, ValueError):
        return str(value)


def _validation_check(validation):
    overall = (validation or {}).get("overall")
    failed = (validation or {}).get("failed") or []
    ok = overall == "pass" and not failed
    detail = f"overall={overall}"
    if failed:
        detail += f"，失败项：{', '.join(str(x) for x in failed)}"
    return {"name": "物理一致性验证", "ok": ok, "detail": detail}


def _check_spin(result, validation):
    checks = []
    params = (result or {}).get("params") or {}
    omega = None
    try:
        omega = math.sqrt(sum(float(params[k]) ** 2
                              for k in ("Omega_x", "Omega_y", "Omega_z")))
    except (KeyError, TypeError, ValueError):
        omega = None
    target = math.sqrt(5.0)
    checks.append({
        "name": "|Ω| = √5",
        "ok": omega is not None and abs(omega - target) <= 1e-9,
        "detail": f"|Ω|={_fmt(omega)}，理论 {_fmt(target)}",
    })

    peak = _observable_peak(result, "pop_1")
    checks.append({
        "name": "pop_1 峰值 = 4/5",
        "ok": peak is not None and abs(peak - 0.8) <= 5e-3,
        "detail": f"观测峰值={_fmt(peak)}，理论 0.8",
    })

    analytic = (result or {}).get("analytic") or {}
    values = analytic.get("values") if isinstance(analytic, dict) else None
    if isinstance(values, dict):
        for key, value in values.items():
            low = str(key).lower()
            if not any(word in low for word in ("pop_1", "flip", "transition")):
                continue
            if isinstance(value, bool):
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            checks.append({
                "name": "解析 pop_1 峰值 = 0.8",
                "ok": abs(abs(number) - 0.8) <= 5e-3,
                "detail": f"{key}={_fmt(abs(number))}，理论 0.8",
            })
            break

    checks.append(_validation_check(validation))
    return checks


def _check_oscillator(result, validation):
    checks = []
    eigen = ((result or {}).get("eigen") or {}).get("values")
    levels = None
    if eigen is not None:
        try:
            levels = sorted(float(v) for v in eigen)
        except (TypeError, ValueError):
            levels = None
    if levels:
        ok = len(levels) >= 3 and all(
            abs(levels[i] - (i + 0.5)) <= 1e-8 for i in range(3))
        detail = "，".join(f"E{i}={_fmt(levels[i])}"
                          for i in range(min(3, len(levels))))
        checks.append({"name": "能谱 E_n = n+1/2", "ok": ok, "detail": detail})
    else:
        checks.append({"name": "能谱 E_n = n+1/2", "ok": False,
                       "detail": "未找到 eigen.values"})

    amplitude = _observable_peak(result, "expect_x")
    target = 1.0 / math.sqrt(2.0)
    checks.append({
        "name": "叠加态 <x> 振幅 ≈ 1/√2",
        "ok": amplitude is not None and abs(amplitude - target) <= 1e-3,
        "detail": f"max|<x>|={_fmt(amplitude)}，理论 {_fmt(target)}",
    })
    checks.append(_validation_check(validation))
    return checks


def _check_perturbation(result, validation):
    checks = []
    exact_key, exact = _scalar_by_keywords(result, ("e0_exact", "exact"))
    pt2_key, pt2 = _scalar_by_keywords(result, ("e0_pt2", "pt2", "second"))
    rel_key, rel = _scalar_by_keywords(result, ("rel_err", "relerr", "rel"))

    target_exact = -0.5 * math.sqrt(1.0 + 0.1 ** 2)
    target_pt2 = -0.5 - 0.1 ** 2 / (4.0 * 1.0)
    target_rel = abs((target_pt2 - target_exact) / target_exact)

    checks.append({
        "name": "E0_exact ≈ -0.50249378",
        "ok": exact is not None and abs(exact - target_exact) <= 1e-9,
        "detail": f"{exact_key or 'E0_exact'}={_fmt(exact)}，"
                  f"理论 {_fmt(target_exact)}",
    })
    checks.append({
        "name": "E0_pt2 = -0.5025",
        "ok": pt2 is not None and abs(pt2 - target_pt2) <= 1e-9,
        "detail": f"{pt2_key or 'E0_pt2'}={_fmt(pt2)}，理论 {_fmt(target_pt2)}",
    })
    checks.append({
        "name": "rel_err ≈ 1.2376e-05",
        "ok": rel is not None and abs(rel - target_rel) <= 1e-9,
        "detail": f"{rel_key or 'rel_err'}={_fmt(rel)}，理论 {_fmt(target_rel)}",
    })
    checks.append(_validation_check(validation))
    return checks


def _selfcheck_cases():
    """三条固定基准：自旋 / 谐振子 / 非简并微扰。"""
    return [
        ("自旋 1/2（Ωx=2, Ωz=1）",
         {"model": "spin_half",
          "params": {"Omega_x": 2.0, "Omega_z": 1.0},
          "initial_state": "up",
          "time": {"t_start": 0.0, "t_end": 2.0 * math.pi / math.sqrt(5.0),
                   "n_steps": 401},
          "question": "自检：自旋 1/2 拉比振荡（Ωx=2, Ωz=1）"},
         _check_spin),
        ("谐振子（ω=1，|0⟩+|1⟩ 叠加）",
         {"model": "harmonic_oscillator",
          "params": {"omega": 1.0, "n_levels": 40},
          "initial_state": {"kind": "superposition"},
          "time": {"t_start": 0.0, "t_end": 2.0 * math.pi, "n_steps": 401},
          "question": "自检：谐振子叠加态振荡（ω=1）"},
         _check_oscillator),
        ("微扰（Δ=1, λ=0.1）",
         {"model": "perturbation",
          "params": {"delta": 1.0, "lambda": 0.1},
          "sweep": {"param": "lambda", "values": [0.1]},
          "question": "自检：非简并微扰二阶能量（Δ=1, λ=0.1）"},
         _check_perturbation),
    ]


def _temp_dir(prefix: str) -> str:
    """创建临时目录。

    注意：Windows 沙箱下 ``tempfile.mkdtemp`` 以 0o700 建目录，受限令牌可能
    无法写入其中；因此优先用 ``os.makedirs`` 在工作目录下建普通权限目录。
    """
    name = f"{prefix}{secrets.token_hex(4)}"
    path = os.path.join(os.getcwd(), name)
    try:
        os.makedirs(path, exist_ok=False)
        return path
    except OSError:
        return tempfile.mkdtemp(prefix=prefix)


def _cmd_selfcheck(args) -> int:
    """selfcheck：跑三条基准并断言关键数值，全部 PASS 才返回 0。"""
    root = args.root
    tmp_root = None
    if root is None:  # 默认用临时目录，避免污染当前实验库
        tmp_root = _temp_dir("qllab-selfcheck-")
        root = tmp_root

    if not args.json:
        print(f"【QuantumLab 自检】共 {len(_selfcheck_cases())} 条基准")
        print(f"临时实验库：{_short_path(root)}" if tmp_root else
              f"实验库：{_short_path(root)}")

    cases_out = []
    all_ok = True
    dependency_error = None
    try:
        for index, (label, raw, checker) in enumerate(_selfcheck_cases(), 1):
            case = {"case": label, "checks": []}
            if not args.json:
                print("")
                print(f"【自检 {index}/3】{label}")
            try:
                req = spec.normalize_request(raw)
            except spec.SpecError as exc:
                case["checks"].append({"name": "请求规范化", "ok": False,
                                       "detail": str(exc)})
            else:
                try:
                    summary, result, validation, _figures = _execute_pipeline(
                        root, req)
                except _MissingDependency as exc:
                    dependency_error = exc
                    case["checks"].append({"name": "求解与落盘", "ok": False,
                                           "detail": str(exc)})
                except Exception as exc:
                    case["checks"].append({"name": "求解与落盘", "ok": False,
                                           "detail": f"{type(exc).__name__}: {exc}"})
                else:
                    case["experiment_id"] = summary.get("id")
                    case["checks"].extend(checker(result, validation))
            case["ok"] = all(item.get("ok") for item in case["checks"])
            all_ok = all_ok and case["ok"]
            cases_out.append(case)
            if not args.json:
                for item in case["checks"]:
                    mark = "PASS" if item.get("ok") else "FAIL"
                    print(f"  [{mark}] {item.get('name')}：{item.get('detail')}")
    finally:
        if tmp_root:
            shutil.rmtree(tmp_root, ignore_errors=True)

    total = sum(len(case["checks"]) for case in cases_out)
    passed = sum(1 for case in cases_out for item in case["checks"]
                 if item.get("ok"))
    payload = {"ok": bool(all_ok), "passed": passed, "total": total,
               "cases": cases_out}
    if dependency_error is not None:
        payload["error"] = str(dependency_error)

    if args.json:
        _print_json(payload)
    else:
        print("")
        print(f"【自检汇总】通过 {passed}/{total}；结论："
              f"{'全部通过' if all_ok else '存在失败项'}")
    if dependency_error is not None and not all_ok:
        return 1
    return 0 if all_ok else 1


# ---------------------------------------------------------------- 参数解析
def _add_common(parser):
    parser.add_argument("--json", action="store_true",
                        help="输出机器可读 JSON 摘要")
    parser.add_argument("--out-root", "--root", dest="root", default=None,
                        metavar="DIR",
                        help="实验库根目录（默认 cwd/quantumlab-experiments）")


def _add_input_options(parser):
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--request", metavar="FILE", help="请求 JSON 文件路径")
    group.add_argument("--spec", metavar="JSON", help="请求 JSON 字符串")
    group.add_argument("--stdin", action="store_true", help="从标准输入读取请求 JSON")


def build_parser():
    parser = argparse.ArgumentParser(
        prog="quantum_lab.py",
        description="QuantumLab：受控量子力学数值实验（自旋 1/2 / 谐振子 / 非简并微扰）",
        epilog="退出码：0 成功；1 执行失败或验证未通过；2 参数/请求非法。",
    )
    sub = parser.add_subparsers(dest="command", metavar="子命令")

    run_parser = sub.add_parser("run", help="执行一次实验并落盘")
    _add_common(run_parser)
    _add_input_options(run_parser)
    run_parser.set_defaults(func=_cmd_run)

    sweep_parser = sub.add_parser("sweep", help="带参数扫描执行一次实验")
    _add_common(sweep_parser)
    _add_input_options(sweep_parser)
    sweep_parser.add_argument("--param", metavar="NAME",
                              help="扫描参数名，如 lambda / Omega_z / n_levels")
    sweep_parser.add_argument("--values", metavar="LIST",
                              help="逗号分隔扫描点，如 0,0.5,1")
    sweep_parser.add_argument("--start", type=float, metavar="FLOAT")
    sweep_parser.add_argument("--stop", type=float, metavar="FLOAT")
    sweep_parser.add_argument("--num", type=int, metavar="INT")
    sweep_parser.set_defaults(func=_cmd_sweep)

    list_parser = sub.add_parser("list", help="列出历史实验")
    _add_common(list_parser)
    list_parser.add_argument("--model", metavar="NAME", help="按模型过滤")
    list_parser.add_argument("--text", metavar="TEXT", help="按原始问题子串过滤")
    list_parser.add_argument("--limit", type=int, metavar="N", help="最多返回 N 条")
    list_parser.set_defaults(func=_cmd_list)

    show_parser = sub.add_parser("show", help="查看实验详情")
    _add_common(show_parser)
    show_parser.add_argument("id", metavar="实验ID")
    show_parser.set_defaults(func=_cmd_show)

    rerun_parser = sub.add_parser("rerun", help="同参数重跑，生成新实验")
    _add_common(rerun_parser)
    rerun_parser.add_argument("id", metavar="实验ID")
    rerun_parser.set_defaults(func=_cmd_rerun)

    derive_parser = sub.add_parser("derive", help="改参数派生新实验（父实验保留）")
    _add_common(derive_parser)
    derive_parser.add_argument("id", metavar="实验ID")
    derive_parser.add_argument("--set", dest="set", action="append", default=[],
                               metavar="PATH=VALUE",
                               help="白名单字段赋值，如 params.Omega_z=0（可多次）")
    derive_parser.set_defaults(func=_cmd_derive)

    compare_parser = sub.add_parser("compare", help="对比多个实验")
    _add_common(compare_parser)
    compare_parser.add_argument("ids", nargs="+", metavar="实验ID")
    compare_parser.set_defaults(func=_cmd_compare)

    export_parser = sub.add_parser("export", help="重写报告产物")
    _add_common(export_parser)
    export_parser.add_argument("id", metavar="实验ID")
    export_parser.add_argument("--format", choices=("html", "json", "csv", "all"),
                               default="html", help="导出格式，默认 html")
    export_parser.set_defaults(func=_cmd_export)

    selfcheck_parser = sub.add_parser("selfcheck", help="运行三条基准自检")
    _add_common(selfcheck_parser)
    selfcheck_parser.set_defaults(func=_cmd_selfcheck)

    return parser


def _ensure_utf8_stdout():
    """把管道/重定向下的 stdout/stderr 切到 UTF-8。

    Windows 在非控制台输出时默认用 GBK，打印 ``|↑⟩`` 等字符会抛
    ``UnicodeEncodeError``；改成 UTF-8（无法编码时替换）更稳健。
    """
    for stream in (sys.stdout, sys.stderr):
        if stream is None:
            continue
        encoding = (getattr(stream, "encoding", "") or "").lower()
        if encoding in ("utf-8", "utf8"):
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass


def main(argv=None) -> int:
    """命令行入口：总是返回整数退出码（argparse 的 SystemExit 也归一化）。"""
    _ensure_utf8_stdout()
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        code = exc.code
        return int(code) if isinstance(code, int) else 2
    if not getattr(args, "func", None):
        parser.print_help()
        return 2
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
