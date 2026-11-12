# -*- coding: utf-8 -*-
"""报告导出：汇总数据（bundle）、JSON/CSV、单文件 HTML。

对外接口（见 docs/CONTRACT.md 第 7 节）：

``bundle(req, result, validation, figures, meta) -> dict``
    把一次实验的请求、结果、验证、图与元信息汇总成可直接喂给
    :func:`write_html` 的报告数据结构（含关键数值、检查表、结论边界）。

``write_json(path, payload)``
    经 :func:`util.to_jsonable` 写出 JSON（UTF-8、缩进 2、中文不转义）。

``write_csv(path, result)``
    时间序列 / 参数扫描数据表；UTF-8-SIG，Excel 直接打开不乱码。

``write_html(path, bundle)``
    独立单文件报告：PNG 以 ``data:image/png;base64,`` 内嵌，无外部依赖，
    打印友好（A4、状态色块打印后仍可辨），含目录与 9 个章节。

设计原则：**只引用本次真实数据**。执行失败时报告如实说明“无结果”，不描述未发生的计算。
"""
from __future__ import annotations

import base64
import csv
import html as _html
import json
import math
import numbers
import os
from typing import Any

try:  # spec 由 Lead 维护；导入失败不应连带拖垮报告模块（缺省值见下）
    from . import spec as _spec
except Exception:  # pragma: no cover - 仅在 spec 暂时不可用时触发
    _spec = None

from . import util

__all__ = ["bundle", "write_json", "write_csv", "write_html", "SCHEMA"]

SCHEMA = "quantumlab/report/1"

# spec 不可用时的参数顺序兜底（与 docs/CONTRACT.md 第 1 节一致）。
_FALLBACK_MODEL_PARAMS = {
    "spin_half": ("Omega_x", "Omega_y", "Omega_z"),
    "harmonic_oscillator": ("omega", "n_levels"),
    "perturbation": ("delta", "lambda"),
}


def _model_param_names(model: str, params: dict) -> tuple:
    if _spec is not None:
        try:
            names = _spec.MODEL_PARAMS[model]
            if names:
                return tuple(names)
        except (AttributeError, KeyError, TypeError):
            pass
    return tuple(_FALLBACK_MODEL_PARAMS.get(model, tuple(params.keys())))

# ---------------------------------------------------------------------------
# 模型说明（公式与假设来自 references/models.md，报告只做展示，不参与计算）
# ---------------------------------------------------------------------------
MODEL_META = {
    "spin_half": {
        "name": "自旋 1/2 二能级系统",
        "basis": "|↑⟩=(1,0)ᵀ、|↓⟩=(0,1)ᵀ（σz 本征态）",
        "hamiltonian": "H = (1/2)(Ωx σx + Ωy σy + Ωz σz)",
        "equations": [
            "H = (1/2)(Ωx σx + Ωy σy + Ωz σz)，ħ = 1",
            "|Ω| = √(Ωx² + Ωy² + Ωz²)，本征能量 E± = ±|Ω|/2",
            "演化算符 U(t) = V exp(−iΛ(t−t₀)) V†（H = V Λ V†）",
            "Bloch 进动：n(t) = n̂ cos(|Ω|t) + (Ω̂×n̂) sin(|Ω|t) + Ω̂(Ω̂·n̂)(1−cos(|Ω|t))",
            "从 |↑⟩ 出发：P↓(t) = (Ω⊥²/|Ω|²)·sin²(|Ω|t/2)，Ω⊥² = Ωx² + Ωy²",
        ],
        "assumptions": [
            "理想二能级自旋，ħ = 1，能量与角频率同量纲，时间以 Ω⁻¹ 为单位；",
            "哈密顿量厄米且在本实验时间窗内不显含时间（Ω 恒定）；",
            "演化严格幺正：无退相干、无耗散、无测量反作用；",
            "解析参照为 Bloch 进动闭式解，仅用于交叉校验数值演化。",
        ],
        "limits": [
            "结论仅对本次给定的 Ω 分量、初态与时间窗成立；",
            "未验证其它驱动方向/强度、含时驱动、开放系统与多比特纠缠。",
        ],
    },
    "harmonic_oscillator": {
        "name": "量子谐振子（截断 Fock 空间）",
        "basis": "Fock 基 |0⟩…|N−1⟩，N = n_levels",
        "hamiltonian": "H = ω(a†a + 1/2)",
        "equations": [
            "H = ω(a†a + 1/2)，E_n = ω(n + 1/2)，n = 0…N−1",
            "a[n−1, n] = √n；x = (a + a†)/√2，p = i(a† − a)/√2",
            "相干态展开 c_n = e^{−|α|²/2} α^n/√(n!)，截断后重新归一化并记录泄漏",
            "⟨x⟩(t) = √2 Re[A₀ e^{−iωt}]，⟨p⟩(t) = √2 Im[A₀ e^{−iωt}]，A₀ = Σ_n c_n* c_{n+1} √(n+1)",
        ],
        "assumptions": [
            "一维谐振子，ħ = 1，时间以 ω⁻¹ 为单位；",
            "Fock 空间截断到 N 个能级：有限维近似，不是无穷维精确解；",
            "不引入非谐项、环境耦合或驱动项；",
            "边界能级占据小只说明当前初态/时间窗的截断泄漏小，不构成收敛证明。",
        ],
        "limits": [
            "结论依赖截断维数 N，增大 N 可能改变边界占据等量；",
            "未验证强驱动、非谐势、开放系统与多模耦合。",
        ],
    },
    "perturbation": {
        "name": "非简并微扰（二能级）",
        "basis": "σz 本征态 |0⟩=|↓⟩、|1⟩=|↑⟩",
        "hamiltonian": "H = (Δ/2)σz + (λ/2)σx",
        "equations": [
            "H₀ = (Δ/2)σz，V = (λ/2)σx，H = H₀ + V；E₀⁽⁰⁾ = −Δ/2",
            "一阶修正 E0_pt1 = −Δ/2 + ⟨0|V|0⟩ = −Δ/2（V 无对角元）",
            "二阶修正 E0_pt2 = −Δ/2 − λ²/(4Δ)",
            "精确基态 E0_exact = −½√(Δ² + λ²)，abs_err = |E0_pt2 − E0_exact|，rel_err = abs_err/|E0_exact|",
        ],
        "assumptions": [
            "非简并（Δ > 0），微扰展开以小参数 λ/Δ 为基础；",
            "只取到二阶，未包含三阶及以上修正；",
            "近似适用范围由采样点上的 rel_err ≤ tolerance.perturbation_rel 判定，只对当前 Δ 与采样 λ 成立；",
            "“近似偏离精确解”是物理预期，与“求解器错误”分开判断。",
        ],
        "limits": [
            "微扰近似结论不能外推到未采样的 λ；",
            "未处理简并/近简并情形，也未处理含时微扰。",
        ],
    },
}

PARAM_UNITS = {
    "Omega_x": "rad/time（ħ=1）",
    "Omega_y": "rad/time（ħ=1）",
    "Omega_z": "rad/time（ħ=1）",
    "omega": "rad/time（ħ=1）",
    "n_levels": "个（无量纲，2…400）",
    "delta": "能量（ħ=1 下与角频率同量纲）",
    "lambda": "能量（ħ=1 下与角频率同量纲）",
}

PARAM_LABELS = {
    "Omega_x": "Ωx（横向驱动）",
    "Omega_y": "Ωy（横向驱动）",
    "Omega_z": "Ωz（纵向/失谐）",
    "omega": "ω（角频率）",
    "n_levels": "N（Fock 截断能级数）",
    "delta": "Δ（未扰能级间隙）",
    "lambda": "λ（扰动强度）",
}

SCALAR_LABELS = {
    "Omega_eff": "等效拉比频率 |Ω|",
    "Omega_mag": "等效拉比频率 |Ω|",
    "omega": "谐振子角频率 ω",
    "n_levels": "Fock 截断能级数 N",
    "E_plus": "本征能量 E₊",
    "E_minus": "本征能量 E₋",
    "E_plus_minus": "本征能量 E±",
    "max_flip_prob_theory": "最大跃迁概率（理论）Ω⊥²/|Ω|²",
    "max_flip_prob": "最大跃迁概率",
    "pop_1_max_theory": "pop_1 理论峰值",
    "E0_exact": "基态能量（精确解）",
    "E0_pt1": "一阶微扰能量 E0⁽¹⁾",
    "E0_pt2": "二阶微扰能量 E0⁽²⁾",
    "E0_0": "未扰基态能量 E0⁽⁰⁾",
    "spectrum": "能谱 E_n",
    "lambda_valid_max": "微扰近似适用上限 λ（本次采样）",
    "boundary_pop_max": "截断边界能级最大占据",
    "coherent_truncation_error": "相干态截断泄漏 1−Σ|c_n|²",
    "truncation_error": "截断泄漏 1−Σ|c_n|²",
    "energy_ground": "基态能量",
    "norm_max_dev": "态范数最大偏差",
}

METRIC_LABELS = {
    "analytic_abs_dev": "与解析解最大绝对偏差",
    "max_abs_dev": "最大绝对偏差",
    "pop_1_abs_dev": "pop_1 与解析解最大绝对偏差",
    "expect_x_abs_dev": "⟨x⟩ 与解析解最大绝对偏差",
    "spectrum_abs_dev": "能谱与解析值最大偏差",
    "abs_err": "二阶近似绝对误差",
    "rel_err": "二阶近似相对误差",
    "norm_max_dev": "态范数最大偏差",
    "norm_dev": "态范数最大偏差",
    "probability_sum_dev": "占据概率和与 1 的最大偏差",
    "energy_drift": "能量相对漂移",
    "boundary_pop": "边界能级最大占据",
    "truncation_leak": "截断泄漏",
    "hermiticity": "max|H − H†|",
    "eigen_residual": "本征方程残差",
    "orthonormality": "正交归一性偏差",
}

MODEL_ORDER = ("spin_half", "harmonic_oscillator", "perturbation")

# CSV 列定义（见 references/report-format.md）
CSV_COLUMNS = {
    "spin_half": ["t", "pop_0", "pop_1", "expect_x", "expect_y", "expect_z",
                  "energy", "pop_1_theory"],
    "harmonic_oscillator": ["t", "pop_0", "pop_1", "pop_2", "pop_3", "expect_x",
                            "expect_p", "expect_n", "boundary_pop", "expect_x_theory"],
    "perturbation": ["lambda", "E0_pt1", "E0_pt2", "E0_exact", "abs_err",
                     "rel_err", "within_threshold"],
}

_THEORY_ALIASES = {
    "pop_1_theory": ("pop_1_theory", "pop_1", "P_down", "P_down_theory", "p1_theory"),
    "expect_x_theory": ("expect_x_theory", "expect_x", "x_theory", "expect_x_analytic"),
}

_STATUS_TEXT = {"pass": "通过", "fail": "未通过", "skipped": "跳过"}
_STATUS_CLASS = {"pass": "pass", "fail": "fail", "skipped": "skip"}

SECTIONS = [
    ("sec-1", "研究问题与实验目的"),
    ("sec-2", "模型公式与假设"),
    ("sec-3", "参数、初态与单位"),
    ("sec-4", "数值方法"),
    ("sec-5", "结果图像与关键数据"),
    ("sec-6", "物理一致性验证"),
    ("sec-7", "结果解释与适用限制"),
    ("sec-8", "复现配置"),
    ("sec-9", "附录：生成信息与文件清单"),
]


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------
def _e(text: Any) -> str:
    return _html.escape(str(text if text is not None else ""), quote=True)


def _is_num(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _value_text(value: Any) -> str:
    """把标量/列表格式化成报告用字符串；不编造，只转写。"""
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "是" if value else "否"
    if _is_num(value):
        if not math.isfinite(float(value)):
            return util.fmt_float(value, 6)
        return util.fmt_float(value, 6)
    if isinstance(value, (list, tuple)):
        items = list(value)
        if not items:
            return "（空）"
        if len(items) <= 6:
            return "、".join(_value_text(v) for v in items)
        return f"{len(items)} 项（{_value_text(items[0])} … {_value_text(items[-1])}）"
    if isinstance(value, complex):
        return util.complex_str(value, 6)
    return str(value)


def _to_list(value: Any) -> list:
    """numpy / tuple / None -> list（保留原值，不做数值转换）。"""
    if value is None:
        return []
    if isinstance(value, (str, bytes, dict)):
        return []
    if hasattr(value, "tolist"):
        try:
            value = value.tolist()
        except (ValueError, AttributeError):
            return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _finite_numbers(value: Any) -> list:
    out = []
    for item in _to_list(value):
        try:
            number = float(item)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            out.append(number)
    return out


def _relpath(path: str) -> str:
    try:
        return os.path.relpath(path)
    except (ValueError, OSError):  # pragma: no cover
        return str(path)


# ---------------------------------------------------------------------------
# bundle 的各个组成部分
# ---------------------------------------------------------------------------
def _status(validation: dict, executed: bool, error: str) -> dict:
    """三态总状态：执行失败 / 部分验证未通过 / 通过（外加“验证被跳过”兜底）。"""
    if not executed:
        return {
            "overall": "fail",
            "label": "执行失败（没有结果）",
            "class": "bad",
            "detail": error or "求解器未返回完整结果，报告中不提供数值解释。",
            "counts": {"pass": 0, "fail": 0, "skipped": 0},
        }
    validation = validation or {}
    checks = validation.get("checks") or []
    counts = validation.get("counts") or {}
    if not counts and checks:
        counts = {"pass": 0, "fail": 0, "skipped": 0}
        for check in checks:
            key = str((check or {}).get("status") or "skipped")
            counts[key if key in counts else "skipped"] = counts.get(
                key if key in counts else "skipped", 0) + 1
    counts = {"pass": int(counts.get("pass", 0) or 0),
              "fail": int(counts.get("fail", 0) or 0),
              "skipped": int(counts.get("skipped", 0) or 0)}
    overall = validation.get("overall")
    failures = [str(c.get("name") or c.get("id") or "?")
                for c in checks if str((c or {}).get("status")) == "fail"]
    if counts["fail"] > 0 or overall in ("fail", "partial"):
        detail = (f"计算完成，但有 {counts['fail']} 项验证未通过"
                  + (f"：{'、'.join(failures)}。" if failures else "。"))
        return {"overall": "partial", "label": "部分验证未通过",
                "class": "warn", "detail": detail, "counts": counts}
    if overall == "pass" and (counts["pass"] > 0 or counts["skipped"] > 0):
        detail = (f"共执行 {counts['pass'] + counts['fail']} 项检查，"
                  f"通过的 {counts['pass']} 项均满足容差"
                  + (f"；{counts['skipped']} 项因不适用被跳过。" if counts["skipped"] else "。"))
        return {"overall": "pass", "label": "通过", "class": "ok",
                "detail": detail, "counts": counts}
    return {
        "overall": "skipped",
        "label": "验证被跳过（无适用检查）",
        "class": "skip",
        "detail": "没有可判定的检查项，无法给出通过结论。",
        "counts": counts,
    }


def _param_rows(req: dict) -> list:
    model = req.get("model") or ""
    params = req.get("params") or {}
    provenance = req.get("provenance") or {}
    names = _model_param_names(model, params)
    rows = []
    for name in names:
        if name not in params:
            continue
        value = params[name]
        if name == "n_levels":
            value_text = str(int(value)) if _is_num(value) else _value_text(value)
        else:
            value_text = _value_text(value)
        source = provenance.get(name, "given")
        rows.append({
            "name": name,
            "label": PARAM_LABELS.get(name, name),
            "value": value_text,
            "unit": PARAM_UNITS.get(name, "—"),
            "source": source,
            "source_text": "默认值" if source == "default" else "用户给定",
        })
    return rows


def _initial_state(req: dict, result: dict) -> dict:
    model = req.get("model") or ""
    state = req.get("initial_state")
    if model == "perturbation":
        return {"kind": "none", "kind_label": "无初态（定态能谱计算）",
                "text": "微扰模型只做基态能量对比，不定义时间演化初态。",
                "details": [], "raw": util.to_jsonable(state)}
    if not isinstance(state, dict):
        return {"kind": "unknown", "kind_label": "未提供",
                "text": "请求中未给出初态，按规范化结果执行。",
                "details": [], "raw": util.to_jsonable(state)}
    kind = state.get("kind")
    if model == "spin_half":
        labels = {"up": "|↑⟩（σz 本征态）", "down": "|↓⟩",
                  "plus_x": "|+x⟩=(|↑⟩+|↓⟩)/√2", "plus_y": "|+y⟩=(|↑⟩+i|↓⟩)/√2",
                  "custom": "自定义复向量"}
        details = []
        if kind == "custom" and state.get("vector"):
            for index, pair in enumerate(state["vector"]):
                details.append(f"分量 {index}：{util.complex_str(complex(pair[0], pair[1]), 4)}")
            if state.get("renormalized"):
                details.append("注意：输入未归一化，已自动重新归一化。")
        return {"kind": kind, "kind_label": labels.get(kind, str(kind)),
                "text": f"初态：{labels.get(kind, kind)}",
                "details": details, "raw": util.to_jsonable(state)}
    # harmonic_oscillator
    if kind == "fock":
        return {"kind": kind, "kind_label": f"Fock 本征态 |{state.get('n', 0)}⟩",
                "text": f"初态：Fock 本征态 |{state.get('n', 0)}⟩",
                "details": [], "raw": util.to_jsonable(state)}
    if kind == "coherent":
        alpha = state.get("alpha") or [0.0, 0.0]
        z = complex(alpha[0], alpha[1])
        return {"kind": kind, "kind_label": f"相干态 |α⟩，α = {util.complex_str(z, 4)}",
                "text": f"初态：相干态 |α⟩，α = {util.complex_str(z, 4)}",
                "details": [], "raw": util.to_jsonable(state)}
    components = state.get("components") or []
    parts = []
    for comp in components:
        amp = comp.get("amplitude") or [1.0, 0.0]
        parts.append(f"({util.complex_str(complex(amp[0], amp[1]), 4)})|{comp.get('n', 0)}⟩")
    details = ["注意：输入未按模方归一，已自动重新归一化。"] if state.get("renormalized") else []
    return {"kind": kind, "kind_label": "Fock 空间叠加态",
            "text": "初态：" + (" + ".join(parts) if parts else "叠加态（无分量记录）"),
            "details": details, "raw": util.to_jsonable(state)}


def _time_block(req: dict) -> dict:
    time = req.get("time")
    if not isinstance(time, dict):
        return {"present": False, "text": "本模型的请求没有时间演化（静态能谱计算）。"}
    try:
        t0, t1 = float(time["t_start"]), float(time["t_end"])
        steps = int(time["n_steps"])
    except (KeyError, TypeError, ValueError):
        return {"present": False, "text": "时间配置不完整。"}
    span = t1 - t0
    dt = span / (steps - 1) if steps > 1 else 0.0
    return {
        "present": True,
        "t_start": util.fmt_float(t0, 6),
        "t_end": util.fmt_float(t1, 6),
        "n_steps": steps,
        "dt": util.fmt_float(dt, 6),
        "text": (f"时间窗 t ∈ [{util.fmt_float(t0, 6)}, {util.fmt_float(t1, 6)}]，"
                 f"均匀采样 {steps} 点，步长 Δt = {util.fmt_float(dt, 6)}。"),
    }


def _method_block(req: dict, result: dict) -> dict:
    model = req.get("model") or ""
    solver = req.get("solver") or {}
    steps = []
    if model == "perturbation":
        steps.append("在 λ 扫描的每个采样点上构造 H(λ) = (Δ/2)σz + (λ/2)σx（2×2 厄米矩阵）。")
        steps.append("用 np.linalg.eigh 做精确对角化，得到 E0_exact = −½√(Δ²+λ²) 的数值对照。")
        steps.append("由微扰展开计算 E0_pt1、E0_pt2，并与精确解比较得到 abs_err、rel_err。")
        steps.append("按 rel_err ≤ tolerance.perturbation_rel 判定每个采样点的近似适用范围。")
    else:
        steps.append(f"把哈密顿量写成矩阵，基矢：{MODEL_META.get(model, {}).get('basis', '—')}。")
        steps.append("用 np.linalg.eigh 对厄米矩阵做谱分解 H = V Λ V†。")
        steps.append("对每个采样时刻构造 U(t) = V exp(−iΛ(t−t₀)) V† 并作用于初态。")
        steps.append("在时间网格上计算各观察量，并与解析参照逐点比较得到误差指标。")
    solver_note = solver.get("note") or "H 厄米 → np.linalg.eigh 谱分解，U(t)=V exp(−iΛt) V†"
    tolerance = req.get("tolerance") or {}
    return {
        "steps": steps,
        "solver_note": solver_note,
        "tolerance": [{"name": name, "value_text": util.fmt_float(value, 4)}
                      for name, value in sorted(tolerance.items())],
        "backend": "numpy（本地，无外部求解器）",
    }


def _figure_records(figures: list) -> list:
    records = []
    for index, figure in enumerate(figures or []):
        if not isinstance(figure, dict):
            continue
        png = figure.get("png")
        if isinstance(png, memoryview):
            png = png.tobytes()
        if not png:
            continue
        try:
            blob = base64.b64encode(bytes(png)).decode("ascii")
        except (TypeError, ValueError):
            continue
        records.append({
            "index": index + 1,
            "name": util.safe_stem(figure.get("name") or f"fig_{index + 1}",
                                   fallback=f"fig_{index + 1}"),
            "caption": str(figure.get("caption") or ""),
            "png_b64": blob,
            "bytes": len(png),
        })
    return records


def _key_rows(result: dict) -> list:
    """关键数值表：scalars + metrics + 由时间序列直接算出的峰值（全部来自本次结果）。"""
    rows, seen = [], set()

    def add(label: str, value: Any, note: str):
        if label in seen:
            return
        seen.add(label)
        rows.append({"label": label, "value": _value_text(value), "note": note})

    for name, value in (result.get("scalars") or {}).items():
        add(SCALAR_LABELS.get(name, name), value, f"scalars.{name}")
    for name, value in (result.get("metrics") or {}).items():
        if _is_num(value) or isinstance(value, str):
            add(METRIC_LABELS.get(name, name), value, f"metrics.{name}")

    observables = result.get("observables") or {}
    for name in ("pop_0", "pop_1", "pop_2", "pop_3", "boundary_pop", "pop_n"):
        numbers = _finite_numbers(observables.get(name))
        if numbers:
            add(f"{name} 最大值（时间序列）", max(numbers), "由本实验时间序列计算")
    for name in ("expect_x", "expect_p"):
        numbers = _finite_numbers(observables.get(name))
        if numbers:
            add(f"{name} 振幅 (max−min)/2", (max(numbers) - min(numbers)) / 2.0,
                "由本实验时间序列计算")
    for name in ("expect_z", "expect_y"):
        numbers = _finite_numbers(observables.get(name))
        if numbers:
            add(f"{name} 取值范围", [min(numbers), max(numbers)],
                "由本实验时间序列计算")
    energy = _finite_numbers(observables.get("energy"))
    if energy:
        add("energy 取值范围", [min(energy), max(energy)], "由本实验时间序列计算")
    return rows


def _validation_block(validation: dict, status: dict) -> dict:
    validation = validation or {}
    checks = []
    for check in validation.get("checks") or []:
        if not isinstance(check, dict):
            continue
        state = str(check.get("status") or "skipped")
        metric = check.get("metric")
        if isinstance(metric, dict):
            metric_name = str(metric.get("name") or "—")
            metric_value = metric.get("value")
        elif metric is None:
            metric_name, metric_value = "—", None
        else:
            metric_name, metric_value = "—", metric
        tolerance = check.get("tolerance")
        checks.append({
            "id": str(check.get("id") or "—"),
            "name": str(check.get("name") or check.get("id") or "—"),
            "status": state,
            "status_label": _STATUS_TEXT.get(state, state),
            "status_class": _STATUS_CLASS.get(state, "skip"),
            "metric_name": metric_name,
            "metric_value": util.fmt_float(metric_value, 6) if _is_num(metric_value)
                            else _value_text(metric_value),
            "tolerance": util.fmt_float(tolerance, 4) if _is_num(tolerance) else "—",
            "detail": str(check.get("detail") or ""),
            "scope": str(check.get("scope") or ""),
        })
    limits = [str(item) for item in (validation.get("limits") or [])]
    return {
        "overall": status["overall"],
        "label": status["label"],
        "class": status["class"],
        "detail": status["detail"],
        "counts": status["counts"],
        "checks": checks,
        "limits": limits,
    }


def _metric_by_name(metrics: dict) -> str:
    for name in ("analytic_abs_dev", "max_abs_dev", "pop_1_abs_dev", "rel_err",
                 "abs_err", "spectrum_abs_dev", "norm_max_dev", "boundary_pop"):
        value = metrics.get(name)
        if _is_num(value):
            return f"{METRIC_LABELS.get(name, name)} = {util.fmt_float(value, 6)}"
    return ""


def _why_bullets(model: str, req: dict, result: dict) -> list:
    """“为什么”部分：只用本次结果里的真实数值。"""
    params = req.get("params") or {}
    scalars = result.get("scalars") or {}
    metrics = result.get("metrics") or {}
    observables = result.get("observables") or {}
    analytic = result.get("analytic") or {}
    out = []

    if model == "spin_half":
        try:
            omega_x = float(params.get("Omega_x", 0.0))
            omega_y = float(params.get("Omega_y", 0.0))
            omega_z = float(params.get("Omega_z", 0.0))
            total = math.sqrt(omega_x ** 2 + omega_y ** 2 + omega_z ** 2)
            perp2 = omega_x ** 2 + omega_y ** 2
        except (TypeError, ValueError):
            total, perp2 = 0.0, 0.0
        if total > 0:
            out.append(f"等效拉比频率 |Ω| = {util.fmt_float(total, 6)}，"
                       f"Bloch 矢量以该角频率绕 Ω 方向进动，故时间振荡的角频率是 |Ω| 而不是 Ω⊥。")
            if perp2 > 0:
                ratio = perp2 / (total ** 2)
                out.append(f"Ωz = {util.fmt_float(omega_z, 6)} 使本征态偏离 σz 基，"
                           f"最大跃迁概率被限制为 Ω⊥²/|Ω|² = {util.fmt_float(ratio, 6)}。")
            else:
                out.append("Ω⊥ = 0（只有纵向分量），σz 占据不变，只有整体相位在演化。")
        if analytic.get("name"):
            metric_text = _metric_by_name(metrics)
            if metric_text:
                out.append(f"数值演化与解析参照「{analytic.get('name')}」对比：{metric_text}。")
            else:
                out.append(f"本实验含解析参照「{analytic.get('name')}」，对比指标见验证表。")
        return out

    if model == "harmonic_oscillator":
        n_levels = params.get("n_levels")
        if _is_num(n_levels):
            out.append(f"截断到 N = {int(n_levels)} 个 Fock 能级，"
                       f"能级间隔为 ω = {util.fmt_float(params.get('omega', 0.0), 6)}。")
        boundary = scalars.get("boundary_pop_max")
        if not _is_num(boundary):
            numbers = _finite_numbers(observables.get("boundary_pop"))
            boundary = max(numbers) if numbers else None
        if _is_num(boundary):
            tolerance = (req.get("tolerance") or {}).get("truncation_pop")
            if _is_num(tolerance):
                verdict = ("在当前条件下边界泄漏可忽略" if float(boundary) <= float(tolerance)
                           else "当前边界泄漏超过容差，应增大 n_levels")
                out.append(f"边界能级最大占据 = {util.fmt_float(boundary, 6)}"
                           f"（容差 {util.fmt_float(tolerance, 4)}）：{verdict}；"
                           f"但这只是本次条件下的诊断，不构成收敛证明。")
            else:
                out.append(f"边界能级最大占据 = {util.fmt_float(boundary, 6)}；"
                           f"边界泄漏小不构成收敛证明。")
        amp = _finite_numbers(observables.get("expect_x"))
        if amp:
            out.append(f"⟨x⟩(t) 在本次时间窗内于 "
                       f"[{util.fmt_float(min(amp), 6)}, {util.fmt_float(max(amp), 6)}] 之间振荡，"
                       f"振幅 (max−min)/2 = {util.fmt_float((max(amp) - min(amp)) / 2.0, 6)}。"
                       f"相干态对应 √2|α|、叠加态由相对相位驱动，与本次初态设置一致。")
        metric_text = _metric_by_name(metrics)
        if metric_text:
            out.append(f"与解析参照的定量对比：{metric_text}（详见验证表）。")
        return out

    # perturbation
    delta = params.get("delta")
    lam = params.get("lambda")
    if _is_num(delta) and _is_num(lam):
        out.append(f"本次 Δ = {util.fmt_float(delta, 6)}，λ = {util.fmt_float(lam, 6)}；"
                   f"一阶修正为 0（V 无对角元），二阶修正为 −λ²/(4Δ)。")
    exact = scalars.get("E0_exact")
    pt2 = scalars.get("E0_pt2")
    if _is_num(exact) and _is_num(pt2):
        out.append(f"精确基态能量 E0_exact = {util.fmt_float(exact, 6)}，"
                   f"二阶近似 E0_pt2 = {util.fmt_float(pt2, 6)}。")
    err = scalars.get("rel_err")
    if not _is_num(err):
        err = metrics.get("rel_err")
    if _is_num(err):
        out.append(f"相对误差 rel_err = {util.fmt_float(err, 6)}："
                   f"误差随 λ/Δ 增大而增长，这是微扰近似偏离精确解的物理预期，"
                   f"不是求解器错误。")
    valid_max = scalars.get("lambda_valid_max")
    tolerance = (req.get("tolerance") or {}).get("perturbation_rel")
    if _is_num(valid_max):
        out.append(f"按容差 perturbation_rel = {util.fmt_float(tolerance, 4) if _is_num(tolerance) else '—'}，"
                   f"本次采样中近似可用到 λ ≤ {util.fmt_float(valid_max, 6)}；"
                   f"该范围只对当前 Δ 与采样点成立。")
    elif result.get("sweep"):
        sweep = result["sweep"]
        values = _finite_numbers(sweep.get("values"))
        if values:
            out.append(f"本次扫描 λ ∈ [{util.fmt_float(min(values), 6)}, "
                       f"{util.fmt_float(max(values), 6)}]，共 {len(values)} 个采样点；"
                       f"适用范围判定只对这些采样点成立。")
    return out


def _interpretation(model: str, req: dict, result: dict, status: dict,
                    key_rows: list) -> dict:
    executed = status["overall"] != "fail"
    if not executed:
        return {
            "what": ["本次实验没有产生完整数值结果，因此不描述物理结论。"],
            "why": ["请先根据状态与错误信息修正请求（参数范围、模型选择或数值条件）后重试。"],
            "limits": ["无结果即无结论；未运行的检查与实验一律不宣称。"],
        }
    what = [str(item) for item in ((result.get("summary") or {}).get("bullets") or [])]
    if not what:
        if key_rows:
            what.append(f"本次计算得到 {len(key_rows)} 项关键数值/指标，逐项列于第 5 节。")
        else:
            what.append("本次计算完成，但结果中没有可汇总的关键标量。")
    why = _why_bullets(model, req, result)
    if not why:
        why.append("结果解释所需的物理量未在本次输出中提供，故不对机制作推测。")
    limits = list(model_meta(model).get("limits", []))
    params_text = "、".join(f"{row['label']}={row['value']}" for row in
                            _param_rows(req)[:4])
    if params_text:
        limits.insert(0, f"结论仅对本次参数（{params_text}）与给定初态/时间窗成立。")
    limits.append("报告数值均为本地数值求解结果；未采集文献对照，也未做不确定性量化之外的推断。")
    return {"what": what, "why": why, "limits": limits}


def model_meta(model: str) -> dict:
    return MODEL_META.get(model, {
        "name": model or "未知模型",
        "basis": "—",
        "hamiltonian": "—",
        "equations": [],
        "assumptions": [],
        "limits": [],
    })


# ---------------------------------------------------------------------------
# bundle
# ---------------------------------------------------------------------------
def bundle(req: dict, result: dict, validation: dict, figures: list, meta: dict) -> dict:
    """汇总一次实验的报告数据（不写盘、不编造数值）。"""
    req = req if isinstance(req, dict) else {}
    result = result if isinstance(result, dict) else {}
    meta = meta if isinstance(meta, dict) else {}
    figures = figures if isinstance(figures, list) else []

    model = (result.get("model") or req.get("model") or meta.get("model") or "unknown")
    error = ""
    if isinstance(result.get("error"), str):
        error = result["error"]
    executed = bool(result.get("model")) and not error
    if not executed:
        result = {} if error else result  # 失败时不保留半成品数值供解释引用

    status = _status(validation, executed, error)
    meta_info = {
        "id": str(meta.get("id") or "（未入库）"),
        "created_at": str(meta.get("created_at") or ""),
        "model": model,
        "model_name": model_meta(model)["name"],
        "relation": meta.get("relation"),
        "parent_id": meta.get("parent_id"),
        "duration_ms": meta.get("duration_ms"),
        "solver": (req.get("solver") or {}).get("method") or meta.get("solver") or "—",
        "question": str(req.get("question") or meta.get("question") or ""),
    }
    time_block = _time_block(req)
    figure_records = _figure_records(figures)
    key_rows = _key_rows(result) if executed else []
    validation_block = _validation_block(validation, status)

    what_if = _interpretation(model, req, result, status, key_rows)
    sweep_table = _sweep_table(result, model) if executed else None
    series_table = _time_series_table(result, model) if executed else None

    repro_request = util.to_jsonable(req)
    try:
        repro_text = json.dumps(repro_request, ensure_ascii=False, indent=2)
    except (TypeError, ValueError):  # pragma: no cover
        repro_text = "{}"

    warnings = []
    for source in (req.get("warnings"), result.get("warnings")):
        for item in source or []:
            text = str(item)
            if text and text not in warnings:
                warnings.append(text)

    files = ["request.json", "result.json", "validation.json", "meta.json",
             "data.csv"]
    files += [f"figures/{record['name']}.png" for record in figure_records]
    files.append("report.html")

    return {
        "schema": SCHEMA,
        "generated_at": util.now_iso(),
        "status": status["overall"],
        "status_label": status["label"],
        "status_class": status["class"],
        "status_detail": status["detail"],
        "executed": executed,
        "error": error,
        "experiment": meta_info,
        "question": meta_info["question"],
        "purpose": _purpose(meta_info["question"], model, executed),
        "model": {
            "key": model,
            "name": model_meta(model)["name"],
            "basis": model_meta(model)["basis"],
            "hamiltonian": model_meta(model)["hamiltonian"],
            "equations": list(model_meta(model)["equations"]),
            "assumptions": list(model_meta(model)["assumptions"]),
        },
        "params": _param_rows(req),
        "initial_state": _initial_state(req, result),
        "time": time_block,
        "units": dict(req.get("units") or {}),
        "method": _method_block(req, result),
        "figures": figure_records,
        "key_values": key_rows,
        "series_table": series_table,
        "sweep_table": sweep_table,
        "validation": validation_block,
        "checks": validation_block["checks"],
        "interpretation": what_if,
        "limits": validation_block["limits"] + what_if["limits"],
        "reproduction": {
            "request": repro_request,
            "json_text": repro_text,
            "command": "python scripts/quantum_lab.py run --request request.json",
            "inline_command": "python scripts/quantum_lab.py run --spec '<请求 JSON>'",
            "files": files,
        },
        "warnings": warnings,
        "sections": [{"id": sid, "title": title} for sid, title in SECTIONS],
    }


def _purpose(question: str, model: str, executed: bool) -> list:
    out = []
    if question:
        out.append(f"用户原始问题：{question}")
    else:
        out.append("用户未提供原始问题文本；本实验按规范化请求执行基准计算。")
    out.append(f"实验目的：用本地谱分解求解 {model_meta(model)['name']}，"
               f"并按固定检查表验证数值与解析参照的一致性。")
    if not executed:
        out.append("注意：本次执行未产生完整结果，目的项只说明计划，不代表已完成。")
    return out


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------
def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "True" if value else "False"
    if _is_num(value):
        number = float(value)
        if math.isnan(number):
            return "nan"
        if math.isinf(number):
            return "inf" if number > 0 else "-inf"
        return f"{number:.12g}"
    if isinstance(value, complex):
        return f"{value.real:.12g}{value.imag:+.12g}j"
    if isinstance(value, (list, tuple)):
        return ";".join(_csv_value(item) for item in value)
    return str(value)


def _lookup_column(result: dict, column: str) -> list:
    """按列名在 observables / 解析值中查找数据列。"""
    observables = result.get("observables") or {}
    analytic_values = (result.get("analytic") or {}).get("values") or {}
    candidates = _THEORY_ALIASES.get(column, (column,))
    if column.endswith("_theory"):
        # 理论列优先取解析值，避免把数值曲线当成理论曲线。
        for name in candidates:
            if name in analytic_values:
                return _to_list(analytic_values[name])
        return _to_list(observables.get(column, []))
    if column in observables:
        return _to_list(observables[column])
    for name in candidates:
        if name in analytic_values:
            return _to_list(analytic_values[name])
        if name in observables:
            return _to_list(observables[name])
    return []


def _time_series_table(result: dict, model: str) -> dict:
    """构造时间序列表（columns + rows），供 CSV 与 HTML 共用。"""
    times = _to_list(result.get("times"))
    ordered = CSV_COLUMNS.get(model, [])
    columns, data = [], {}

    if times:
        columns.append("t")
        data["t"] = times
    for column in ordered:
        if column == "t":
            continue
        values = _lookup_column(result, column)
        if values:
            columns.append(column)
            data[column] = values
    observables = result.get("observables") or {}
    for name in sorted(observables.keys()):
        if name in columns:
            continue
        values = _to_list(observables[name])
        if values:
            columns.append(name)
            data[name] = values
    if not columns:
        return {"columns": [], "rows": [], "present": False}
    length = max(len(data[column]) for column in columns)
    rows = []
    for index in range(length):
        rows.append([data[column][index] if index < len(data[column]) else None
                     for column in columns])
    return {"columns": columns, "rows": rows, "present": True}


def _scalar_cell_ok(value: Any) -> bool:
    """扫描表只接受可直接写进 CSV 单元格的标量（列表/None/字典一律丢弃）。"""
    if value is None or isinstance(value, (list, tuple, dict, set)):
        return False
    return isinstance(value, (int, float, bool, complex, str)) or isinstance(value, numbers.Number)


def _flatten_sweep_entry(entry: dict, param: str, index: int, values: list) -> dict:
    """把 ``{"value": λ, "scalars": {...}, "observables": {...}}`` 展平成一行的列-值映射。

    - ``observables`` 不进表（时间序列型数据不适合一维扫描表）；
    - ``scalars`` 子字典按键展开；
    - 与参数名同名的标量列丢弃（避免与 param 列重复）；
    - 兼容另一种扁平写法（标量直接放在条目顶层）。
    """
    flat = {}
    if "value" in entry:
        flat[param] = entry.get("value")
    elif param in entry:
        flat[param] = entry.get(param)
    elif index < len(values):
        flat[param] = values[index]
    scalars = entry.get("scalars")
    if isinstance(scalars, dict):
        for name, value in scalars.items():
            if name == param:
                continue
            flat.setdefault(name, value)
    for name, value in entry.items():
        if name in ("value", "scalars", "observables", "series", "param", param):
            continue
        flat.setdefault(name, value)
    return flat


def _sweep_table(result: dict, model: str) -> dict:
    """构造参数扫描表（columns + rows），供 CSV 与 HTML 共用（同源）。

    列序：微扰严格按 ``references/report-format.md`` 的
    ``lambda, E0_pt1, E0_pt2, E0_exact, abs_err, rel_err, within_threshold``，
    其余标量按出现顺序追加；自旋/谐振子为 ``<param>`` + 标量列（按出现顺序）。
    """
    sweep = result.get("sweep")
    if not isinstance(sweep, dict):
        return None
    param = str(sweep.get("param") or "param")
    series = sweep.get("series") or []
    values = _to_list(sweep.get("values"))
    if not series:
        return None
    entries = []
    for index, item in enumerate(series):
        if not isinstance(item, dict):
            item = {"value": item}
        entries.append(_flatten_sweep_entry(item, param, index, values))

    # 收集出现过的标量列，保持首次出现顺序
    ordered_keys = []
    for entry in entries:
        for key in entry:
            if key not in ordered_keys:
                ordered_keys.append(key)
    # 非标量列（列表 / None / 字典）整列丢弃
    usable = []
    for key in ordered_keys:
        present = [entry[key] for entry in entries if key in entry]
        if present and all(_scalar_cell_ok(value) for value in present):
            usable.append(key)

    if model == "perturbation":
        canonical = [name for name in CSV_COLUMNS["perturbation"]
                     if name in usable and name != param]
        columns = ([param] if param in usable else []) + canonical
        columns += [name for name in usable if name not in columns]
    else:
        columns = ([param] if param in usable else [])
        columns += [name for name in usable if name != param]

    rows = [[entry.get(column) for column in columns] for entry in entries]
    return {"param": param, "columns": columns, "rows": rows, "present": True}


def write_csv(path: str, result: dict) -> str:
    """写出数据 CSV（utf-8-sig）：时间序列或 λ 扫描表。

    自旋/谐振子：``t`` + 各 observables 列（缺的列自动跳过）；
    微扰：``lambda, E0_pt1, …`` 扫描表；若自旋/谐振子同时带参数扫描，
    额外在同目录写出 ``sweep.csv``（见 references/report-format.md 第 4 节）。
    """
    path = str(path)
    directory = os.path.dirname(os.path.abspath(path))
    util.ensure_dir(directory)
    result = result if isinstance(result, dict) else {}
    model = result.get("model") or ""
    if not result:
        with open(path, "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["status", "message"])
            writer.writerow(["failed", "执行失败（没有结果）"])
        return path

    if model == "perturbation":
        table = _sweep_table(result, model)
        if table is None:
            scalars = result.get("scalars") or {}
            columns = [name for name in CSV_COLUMNS["perturbation"] if name in scalars]
            columns += [name for name in sorted(scalars.keys()) if name not in columns]
            table = {"columns": columns,
                     "rows": [[scalars.get(column) for column in columns]] if columns else []}
        _write_table(path, table["columns"], table["rows"])
        return path

    table = _time_series_table(result, model)
    _write_table(path, table["columns"], table["rows"])
    sweep = _sweep_table(result, model)
    if sweep is not None:
        _write_table(os.path.join(directory, "sweep.csv"),
                     sweep["columns"], sweep["rows"])
    return path


def _write_table(path: str, columns: list, rows: list):
    util.ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(columns))
        for row in rows:
            writer.writerow([_csv_value(value) for value in row])


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------
def write_json(path: str, payload: Any) -> str:
    """写出 JSON（UTF-8、中文不转义、缩进 2）。"""
    path = str(path)
    util.ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(util.to_jsonable(payload), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return path


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------
_CSS = """
:root { color-scheme: light; }
* { box-sizing: border-box; }
body {
  font-family: "Microsoft YaHei", "PingFang SC", "Hiragino Sans GB", "Noto Sans CJK SC", sans-serif;
  color: #222; line-height: 1.68; max-width: 1040px; margin: 0 auto; padding: 26px 22px 60px;
  background: #ffffff;
}
h1 { font-size: 26px; margin: 0 0 6px; line-height: 1.35; }
h2 { font-size: 20px; margin: 34px 0 12px; padding-bottom: 6px;
     border-bottom: 2px solid #2E75B6; page-break-after: avoid; }
h3 { font-size: 16px; margin: 18px 0 8px; color: #1f3b57; }
p, li { font-size: 14.5px; }
.subtitle { color: #556; font-size: 14px; margin: 0 0 12px; }
.meta { display: flex; flex-wrap: wrap; gap: 8px; margin: 12px 0 6px; }
.chip { background: #F2F6FA; border: 1px solid #D0D7DE; border-radius: 6px;
        padding: 4px 10px; font-size: 13px; }
.badge { display: inline-block; padding: 5px 14px; border-radius: 16px;
         font-weight: 700; font-size: 14px; border: 1px solid transparent; }
.badge.ok { background: #E8F5E9; color: #1B5E20; border-color: #A5D6A7; }
.badge.warn { background: #FFF8E1; color: #8D6E00; border-color: #FFE082; }
.badge.bad { background: #FDECEA; color: #B71C1C; border-color: #EF9A9A; }
.badge.skip { background: #ECEFF1; color: #455A64; border-color: #B0BEC5; }
.banner { border: 1px solid #D0D7DE; border-left-width: 6px; border-radius: 6px;
          padding: 10px 14px; margin: 14px 0; background: #FAFBFC; }
.banner.ok { border-left-color: #2E7D32; background: #F4FBF4; }
.banner.warn { border-left-color: #F9A825; background: #FFFCF3; }
.banner.bad { border-left-color: #C0392B; background: #FDF6F5; }
.banner.skip { border-left-color: #78909C; background: #F7F9FA; }
table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 13.5px; }
th, td { border: 1px solid #D0D7DE; padding: 6px 9px; text-align: left;
         vertical-align: top; word-break: break-word; }
th { background: #F2F6FA; font-weight: 600; }
tr { page-break-inside: avoid; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
.status-pass { color: #1B5E20; font-weight: 700; }
.status-fail { color: #B71C1C; font-weight: 700; }
.status-skip { color: #546E7A; font-weight: 700; }
figure { margin: 20px 0; page-break-inside: avoid; }
figure img { max-width: 100%; height: auto; border: 1px solid #E3E8EE; border-radius: 4px; }
figcaption { font-size: 13.5px; color: #444; margin-top: 7px; }
ol.toc { padding-left: 22px; }
ol.toc li { margin: 3px 0; }
pre { background: #F6F8FA; border: 1px solid #E3E8EE; border-radius: 6px;
      padding: 12px; overflow-x: auto; font-size: 12.5px; line-height: 1.5;
      white-space: pre-wrap; word-break: break-all; font-family: Consolas, "Courier New", monospace; }
code { font-family: Consolas, "Courier New", monospace; }
.muted { color: #667; font-size: 13px; }
.kv-limits li { margin: 4px 0; }
.footer { margin-top: 40px; border-top: 1px solid #D0D7DE; padding-top: 10px;
          color: #667; font-size: 12.5px; }
@media print {
  body { max-width: none; padding: 0; font-size: 12pt; }
  h2 { page-break-after: avoid; }
  a { color: inherit; text-decoration: none; }
  .no-print { display: none; }
  figure img { max-height: 150mm; }
}
"""


def _render_html(bundle_data: dict) -> str:
    """把 bundle 渲染成单文件 HTML（中文、离线、可打印）。"""
    status_class = str(bundle_data.get("status_class") or "skip")
    status_label = str(bundle_data.get("status_label") or "未知")
    model = bundle_data.get("model") or {}
    experiment = bundle_data.get("experiment") or {}
    parts = []
    parts.append("<!DOCTYPE html>")
    parts.append('<html lang="zh-CN"><head><meta charset="utf-8">')
    parts.append('<meta name="viewport" content="width=device-width, initial-scale=1">')
    parts.append('<meta name="generator" content="QuantumLab">')
    title = f"QuantumLab 实验报告 · {model.get('name', '')} · {experiment.get('id', '')}"
    parts.append(f"<title>{_e(title)}</title><style>{_CSS}</style></head><body>")

    # ------------------------------------------------------------ 标题区
    parts.append('<header id="top">')
    parts.append(f"<h1>QuantumLab 实验报告：{_e(model.get('name') or model.get('key') or '未知模型')}</h1>")
    parts.append('<p class="subtitle">受控本地数值实验 · 物理一致性检查 · 单文件可打印报告</p>')
    chips = [
        ("实验编号", experiment.get("id")),
        ("生成时间", bundle_data.get("generated_at")),
        ("模型", f"{model.get('key', '—')}"),
        ("关系", experiment.get("relation") or "首次"),
        ("求解器", experiment.get("solver")),
        ("耗时", f"{experiment.get('duration_ms')} ms"
                 if experiment.get("duration_ms") is not None else "—"),
    ]
    parts.append('<div class="meta">')
    for label, value in chips:
        parts.append(f'<span class="chip">{_e(label)}：{_e(value)}</span>')
    parts.append("</div>")
    parts.append(f'<div class="banner {status_class}"><span class="badge {status_class}">'
                 f'{_e(status_label)}</span> <span>{_e(bundle_data.get("status_detail"))}</span></div>')
    parts.append("</header>")

    # ------------------------------------------------------------ 目录
    parts.append('<nav class="no-print"><h2>目录</h2><ol class="toc">')
    for section in bundle_data.get("sections") or []:
        parts.append(f'<li><a href="#{_e(section["id"])}">{_e(section["title"])}</a></li>')
    parts.append("</ol></nav>")

    # ------------------------------------------------------------ 1 研究问题与目的
    parts.append('<section id="sec-1"><h2>1. 研究问题与实验目的</h2>')
    question = bundle_data.get("question") or ""
    if question:
        parts.append("<h3>原始问题</h3>")
        parts.append(f"<blockquote>{_e(question)}</blockquote>")
    parts.append("<h3>实验目的</h3><ul>")
    for item in bundle_data.get("purpose") or []:
        parts.append(f"<li>{_e(item)}</li>")
    parts.append("</ul></section>")

    # ------------------------------------------------------------ 2 模型公式与假设
    parts.append('<section id="sec-2"><h2>2. 模型公式与假设</h2>')
    parts.append(f"<p><strong>模型：</strong>{_e(model.get('name'))}"
                 f"（<code>{_e(model.get('key'))}</code>）</p>")
    parts.append(f"<p><strong>基矢：</strong>{_e(model.get('basis'))}</p>")
    parts.append(f"<p><strong>哈密顿量：</strong><code>{_e(model.get('hamiltonian'))}</code></p>")
    parts.append("<h3>公式</h3><ul>")
    for equation in model.get("equations") or []:
        parts.append(f"<li><code>{_e(equation)}</code></li>")
    parts.append("</ul><h3>假设与约定</h3><ul>")
    for assumption in model.get("assumptions") or []:
        parts.append(f"<li>{_e(assumption)}</li>")
    parts.append("</ul>")
    units = bundle_data.get("units") or {}
    if units:
        parts.append(f'<p class="muted">单位约定：ħ = {_e(units.get("hbar", 1.0))}，'
                     f'{_e(units.get("note", ""))}</p>')
    parts.append("</section>")

    # ------------------------------------------------------------ 3 参数、初态与单位
    parts.append('<section id="sec-3"><h2>3. 参数、初态与单位</h2>')
    parts.append("<h3>参数表</h3>")
    parts.append("<table><thead><tr><th>参数</th><th>取值</th><th>单位</th>"
                 "<th>来源</th></tr></thead><tbody>")
    for row in bundle_data.get("params") or []:
        source_class = "status-skip" if row["source"] == "default" else "status-pass"
        parts.append(f"<tr><td>{_e(row['label'])}</td><td class=\"num\">{_e(row['value'])}</td>"
                     f"<td>{_e(row['unit'])}</td>"
                     f"<td class=\"{source_class}\">{_e(row['source_text'])}</td></tr>")
    if not bundle_data.get("params"):
        parts.append('<tr><td colspan="4">请求中没有可展示的参数。</td></tr>')
    parts.append("</tbody></table>")
    state = bundle_data.get("initial_state") or {}
    parts.append(f"<h3>初态</h3><p>{_e(state.get('text'))}</p>")
    if state.get("details"):
        parts.append("<ul>")
        for item in state["details"]:
            parts.append(f"<li>{_e(item)}</li>")
        parts.append("</ul>")
    time_block = bundle_data.get("time") or {}
    parts.append(f"<h3>时间设置</h3><p>{_e(time_block.get('text'))}</p>")
    parts.append("</section>")

    # ------------------------------------------------------------ 4 数值方法
    method = bundle_data.get("method") or {}
    parts.append('<section id="sec-4"><h2>4. 数值方法</h2><ol>')
    for step in method.get("steps") or []:
        parts.append(f"<li>{_e(step)}</li>")
    parts.append("</ol>")
    parts.append(f"<p><strong>求解器：</strong>{_e(method.get('solver_note'))}"
                 f"（后端：{_e(method.get('backend'))}）</p>")
    tolerance = method.get("tolerance") or []
    if tolerance:
        parts.append("<h3>容差设置</h3><table><thead><tr><th>容差项</th>"
                     "<th class=\"num\">值</th></tr></thead><tbody>")
        for item in tolerance:
            parts.append(f"<tr><td>{_e(item['name'])}</td>"
                         f"<td class=\"num\">{_e(item['value_text'])}</td></tr>")
        parts.append("</tbody></table>")
    parts.append("</section>")

    # ------------------------------------------------------------ 5 结果图像与关键数据
    parts.append('<section id="sec-5"><h2>5. 结果图像与关键数据</h2>')
    figures = bundle_data.get("figures") or []
    if figures:
        for record in figures:
            parts.append("<figure>")
            parts.append(f'<img src="data:image/png;base64,{record["png_b64"]}" '
                         f'alt="图 {record["index"]}：{_e(record["caption"])}">')
            caption = record["caption"] or "（无图注）"
            parts.append(f'<figcaption>图 {record["index"]}：{_e(caption)}'
                         f'<span class="muted">（{record["bytes"]} 字节 PNG，'
                         f'3000×1760）</span></figcaption>')
            parts.append("</figure>")
    else:
        parts.append('<p class="muted">本次未生成图像'
                     '（执行失败，或 output.figures 为 false / formats 不含 png）。</p>')

    key_rows = bundle_data.get("key_values") or []
    parts.append("<h3>关键数值</h3>")
    if key_rows:
        parts.append("<table><thead><tr><th>指标</th><th>数值</th><th>说明</th>"
                     "</tr></thead><tbody>")
        for row in key_rows:
            parts.append(f"<tr><td>{_e(row['label'])}</td>"
                         f"<td class=\"num\">{_e(row['value'])}</td>"
                         f"<td>{_e(row['note'])}</td></tr>")
        parts.append("</tbody></table>")
    else:
        parts.append('<p class="muted">执行失败或结果中没有可展示的关键数值。</p>')

    sweep_table = bundle_data.get("sweep_table")
    if sweep_table and sweep_table.get("rows"):
        parts.append(f"<h3>参数扫描数据（扫描变量：<code>{_e(sweep_table['param'])}</code>）</h3>")
        parts.append(_html_table(sweep_table["columns"], sweep_table["rows"]))

    series_table = bundle_data.get("series_table")
    if series_table and series_table.get("present"):
        parts.append(f'<p class="muted">时间序列表的列：'
                     f'<code>{_e(", ".join(series_table["columns"]))}</code>'
                     f'（共 {len(series_table["rows"])} 行；完整数据见 data.csv）。</p>')
    parts.append("</section>")

    # ------------------------------------------------------------ 6 物理一致性验证
    validation = bundle_data.get("validation") or {}
    parts.append('<section id="sec-6"><h2>6. 物理一致性验证</h2>')
    counts = validation.get("counts") or {}
    parts.append(f'<div class="banner {_e(validation.get("class"))}">'
                 f'<span class="badge {_e(validation.get("class"))}">'
                 f'{_e(validation.get("label"))}</span> '
                 f'<span>通过 {int(counts.get("pass", 0))} 项 · '
                 f'未通过 {int(counts.get("fail", 0))} 项 · '
                 f'跳过 {int(counts.get("skipped", 0))} 项。'
                 f'{_e(validation.get("detail"))}</span></div>')
    checks = validation.get("checks") or []
    if checks:
        parts.append("<table><thead><tr><th>#</th><th>检查项</th><th>状态</th>"
                     "<th>指标</th><th class=\"num\">指标值</th>"
                     "<th class=\"num\">容差</th><th>说明</th></tr></thead><tbody>")
        for index, check in enumerate(checks, start=1):
            parts.append(
                f"<tr><td>{index}</td><td>{_e(check['name'])}<br>"
                f"<span class=\"muted\">{_e(check['id'])}</span></td>"
                f"<td class=\"status-{_e(check['status_class'])}\">{_e(check['status_label'])}</td>"
                f"<td>{_e(check['metric_name'])}</td>"
                f"<td class=\"num\">{_e(check['metric_value'])}</td>"
                f"<td class=\"num\">{_e(check['tolerance'])}</td>"
                f"<td>{_e(check['detail'])}</td></tr>")
        parts.append("</tbody></table>")
    else:
        parts.append('<p class="muted">本次没有可展示的检查项'
                     '（执行失败或验证未运行）。</p>')
    limits = validation.get("limits") or []
    if limits:
        parts.append("<h3>本轮验证的适用边界</h3><ul class=\"kv-limits\">")
        for item in limits:
            parts.append(f"<li>{_e(item)}</li>")
        parts.append("</ul>")
    parts.append("</section>")

    # ------------------------------------------------------------ 7 结果解释与适用限制
    interpretation = bundle_data.get("interpretation") or {}
    parts.append('<section id="sec-7"><h2>7. 结果解释与适用限制</h2>')
    parts.append("<h3>发生了什么</h3><ul>")
    for item in interpretation.get("what") or []:
        parts.append(f"<li>{_e(item)}</li>")
    parts.append("</ul><h3>为什么</h3><ul>")
    for item in interpretation.get("why") or []:
        parts.append(f"<li>{_e(item)}</li>")
    parts.append("</ul><h3>限制与未验证部分</h3><ul class=\"kv-limits\">")
    for item in bundle_data.get("limits") or []:
        parts.append(f"<li>{_e(item)}</li>")
    parts.append("</ul>")
    warnings = bundle_data.get("warnings") or []
    if warnings:
        parts.append("<h3>数值层提示</h3><ul>")
        for item in warnings:
            parts.append(f"<li>{_e(item)}</li>")
        parts.append("</ul>")
    parts.append("</section>")

    # ------------------------------------------------------------ 8 复现配置
    repro = bundle_data.get("reproduction") or {}
    parts.append('<section id="sec-8"><h2>8. 复现配置</h2>')
    parts.append("<h3>规范化请求（可直接复用）</h3>")
    parts.append(f"<pre>{_e(repro.get('json_text', '{}'))}</pre>")
    parts.append("<h3>运行命令</h3><pre>"
                 f"{_e(repro.get('command', ''))}\n"
                 f"{_e(repro.get('inline_command', ''))}</pre>")
    parts.append("<h3>输出文件清单</h3><ul>")
    for name in repro.get("files") or []:
        parts.append(f"<li><code>{_e(name)}</code></li>")
    parts.append("</ul>")
    if not bundle_data.get("executed"):
        parts.append('<p class="muted">注意：本次执行失败，请求 JSON 可用于修正后重跑，'
                     '但当前没有结果文件可供复现。</p>')
    parts.append("</section>")

    # ------------------------------------------------------------ 9 附录
    parts.append('<section id="sec-9"><h2>9. 附录：生成信息与文件清单</h2>')
    parts.append("<table><tbody>")
    appendix = [
        ("实验编号", experiment.get("id")),
        ("生成时间", bundle_data.get("generated_at")),
        ("实验创建时间", experiment.get("created_at") or "—"),
        ("模型", f"{model.get('name')}（{model.get('key')}）"),
        ("求解器", experiment.get("solver")),
        ("父实验", experiment.get("parent_id") or "—"),
        ("派生关系", experiment.get("relation") or "首次"),
        ("总状态", f"{status_label}（overall={bundle_data.get('status')}）"),
        ("图数量", len(figures)),
    ]
    for label, value in appendix:
        parts.append(f"<tr><th>{_e(label)}</th><td>{_e(value)}</td></tr>")
    parts.append("</tbody></table>")
    parts.append('<p class="muted">本报告由 QuantumLab 本地求解器生成：数值仅来自本次实验，'
                 '不含任何未运行实验的描述；图中 PNG 与数据表均可离线查看。</p>')
    parts.append("</section>")

    parts.append('<p class="footer">QuantumLab · schema '
                 f'{_e(bundle_data.get("schema"))} · 生成于 '
                 f'{_e(bundle_data.get("generated_at"))}</p>')
    parts.append("</body></html>")
    return "\n".join(parts)


def _html_table(columns: list, rows: list) -> str:
    out = ["<table><thead><tr>"]
    for column in columns:
        out.append(f"<th>{_e(column)}</th>")
    out.append("</tr></thead><tbody>")
    for row in rows:
        out.append("<tr>")
        for value in row:
            out.append(f'<td class="num">{_e(_csv_value(value))}</td>')
        out.append("</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def write_html(path: str, bundle: dict) -> str:
    """写出独立单文件 HTML 报告（图以 base64 内嵌，无外部依赖）。"""
    path = str(path)
    util.ensure_dir(os.path.dirname(os.path.abspath(path)))
    document = _render_html(bundle if isinstance(bundle, dict) else {})
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(document)
    return path
