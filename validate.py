# -*- coding: utf-8 -*-
"""QuantumLab 验证内核：物理一致性与数值验证检查表。

``validate(req, result)`` 返回 ``docs/CONTRACT.md`` 第 3 节规定的报告结构：

```python
{"checks": [{"id", "name", "status", "metric", "tolerance", "detail", "scope"}, ...],
 "overall": "pass" | "partial" | "skipped",
 "counts": {"pass": int, "fail": int, "skipped": int},
 "failed": [check_id, ...],
 "limits": [str, ...]}
```

设计原则：

- **不制造通过**：指标全部从 result 的原始数组重新计算，不照抄模型自报的
  ``metrics``；``skipped`` 必须写明"为什么不适用"。
- **区分"近似偏离"与"求解器错误"**：微扰二阶近似超出适用范围只会在
  ``perturbation_scope`` 里记 ``skipped``（物理预期），而精确对角化与闭式解
  不一致属于 ``solver_reference`` 失败（数值错误）。
- **检查不能把验证器自己弄崩**：单项检查因数据结构不满足前置条件而无法执行时，
  记为 ``skipped`` 并说明"检查覆盖缺口"，而不是抛异常终止整个报告。

检查项与容差见 ``references/validation.md``，默认容差来自 :mod:`qllab.spec`。
"""
from __future__ import annotations

import math

import numpy as np

from . import spec

__all__ = ["validate"]

#: 检查项顺序（与 references/validation.md 的表格一致）。
CHECK_IDS = (
    "input_legality", "finite_values", "hermiticity", "eigen_residual",
    "orthonormality", "state_norm", "probability", "energy_real",
    "energy_conservation", "analytic_agreement", "spectrum_agreement",
    "truncation_diagnostic", "solver_reference", "perturbation_scope",
)


class _NotApplicable(Exception):
    """该项检查不适用于当前模型/结果；消息会写入 detail。"""


def _need(condition, message):
    if not condition:
        raise _NotApplicable(message)


def _check(check_id, name, status, metric_name, metric_value, tolerance,
           detail, scope=""):
    """组装一条检查记录。

    ``metric_value`` 一律取 float；若出现 NaN/Inf，强制改为 fail——验证器
    不允许"非有限的指标"冒充通过。
    """
    try:
        value = float(metric_value)
    except (TypeError, ValueError):
        value = float("nan")
    if math.isnan(value):
        status = "fail"
    elif math.isinf(value):
        status = "fail"
    try:
        tol = float(tolerance)
    except (TypeError, ValueError):
        tol = 0.0
    return {
        "id": str(check_id),
        "name": str(name),
        "status": str(status),
        "metric": {"name": str(metric_name), "value": value},
        "tolerance": tol,
        "detail": str(detail),
        "scope": str(scope),
    }


def _safe(build, check_id, name, scope):
    """执行单项检查；把"不适用/无法执行"降级为 skipped，绝不让验证崩溃。"""
    try:
        return build()
    except _NotApplicable as exc:
        return _check(check_id, name, "skipped", "不适用", 0.0, 0.0,
                      str(exc), scope)
    except Exception as exc:  # pragma: no cover - 防御性分支
        return _check(
            check_id, name, "skipped", "检查执行异常", 0.0, 0.0,
            f"该检查因结果结构不满足前置条件而无法执行"
            f"（{type(exc).__name__}: {exc}）；这是检查覆盖缺口，未被判为数值错误。",
            scope)


# ---------------------------------------------------------------------------
# result 取值与遍历
# ---------------------------------------------------------------------------
def _matrix(result):
    _need(result, "缺少 result，无法验证")
    hamiltonian = result.get("hamiltonian")
    _need(hamiltonian is not None, "result 缺少 hamiltonian，无法验证")
    arr = np.asarray(hamiltonian)
    _need(arr.ndim == 2, f"hamiltonian 应为二维方阵，实际 ndim={arr.ndim}")
    _need(arr.shape[0] == arr.shape[1], f"hamiltonian 不是方阵：shape={arr.shape}")
    return arr


def _eigen(result):
    eigen = result.get("eigen")
    _need(isinstance(eigen, dict), "result 缺少 eigen 子字典，无法验证")
    values = eigen.get("values")
    vectors = eigen.get("vectors")
    _need(values is not None and vectors is not None,
          "eigen 缺少 values/vectors")
    values = np.asarray(values)
    vectors = np.asarray(vectors)
    _need(values.ndim == 1, "eigen.values 应为一维")
    _need(vectors.ndim == 2, "eigen.vectors 应为二维")
    _need(vectors.shape == (values.size, values.size),
          f"eigen.vectors shape={vectors.shape} 与能级数 {values.size} 不符")
    return values, vectors


def _observable(result, name):
    obs = result.get("observables")
    _need(isinstance(obs, dict), "result 缺少 observables 子字典，无法验证")
    _need(name in obs, f"观察量 {name} 不存在，无法验证")
    arr = np.asarray(obs[name], dtype=float).ravel()
    _need(arr.size > 0, f"观察量 {name} 为空")
    return arr


def _times_states(result):
    times = result.get("times")
    states = result.get("states")
    _need(times is not None, "该结果没有时间网格（times=None）")
    _need(states is not None, "该结果没有演化态（states=None）")
    times = np.asarray(times, dtype=float).ravel()
    states = np.asarray(states)
    _need(states.ndim == 2, "states 应为 (n_steps, dim) 二维数组")
    _need(times.size == states.shape[0],
          f"times 长度 {times.size} 与 states 行数 {states.shape[0]} 不一致")
    return times, states


def _collect_numeric(obj, path, arrays, scalars):
    """递归收集 result 中的数值数组与标量（用于有限性检查）。"""
    if obj is None or isinstance(obj, str):
        return
    if isinstance(obj, (bool, np.bool_)):
        return
    if isinstance(obj, (int, float, np.integer, np.floating,
                        complex, np.complexfloating)):
        scalars.append((path, obj))
        return
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind in "fcui" and obj.size > 0:
            arrays.append((path, obj))
        return
    if isinstance(obj, dict):
        for key, value in obj.items():
            _collect_numeric(value, f"{path}.{key}" if path else str(key),
                             arrays, scalars)
        return
    if isinstance(obj, (list, tuple)):
        try:
            arr = np.asarray(obj)
        except (TypeError, ValueError):
            arr = None
        if arr is not None and arr.dtype.kind in "fcui" and arr.size > 0:
            arrays.append((path, arr))
            return
        for index, item in enumerate(obj):
            _collect_numeric(item, f"{path}[{index}]", arrays, scalars)


def _nonfinite_count(arr):
    """返回数组中的 NaN/Inf 个数（复数按实部/虚部分别计）。"""
    if arr.dtype.kind == "c":
        return int(np.count_nonzero(~np.isfinite(arr.real))
                   + np.count_nonzero(~np.isfinite(arr.imag)))
    if arr.dtype.kind in "fiu":
        return int(np.count_nonzero(~np.isfinite(arr)))
    return 0


# ---------------------------------------------------------------------------
# 各检查项
# ---------------------------------------------------------------------------
def _input_legality(req, result, tol):
    """输入合法性：canonical request 是否满足白名单/范围，result 结构是否符合契约。"""
    issues = []
    if not isinstance(req, dict) or not req:
        issues.append("缺少 canonical request")
    model = req.get("model") if isinstance(req, dict) else None
    if model not in spec.MODELS:
        issues.append(f"model={model!r} 不在白名单 {list(spec.MODELS)}")

    params = req.get("params") if isinstance(req, dict) else None
    required = spec.MODEL_PARAMS.get(model, ())
    finite_params = True
    if not isinstance(params, dict):
        issues.append("params 缺失或不是对象")
        finite_params = False
    else:
        for key in required:
            if key not in params:
                issues.append(f"缺少参数 {key}")
                continue
            try:
                value = float(params[key])
            except (TypeError, ValueError):
                issues.append(f"参数 {key}={params[key]!r} 不是实数")
                finite_params = False
                continue
            if not math.isfinite(value):
                issues.append(f"参数 {key} 非有限")
                finite_params = False
        if model == "spin_half" and finite_params and all(
                key in params for key in required):
            om = math.sqrt(sum(float(params[key]) ** 2 for key in required))
            if om <= 0.0:
                issues.append("Ωx=Ωy=Ωz=0，哈密顿量为零")
            if om > spec.LIMITS["abs_param"]:
                issues.append(f"|Ω|={om:g} 超过上限 {spec.LIMITS['abs_param']:g}")
        if model == "harmonic_oscillator" and finite_params and "n_levels" in params:
            if not (spec.LIMITS["n_levels"][0] <= int(params["n_levels"])
                    <= spec.LIMITS["n_levels"][1]):
                issues.append(f"n_levels={params['n_levels']} 超出范围")
        if model == "perturbation" and finite_params and "delta" in params:
            if float(params["delta"]) <= 0.0:
                issues.append("微扰模型要求 Δ>0（非简并）")

    time_block = req.get("time") if isinstance(req, dict) else None
    if model == "perturbation":
        if time_block is not None:
            issues.append("微扰模型的 time 应为 None（静态能谱）")
    else:
        if not isinstance(time_block, dict):
            issues.append("time 缺失或不是对象")
        else:
            for key in ("t_start", "t_end", "n_steps"):
                if key not in time_block:
                    issues.append(f"time 缺少 {key}")
            if "t_start" in time_block and "t_end" in time_block:
                try:
                    t0 = float(time_block["t_start"])
                    t1 = float(time_block["t_end"])
                    if not (math.isfinite(t0) and math.isfinite(t1)):
                        issues.append("time 起止非有限")
                    elif not t1 > t0:
                        issues.append(f"time.t_end={t1} 必须大于 t_start={t0}")
                except (TypeError, ValueError):
                    issues.append("time 起止不是实数")
            if "n_steps" in time_block:
                try:
                    n_steps = int(time_block["n_steps"])
                    lo, hi = spec.LIMITS["n_steps"]
                    if not (lo <= n_steps <= hi):
                        issues.append(f"n_steps={n_steps} 超出范围 [{lo}, {hi}]")
                except (TypeError, ValueError):
                    issues.append("time.n_steps 不是整数")

    sweep = req.get("sweep") if isinstance(req, dict) else None
    if sweep is not None:
        if not isinstance(sweep, dict):
            issues.append("sweep 不是对象")
        else:
            if sweep.get("param") not in required:
                issues.append(f"sweep.param={sweep.get('param')!r} 不属于模型 {model}")
            values = sweep.get("values")
            if not isinstance(values, (list, tuple)) or not values:
                issues.append("sweep.values 缺失或为空")
            elif len(values) > spec.LIMITS["sweep_points"]:
                issues.append(f"sweep 点数 {len(values)} 超过上限")
            else:
                for value in values:
                    try:
                        if not math.isfinite(float(value)):
                            issues.append("sweep.values 含非有限值")
                            break
                    except (TypeError, ValueError):
                        issues.append("sweep.values 含非实数")
                        break

    if isinstance(result, dict) and result:
        if result.get("model") != model:
            issues.append(f"result.model={result.get('model')!r} 与请求 {model!r} 不一致")
        for key in ("hamiltonian", "basis", "eigen", "observables", "scalars",
                    "metrics", "figure_specs", "warnings", "summary"):
            if key not in result:
                issues.append(f"result 缺少字段 {key}")
        try:
            hamiltonian = _matrix(result)
        except _NotApplicable:
            hamiltonian = None
        if hamiltonian is not None:
            try:
                values, vectors = _eigen(result)
                dim = hamiltonian.shape[0]
                if values.size != dim:
                    issues.append(f"能级数 {values.size} 与哈密顿量维数 {dim} 不一致")
                if vectors.shape != (dim, dim):
                    issues.append(f"eigen.vectors shape={vectors.shape} 与 H 维数不符")
            except _NotApplicable as exc:
                issues.append(str(exc))
            basis = result.get("basis")
            if isinstance(basis, dict) and "dim" in basis:
                if int(basis["dim"]) != hamiltonian.shape[0]:
                    issues.append("basis.dim 与 H 维数不一致")
            if model in ("spin_half", "harmonic_oscillator"):
                try:
                    times, states = _times_states(result)
                    if time_block and "n_steps" in time_block:
                        if times.size != int(time_block["n_steps"]):
                            issues.append("times 长度与 time.n_steps 不一致")
                    if states.shape[1] != hamiltonian.shape[0]:
                        issues.append("states 列数与 H 维数不一致")
                except _NotApplicable as exc:
                    issues.append(str(exc))
            elif model == "perturbation":
                if result.get("times") is not None or result.get("states") is not None:
                    issues.append("微扰模型的 times/states 应为 None")
        if sweep is not None and result.get("sweep") is None:
            issues.append("请求有 sweep，但 result.sweep 为 None")
        if sweep is None and result.get("sweep") is not None and model != "perturbation":
            issues.append("请求没有 sweep，但 result 携带了 sweep")
        result_params = result.get("params")
        if isinstance(params, dict) and isinstance(result_params, dict):
            for key in required:
                try:
                    if abs(float(params[key]) - float(result_params[key])) > 1e-12:
                        issues.append(f"result.params[{key}] 与请求参数不一致")
                except (KeyError, TypeError, ValueError):
                    pass

    status = "pass" if not issues else "fail"
    shown = "；".join(issues[:10])
    if len(issues) > 10:
        shown += f"；…（共 {len(issues)} 项）"
    detail = ("canonical request 与 result 结构满足白名单、范围与契约结构约束。"
              if not issues else f"发现 {len(issues)} 个问题：{shown}")
    return _check("input_legality", "输入合法性", status,
                  "问题项数", float(len(issues)), 0.0, detail,
                  "spec.normalize_request 的 canonical 约束 + docs/CONTRACT.md 第 2 节结构")


def _finite_values(result):
    """数值有限性：result 中所有数值数组/标量不含 NaN/Inf。"""
    _need(isinstance(result, dict) and result, "缺少 result，无法检查有限性")
    arrays, scalars = [], []
    _collect_numeric(result, "", arrays, scalars)
    _need(arrays or scalars, "结果中没有可检查的数值")
    bad_paths = []
    bad_count = 0
    total = 0
    for path, arr in arrays:
        total += int(arr.size)
        bad = _nonfinite_count(arr)
        if bad:
            bad_paths.append(f"{path}（{bad} 个）")
            bad_count += bad
    for path, value in scalars:
        total += 1
        try:
            ok = math.isfinite(float(value)) if not isinstance(value, complex) \
                else (math.isfinite(value.real) and math.isfinite(value.imag))
        except (TypeError, ValueError):
            ok = False
        if not ok:
            bad_paths.append(f"{path}（非有限标量）")
            bad_count += 1
    status = "pass" if bad_count == 0 else "fail"
    detail = (f"共检查 {total} 个数值，未发现 NaN/Inf。"
              if bad_count == 0 else
              f"发现 {bad_count} 个非有限数值：{'；'.join(bad_paths[:8])}")
    return _check("finite_values", "数值有限性", status,
                  "NaN/Inf 计数", float(bad_count), 0.0, detail,
                  "result 中所有数值数组与标量（含 sweep 摘要）")


def _hermiticity(result, tol):
    """厄米性：max|H - H†|。"""
    hamiltonian = _matrix(result)
    dev = float(np.max(np.abs(hamiltonian - hamiltonian.conj().T))) \
        if hamiltonian.size else 0.0
    limit = float(tol["hermiticity_abs"])
    status = "pass" if dev <= limit else "fail"
    detail = (f"max|H-H†|={dev:.3e}；H 为厄米矩阵，谱分解前提成立。"
              if status == "pass" else
              f"max|H-H†|={dev:.3e} 超过容差 {limit:.1e}：输入矩阵不是厄米矩阵，"
              f"np.linalg.eigh 的结果不可信。")
    return _check("hermiticity", "厄米性", status, "max|H-H†|", dev, limit,
                  detail, "全体模型；H 必须等于其共轭转置")


def _eigen_residual(result, tol):
    """本征方程残差：max|H v_k - E_k v_k|。"""
    hamiltonian = _matrix(result)
    values, vectors = _eigen(result)
    residual = hamiltonian @ vectors - vectors * values[None, :]
    dev = float(np.max(np.abs(residual))) if residual.size else 0.0
    limit = float(tol["eigen_residual_abs"])
    status = "pass" if dev <= limit else "fail"
    detail = (f"max|Hv-Ev|={dev:.3e}；本征对满足本征方程。"
              if status == "pass" else
              f"max|Hv-Ev|={dev:.3e} 超过容差 {limit:.1e}：谱分解的本征对不自洽，"
              f"属于求解器/数据错误。")
    return _check("eigen_residual", "本征方程残差", status,
                  "max|Hv_k-E_k v_k|", dev, limit, detail,
                  "全体模型；检查 np.linalg.eigh 输出的本征对")


def _orthonormality(result, tol):
    """正交归一性：max|V†V - I|。"""
    _matrix(result)
    _values, vectors = _eigen(result)
    dim = vectors.shape[0]
    gram = vectors.conj().T @ vectors - np.eye(dim, dtype=complex)
    dev = float(np.max(np.abs(gram))) if gram.size else 0.0
    limit = float(tol["orthonormality_abs"])
    status = "pass" if dev <= limit else "fail"
    detail = (f"max|V†V-I|={dev:.3e}；本征矢构成正交归一基。"
              if status == "pass" else
              f"max|V†V-I|={dev:.3e} 超过容差 {limit:.1e}：本征矢不正交，"
              f"谱分解不可用。")
    return _check("orthonormality", "正交归一性", status, "max|V†V-I|", dev,
                  limit, detail, "全体模型；本征矢矩阵应为酉矩阵")


def _state_norm(result, model, tol):
    """态归一化：演化全程 max|||ψ(t)||-1|。"""
    _need(model in ("spin_half", "harmonic_oscillator"),
          f"{model} 没有时间演化态（微扰模型只做静态能谱），态归一化检查不适用")
    _times, states = _times_states(result)
    norms = np.linalg.norm(states, axis=1)
    dev = float(np.max(np.abs(norms - 1.0)))
    limit = float(tol["norm_dev"])
    status = "pass" if dev <= limit else "fail"
    detail = (f"max|||ψ(t)||-1|={dev:.3e}；幺正演化保持态归一。"
              if status == "pass" else
              f"max|||ψ(t)||-1|={dev:.3e} 超过容差 {limit:.1e}：演化不保范数，"
              f"属于求解器/相位实现错误。")
    return _check("state_norm", "态归一化", status, "max|||ψ(t)||-1|", dev,
                  limit, detail, "自旋与谐振子；微扰模型无时间演化")


def _probability(result, model, tol):
    """概率守恒：max|Σ占据 - 1|。"""
    if model == "spin_half":
        total = _observable(result, "pop_0") + _observable(result, "pop_1")
        dev = float(np.max(np.abs(total - 1.0)))
        metric_name = "max|P↑+P↓-1|"
        scope = "自旋模型；两个基态占据之和"
    elif model == "harmonic_oscillator":
        _times, states = _times_states(result)
        total = np.sum(np.abs(states) ** 2, axis=1)
        dev = float(np.max(np.abs(total - 1.0)))
        metric_name = "max|Σₙ|cₙ|²-1|"
        scope = "谐振子；截断子空间内全部能级占据之和"
    else:
        raise _NotApplicable(
            f"{model} 没有占据概率（微扰模型只做静态能谱），概率守恒检查不适用")
    limit = float(tol["probability_sum"])
    status = "pass" if dev <= limit else "fail"
    detail = (f"{metric_name}={dev:.3e}；概率解释自洽。"
              if status == "pass" else
              f"{metric_name}={dev:.3e} 超过容差 {limit:.1e}：占据概率不归一，"
              f"结果不可用于概率性结论。")
    return _check("probability", "概率守恒", status, metric_name, dev, limit,
                  detail, scope)


def _energy_real(result, model, tol):
    """能量实数：本征能量与能量期望的虚部应为 0。"""
    values, _vectors = _eigen(result)
    imag_eigen = float(np.max(np.abs(np.imag(values)))) if values.size else 0.0
    imag_expect = 0.0
    if model in ("spin_half", "harmonic_oscillator"):
        hamiltonian = _matrix(result)
        _times, states = _times_states(result)
        expectations = np.einsum("ti,ij,tj->t", np.conj(states),
                                 hamiltonian, states)
        imag_expect = float(np.max(np.abs(np.imag(expectations))))
    dev = max(imag_eigen, imag_expect)
    limit = float(tol["energy_imag"])
    status = "pass" if dev <= limit else "fail"
    detail = (f"max(|Im E_k|, |Im⟨H⟩|)={dev:.3e}；能量为实数。"
              if status == "pass" else
              f"max(|Im E_k|, |Im⟨H⟩|)={dev:.3e} 超过容差 {limit:.1e}："
              f"能量出现非零虚部，说明 H 非厄米或期望值计算有误。")
    return _check("energy_real", "能量实数", status,
                  "max(|Im E_k|, |Im⟨H⟩|)", dev, limit, detail,
                  "全体模型；厄米 H 的能量本征值与期望值必须为实数")


def _energy_conservation(result, model, tol):
    """能量守恒：max|⟨H⟩(t)-⟨H⟩(t₀)| / max(|⟨H⟩|, 1)。"""
    _need(model in ("spin_half", "harmonic_oscillator"),
          f"{model} 没有时间演化（静态能谱），能量守恒检查不适用")
    hamiltonian = _matrix(result)
    times, states = _times_states(result)
    _need(times.size >= 2, "时间网格少于 2 点，无法判断守恒性")
    expectations = np.einsum("ti,ij,tj->t", np.conj(states),
                             hamiltonian, states).real
    scale = max(float(np.max(np.abs(expectations))), 1.0)
    dev = float(np.max(np.abs(expectations - expectations[0])) / scale)
    limit = float(tol["energy_drift"])
    status = "pass" if dev <= limit else "fail"
    detail = (f"相对漂移 {dev:.3e}；封闭系统 ⟨H⟩ 守恒。"
              if status == "pass" else
              f"相对漂移 {dev:.3e} 超过容差 {limit:.1e}：⟨H⟩ 不守恒，"
              f"说明演化算符实现或时间参考相位有问题。")
    return _check("energy_conservation", "能量守恒", status,
                  "max|Δ⟨H⟩|/max(|⟨H⟩|,1)", dev, limit, detail,
                  "自旋与谐振子；微扰模型无时间演化")


def _analytic_agreement(result, model, tol):
    """解析结果对比：数值观察量与解析解的最大绝对偏差。"""
    if model == "perturbation":
        raise _NotApplicable(
            "微扰模型是静态近似分析，没有含时解析解；精确能谱对比由 "
            "solver_reference 检查覆盖，近似偏离由 perturbation_scope 表达")
    analytic = result.get("analytic")
    _need(isinstance(analytic, dict), "result 未提供 analytic 解析解，无法对比")
    values = analytic.get("values")
    _need(isinstance(values, dict) and values, "analytic.values 缺失或为空")
    observable_block = result.get("observables")
    _need(isinstance(observable_block, dict) and observable_block,
          "result 缺少 observables，无法与解析解对比")
    comparisons = {}
    for name, theory in values.items():
        if name not in observable_block:
            continue
        numeric = np.asarray(observable_block[name], dtype=float).ravel()
        theory = np.asarray(theory, dtype=float).ravel()
        if numeric.size != theory.size or numeric.size == 0:
            raise _NotApplicable(
                f"观察量 {name} 与解析解长度不一致，无法逐点对比")
        comparisons[name] = float(np.max(np.abs(numeric - theory)))
    _need(comparisons, "解析解与记录观察量没有可对比的公共量")
    dev = max(comparisons.values())
    limit = float(tol["analytic_abs"])
    status = "pass" if dev <= limit else "fail"
    per_key = "，".join(f"{key}={value:.3e}" for key, value in comparisons.items())
    detail = (f"数值与解析解最大绝对偏差 {dev:.3e}（{per_key}）；"
              f"解析解是同一离散模型下的精确参照。"
              if status == "pass" else
              f"数值与解析解最大绝对偏差 {dev:.3e}（{per_key}）超过容差 "
              f"{limit:.1e}：属于数值演化误差，不是近似方法偏差。")
    return _check("analytic_agreement", "解析结果对比", status,
                  "max|数值-解析|", dev, limit, detail,
                  "自旋（Bloch 进动）与谐振子（⟨x⟩/⟨p⟩ 解析式）；微扰模型无含时解析解")


def _spectrum_agreement(result, model, tol):
    """能谱对比：谐振子 E_n vs ω(n+½)。"""
    _need(model == "harmonic_oscillator",
          f"按验证规范，能谱对比仅适用于谐振子；{model} 不适用"
          + ("（微扰精确能谱由 solver_reference 覆盖）" if model == "perturbation"
             else "（自旋 E± 由解析 Bloch 对比与厄米性/本征残差检查约束）"))
    values, _vectors = _eigen(result)
    omega = float((result.get("params") or {}).get("omega", 0.0))
    _need(omega > 0.0, "result.params 缺少有效 omega，无法生成解析能谱")
    n = np.arange(values.size, dtype=float)
    theory = omega * (n + 0.5)
    dev = float(np.max(np.abs(values - theory)))
    limit = float(tol["spectrum_abs"])
    status = "pass" if dev <= limit else "fail"
    detail = (f"max|Eₙ-ω(n+½)|={dev:.3e}；能谱与解析值逐项一致。"
              if status == "pass" else
              f"max|Eₙ-ω(n+½)|={dev:.3e} 超过容差 {limit:.1e}："
              f"哈密顿量构造或对角化有误。")
    return _check("spectrum_agreement", "能谱对比", status,
                  "max|Eₙ-ω(n+½)|", dev, limit, detail,
                  "谐振子；微扰模型由 solver_reference 覆盖，自旋无此检查项")


def _truncation_diagnostic(result, model, tol):
    """截断诊断：边界能级最大占据（+ 相干态展开泄漏）。"""
    _need(model == "harmonic_oscillator",
          f"{model} 没有 Fock 截断空间，截断诊断不适用")
    boundary = _observable(result, "boundary_pop")
    boundary_max = float(np.max(boundary))
    tail = None
    scalars = result.get("scalars")
    if isinstance(scalars, dict):
        try:
            tail = float(scalars.get("truncation_tail_prob"))
        except (TypeError, ValueError):
            tail = None
    limit = float(tol["truncation_pop"])
    passed = boundary_max <= limit and (tail is None or tail <= limit)
    status = "pass" if passed else "fail"
    tail_text = (f"；相干态 Fock 展开截断尾部概率 {tail:.3e}" if tail is not None
                 else "")
    if status == "pass":
        detail = (f"边界能级最大占据 {boundary_max:.3e}{tail_text}，均在容差 "
                  f"{limit:.1e} 内：当前条件下截断泄漏可忽略。"
                  f"注意这不是收敛性证明。")
    else:
        detail = (f"边界能级最大占据 {boundary_max:.3e}{tail_text} 超过容差 "
                  f"{limit:.1e}：Fock 截断可能不足，建议增大 n_levels。"
                  f"这是模型截断限制，不等于数值发散。")
    return _check("truncation_diagnostic", "截断诊断", status,
                  "max boundary_pop", boundary_max, limit, detail,
                  "谐振子；有限 N 只是近似，占据小不等于已证明收敛")


def _solver_reference(result, model, tol):
    """求解器参照：微扰模型精确对角化 vs 闭式解 -½√(Δ²+λ²)。"""
    _need(model == "perturbation",
          f"求解器闭式解参照仅用于微扰模型，{model} 不适用")
    values, _vectors = _eigen(result)
    params = result.get("params") or {}
    delta = float(params.get("delta", 0.0))
    lam = float(params.get("lambda", 0.0))
    _need(delta > 0.0, "result.params 缺少有效 delta，无法构造闭式解")
    exact = -0.5 * math.sqrt(delta * delta + lam * lam)
    theory = np.array([exact, -exact])
    dev = float(np.max(np.abs(np.sort(values) - np.sort(theory))))
    limit = float(tol["analytic_abs"])
    status = "pass" if dev <= limit else "fail"
    detail = (f"max|E_eigh-E_closed|={dev:.3e}；精确对角化与闭式解 "
              f"-½√(Δ²+λ²) 一致。"
              if status == "pass" else
              f"max|E_eigh-E_closed|={dev:.3e} 超过容差 {limit:.1e}："
              f"这是数值求解器错误（闭式解是精确参照），不是微扰近似偏差。")
    return _check("solver_reference", "求解器参照", status,
                  "max|E_eigh-E_closed|", dev, limit, detail,
                  "微扰模型；精确对角化与闭式解应达到机器精度")


def _perturbation_scope(result, model, tol):
    """微扰适用范围：当前 λ 的二阶近似相对误差。"""
    _need(model == "perturbation",
          f"{model} 不包含微扰展开，适用范围检查不适用")
    params = result.get("params") or {}
    delta = float(params.get("delta", 0.0))
    lam = float(params.get("lambda", 0.0))
    _need(delta > 0.0, "result.params 缺少有效 delta，无法计算微扰误差")
    exact = -0.5 * math.sqrt(delta * delta + lam * lam)
    pt2 = -0.5 * delta - lam * lam / (4.0 * delta)
    rel_err = abs(pt2 - exact) / abs(exact) if exact != 0.0 else 0.0
    limit = float(tol["perturbation_rel"])
    lambda_valid_max = None
    sweep = result.get("sweep")
    if isinstance(sweep, dict):
        for item in sweep.get("series") or []:
            try:
                if float(item["scalars"]["within_threshold"]) > 0.5:
                    value = float(item["value"])
                    lambda_valid_max = value if lambda_valid_max is None \
                        else max(lambda_valid_max, value)
            except (KeyError, TypeError, ValueError):
                continue
    valid_text = (f"；本次扫描中满足阈值的最大采样 λ={lambda_valid_max:g}"
                  if lambda_valid_max is not None else "")
    if rel_err <= limit:
        status = "pass"
        detail = (f"当前 λ={lam:g}、Δ={delta:g} 下二阶微扰相对误差 {rel_err:.3e} "
                  f"≤ 阈值 {limit:.1e}：二阶近似在当前参数点可用{valid_text}。"
                  f"适用范围只对当前 Δ 与采样点成立。")
    else:
        status = "skipped"
        detail = (f"当前 λ={lam:g}、Δ={delta:g} 下二阶微扰相对误差 {rel_err:.3e} "
                  f"> 阈值 {limit:.1e}：近似已超出适用范围，**这是物理预期而非数值错误**"
                  f"（λ/Δ≫1 时需精确对角化或更高阶微扰）{valid_text}。")
    return _check("perturbation_scope", "微扰适用范围", status,
                  "rel_err(二阶近似)", rel_err, limit, detail,
                  "微扰模型；只对当前 Δ 与扫描采样点成立")


# ---------------------------------------------------------------------------
# 报告组装
# ---------------------------------------------------------------------------
def _build_limits(req, result, model, counts):
    """输出结论适用边界（报告与终端都会展示）。"""
    limits = [
        "统一采用 ħ=1 的无量纲约定：能量与角频率同单位、时间以 Ω⁻¹ 或 ω⁻¹ 度量。",
        "求解器为厄米矩阵谱分解 np.linalg.eigh（U(t)=V e^{-iΛ(t-t_start)} V†）；"
        "误差指标只衡量数值一致性，不替代物理建模假设。",
    ]
    params = (result.get("params") if isinstance(result, dict) else None) or \
             (req.get("params") if isinstance(req, dict) else None) or {}
    if model == "spin_half":
        time_block = (req.get("time") or {}) if isinstance(req, dict) else {}
        state = (req.get("initial_state") or {}) if isinstance(req, dict) else {}
        limits.append(
            f"仅对 Ωx={params.get('Omega_x')!r}、Ωy={params.get('Omega_y')!r}、"
            f"Ωz={params.get('Omega_z')!r}、初态 kind={state.get('kind')!r}、"
            f"t∈[{time_block.get('t_start')!r}, {time_block.get('t_end')!r}] 成立；"
            f"未验证含时驱动、退相干与噪声。")
    elif model == "harmonic_oscillator":
        n_levels = params.get("n_levels")
        limits.append(
            f"Fock 空间截断 N={n_levels}（n=0…{int(n_levels) - 1}）；"
            f"边界占据与截断尾部概率小只说明当前条件下泄漏可忽略，"
            f"不构成 N→∞ 的收敛性证明。")
    elif model == "perturbation":
        sweep = (result.get("sweep") if isinstance(result, dict) else None) or {}
        values = sweep.get("values") or []
        span = (f"{min(values):g}…{max(values):g}" if values else "未扫描")
        limits.append(
            f"二阶微扰结论只对 Δ={params.get('delta')!r} 与扫描采样点 λ∈{span} "
            f"成立；λ/Δ≫1 时需精确对角化或更高阶微扰，"
            f"超出阈值的近似偏离不是计算错误。")
    if counts["fail"] > 0:
        limits.append("存在未通过的检查项：上述数值结论不可直接对外作为已验证结果。")
    limits.append("未涵盖：有限温度、开放量子系统（Lindblad/退相干）、"
                  "多体关联、显含时哈密顿量与实验噪声。")
    return limits


def validate(req: dict, result: dict) -> dict:
    """生成契约第 3 节的验证报告。

    :param req: canonical request（:func:`qllab.spec.normalize_request` 输出）。
    :param result: :func:`qllab.models.run_model` 的输出。
    :return: 含 ``checks`` / ``overall`` / ``counts`` / ``failed`` / ``limits`` 的 dict。
    """
    req = req if isinstance(req, dict) else {}
    result = result if isinstance(result, dict) else {}
    model = req.get("model") or result.get("model")

    tol = dict(spec.DEFAULT_TOL)
    raw_tol = req.get("tolerance")
    if isinstance(raw_tol, dict):
        for key, value in raw_tol.items():
            if key in tol:
                try:
                    tol[key] = float(value)
                except (TypeError, ValueError):
                    pass

    checks = []
    checks.append(_safe(lambda: _input_legality(req, result, tol),
                        "input_legality", "输入合法性",
                        "canonical request 与契约结构"))
    checks.append(_safe(lambda: _finite_values(result),
                        "finite_values", "数值有限性", "result 全部数值"))
    checks.append(_safe(lambda: _hermiticity(result, tol),
                        "hermiticity", "厄米性", "H=H†"))
    checks.append(_safe(lambda: _eigen_residual(result, tol),
                        "eigen_residual", "本征方程残差", "H v = E v"))
    checks.append(_safe(lambda: _orthonormality(result, tol),
                        "orthonormality", "正交归一性", "V†V=I"))
    checks.append(_safe(lambda: _state_norm(result, model, tol),
                        "state_norm", "态归一化", "||ψ(t)||=1"))
    checks.append(_safe(lambda: _probability(result, model, tol),
                        "probability", "概率守恒", "Σ占据=1"))
    checks.append(_safe(lambda: _energy_real(result, model, tol),
                        "energy_real", "能量实数", "Im E=0"))
    checks.append(_safe(lambda: _energy_conservation(result, model, tol),
                        "energy_conservation", "能量守恒", "⟨H⟩ 不随时间变化"))
    checks.append(_safe(lambda: _analytic_agreement(result, model, tol),
                        "analytic_agreement", "解析结果对比",
                        "同一离散模型的精确解析参照"))
    checks.append(_safe(lambda: _spectrum_agreement(result, model, tol),
                        "spectrum_agreement", "能谱对比", "Eₙ vs 解析能谱"))
    checks.append(_safe(lambda: _truncation_diagnostic(result, model, tol),
                        "truncation_diagnostic", "截断诊断",
                        "有限 Fock 截断"))
    checks.append(_safe(lambda: _solver_reference(result, model, tol),
                        "solver_reference", "求解器参照",
                        "精确对角化 vs 闭式解"))
    checks.append(_safe(lambda: _perturbation_scope(result, model, tol),
                        "perturbation_scope", "微扰适用范围",
                        "二阶微扰只在 λ/Δ≪1 时可靠"))

    counts = {"pass": 0, "fail": 0, "skipped": 0}
    for item in checks:
        counts[item["status"]] = counts.get(item["status"], 0) + 1
    failed = [item["id"] for item in checks if item["status"] == "fail"]

    if counts["fail"] > 0:
        overall = "partial"       # 有完整结果，但至少一项检查失败
    elif counts["pass"] == 0:
        overall = "skipped"       # 所有检查都不适用
    else:
        overall = "pass"          # 所有执行的检查通过（skipped 不计入失败）

    return {
        "checks": checks,
        "overall": overall,
        "counts": counts,
        "failed": failed,
        "limits": _build_limits(req, result, model, counts),
    }
