# -*- coding: utf-8 -*-
"""QuantumLab 物理内核：三类模型的正演求解。

本模块只依赖 ``numpy`` 与标准库（无 scipy / matplotlib），不含任何
``eval`` / ``exec`` 或动态代码执行。输入是 :mod:`qllab.spec` 规范化后的
canonical request，输出严格遵循 ``docs/CONTRACT.md`` 第 2 节的 result 结构；
图表只输出声明式 ``figure_specs``（第 4 节），实际绘制交给
:mod:`qllab.charts`。

物理约定：**ħ = 1**，能量与角频率同量纲；时间以 ``t_start`` 为参考相位，
即所有含时因子都写成 ``exp(-i E (t - t_start))``，因此能量期望的守恒性
可以直接比较。

三类模型：

- ``spin_half``：``H = ½(Ωx σx + Ωy σy + Ωz σz)``，解析参照是 Bloch 进动；
- ``harmonic_oscillator``：``H = ω(a†a + ½)`` 在 N 维 Fock 截断子空间内；
- ``perturbation``：``H = (Δ/2)σz + (λ/2)σx``，精确对角化 vs 一阶/二阶微扰。
"""
from __future__ import annotations

import math

import numpy as np

from . import spec

__all__ = ["run_model"]

_SQRT2 = math.sqrt(2.0)

_SIGMA_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
_SIGMA_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
_SIGMA_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)

# 图中时间序列最多画这么多点（防止 n_steps=20000 时 PNG/JSON 过大）；
# 数值验证仍然使用全部采样点。
_MAX_FIG_POINTS = 1200


# ---------------------------------------------------------------------------
# 通用小工具
# ---------------------------------------------------------------------------
def _lst(values):
    """把数组/列表安全地转成 Python float 列表（用于 figspec）。"""
    return [float(v) for v in np.asarray(values, dtype=float).ravel()]


def _decimate(x, y, max_points: int = _MAX_FIG_POINTS):
    """等间隔抽稀曲线，保留首尾点；仅用于绘图。"""
    xs = np.asarray(x, dtype=float).ravel()
    ys = np.asarray(y, dtype=float).ravel()
    n = min(xs.size, ys.size)
    xs, ys = xs[:n], ys[:n]
    if n <= max_points:
        return xs, ys
    stride = int(math.ceil(n / float(max_points)))
    idx = np.arange(0, n, stride, dtype=int)
    if idx.size == 0 or idx[-1] != n - 1:
        idx = np.append(idx, n - 1)
    return xs[idx], ys[idx]


def _series(label, x, y, style: str = "line"):
    """构造 figspec 的一条曲线；超长序列自动抽稀。"""
    xs, ys = _decimate(x, y)
    return {"label": str(label),
            "x": [float(v) for v in xs],
            "y": [float(v) for v in ys],
            "style": str(style)}


def _fig(name, kind, title, xlabel, ylabel, series, caption, notes,
         hlines=None, vlines=None, xlim=None, ylim=None, ylog=False):
    """构造一个符合 docs/CONTRACT.md 第 4 节的 figspec（ylog 为 charts 支持的扩展）。"""
    out = {
        "name": str(name),
        "kind": str(kind),
        "title": str(title),
        "xlabel": str(xlabel),
        "ylabel": str(ylabel),
        "series": list(series or []),
        "hlines": list(hlines or []),
        "vlines": list(vlines or []),
        "ylim": None if ylim is None else [float(ylim[0]), float(ylim[1])],
        "xlim": None if xlim is None else [float(xlim[0]), float(xlim[1])],
        "caption": str(caption),
        "notes": [str(n) for n in (notes or [])],
    }
    if ylog:
        out["ylog"] = True
    return out


def _hline(y, label, style: str = "dashed"):
    return {"y": float(y), "label": str(label), "style": str(style)}


def _stats(values):
    """一维序列的汇总统计，供 sweep 系列的诊断信息使用。"""
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size == 0:
        return {"peak": 0.0, "min": 0.0, "final": 0.0, "mean": 0.0}
    return {"peak": float(arr.max()), "min": float(arr.min()),
            "final": float(arr[-1]), "mean": float(arr.mean())}


def _drift(values):
    """相对漂移：max|f(t)-f(t0)| / max(|f|, 1)。

    分母固定下限 1（与 ``references/validation.md`` 的 energy_conservation
    约定严格一致），因此当 ⟨H⟩≈0（例如 Ωz=0、|↑⟩ 时 ⟨σx⟩ 恒为 0）时，
    漂移反映的是绝对数值噪声，而不是被人为放大的相对量。
    """
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size == 0:
        return 0.0
    scale = max(float(np.abs(arr).max()), 1.0)
    return float(np.abs(arr - arr[0]).max() / scale)


def _peak_time(times, values):
    """序列峰值对应的时刻；用于 sweep 诊断。"""
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size == 0:
        return 0.0
    return float(np.asarray(times, dtype=float).ravel()[int(np.argmax(arr))])


def _check_finite(values, label: str, warnings):
    """所有对外数组必须有限；发现 NaN/Inf 时记录 warning（验证会判 fail）。"""
    arr = np.asarray(values)
    if arr.dtype.kind == "c":
        bad = not (np.all(np.isfinite(arr.real)) and np.all(np.isfinite(arr.imag)))
    elif arr.dtype.kind in "fiu":
        bad = not np.all(np.isfinite(arr))
    else:
        return
    if bad:
        warnings.append(f"{label} 中出现非有限数值（NaN/Inf），该结果不可用于结论")


# ---------------------------------------------------------------------------
# 1) spin_half
# ---------------------------------------------------------------------------
_SPIN_LABELS = ["|↑⟩", "|↓⟩"]
_SPIN_STATE_TEXT = {
    "up": "|↑⟩（σz 本征态）",
    "down": "|↓⟩",
    "plus_x": "|+x⟩=(|↑⟩+|↓⟩)/√2（σx 本征态）",
    "plus_y": "|+y⟩=(|↑⟩+i|↓⟩)/√2（σy 本征态）",
    "custom": "自定义复向量（已按给定向量的模长归一化）",
}


def _spin_state_vector(initial_state):
    """把 canonical initial_state 转成 2 分量复振幅，并保证归一化。"""
    kind = (initial_state or {}).get("kind", "up")
    if kind == "up":
        vec = np.array([1.0, 0.0], dtype=complex)
    elif kind == "down":
        vec = np.array([0.0, 1.0], dtype=complex)
    elif kind == "plus_x":
        vec = np.array([1.0, 1.0], dtype=complex) / _SQRT2
    elif kind == "plus_y":
        vec = np.array([1.0, 1.0j], dtype=complex) / _SQRT2
    elif kind == "custom":
        raw = initial_state.get("vector")
        if not raw or len(raw) != 2:
            raise ValueError("custom 初态需要 vector=[[re,im],[re,im]]")
        vec = np.array([complex(float(raw[0][0]), float(raw[0][1])),
                        complex(float(raw[1][0]), float(raw[1][1]))], dtype=complex)
        norm = float(np.linalg.norm(vec))
        if norm <= 0:
            raise ValueError("custom 初态不能是零向量")
        vec = vec / norm
    else:  # pragma: no cover - spec 已限制
        raise ValueError(f"不支持的自旋初态 kind={kind!r}")
    norm = float(np.linalg.norm(vec))
    if abs(norm - 1.0) > 1e-12:
        vec = vec / norm
    return vec


def _bloch_analytic(omega_vec, n0, t):
    """Bloch 进动解析解 n(t)=n0 cos(Ωt)+(Ω̂×n0) sin(Ωt)+Ω̂(Ω̂·n0)(1-cos Ωt)。"""
    t = np.asarray(t, dtype=float).ravel()
    om = float(np.linalg.norm(omega_vec))
    n0 = np.asarray(n0, dtype=float).ravel()
    if om <= 0.0:
        return np.tile(n0, (t.size, 1))
    u = np.asarray(omega_vec, dtype=float) / om
    cross = np.cross(u, n0)
    proj = float(np.dot(u, n0))
    c = np.cos(om * t)[:, None]
    s = np.sin(om * t)[:, None]
    return n0[None, :] * c + cross[None, :] * s + u[None, :] * proj * (1.0 - c)


def _spin_core(params, initial_state, time_block, tol, keep_states=True):
    """自旋独立求解：谱分解演化 + Bloch 解析参照 + 诊断。"""
    warnings = []
    ox = float(params["Omega_x"])
    oy = float(params["Omega_y"])
    oz = float(params["Omega_z"])
    omega_vec = np.array([ox, oy, oz], dtype=float)
    omega = float(np.linalg.norm(omega_vec))
    hbar = 1.0
    hamiltonian = (hbar / 2.0) * (ox * _SIGMA_X + oy * _SIGMA_Y + oz * _SIGMA_Z)
    eigen_values, eigen_vectors = np.linalg.eigh(hamiltonian)

    psi0 = _spin_state_vector(initial_state)
    t0 = float(time_block["t_start"])
    t1 = float(time_block["t_end"])
    n_steps = int(time_block["n_steps"])
    times = np.linspace(t0, t1, n_steps)
    dt = times - t0

    coeff = eigen_vectors.conj().T @ psi0
    phases = np.exp(-1j * np.outer(dt, eigen_values))
    states = (phases * coeff[None, :]) @ eigen_vectors.T

    amp0 = states[:, 0]
    amp1 = states[:, 1]
    pop_0 = np.abs(amp0) ** 2
    pop_1 = np.abs(amp1) ** 2
    expect_x = 2.0 * np.real(np.conj(amp0) * amp1)
    expect_y = 2.0 * np.imag(np.conj(amp0) * amp1)
    expect_z = np.abs(amp0) ** 2 - np.abs(amp1) ** 2
    energy = 0.5 * (ox * expect_x + oy * expect_y + oz * expect_z)

    # 解析参照：用初态的精确 Bloch 矢量，避免把数值舍入误差当作初值。
    n0 = np.array([
        float(2.0 * np.real(np.conj(psi0[0]) * psi0[1])),
        float(2.0 * np.imag(np.conj(psi0[0]) * psi0[1])),
        float(abs(psi0[0]) ** 2 - abs(psi0[1]) ** 2),
    ])
    n_ana = _bloch_analytic(omega_vec, n0, dt)
    pop_1_theory = 0.5 * (1.0 - n_ana[:, 2])
    analytic = {
        "name": "Bloch 进动解析解",
        "formula": "n(t)=n0 cos(|Ω|τ)+(Ω̂×n0) sin(|Ω|τ)+Ω̂(Ω̂·n0)(1-cos(|Ω|τ))，τ=t-t_start；P↓=(1-n_z)/2",
        "values": {"pop_1": pop_1_theory,
                   "expect_x": n_ana[:, 0],
                   "expect_y": n_ana[:, 1],
                   "expect_z": n_ana[:, 2]},
        "note": "解析解使用初态 Bloch 矢量与总驱动 |Ω|；相位以 t_start 为零点。",
    }

    ana_dev_candidates = [
        np.abs(pop_1 - pop_1_theory),
        np.abs(expect_x - n_ana[:, 0]),
        np.abs(expect_y - n_ana[:, 1]),
        np.abs(expect_z - n_ana[:, 2]),
    ]
    analytic_max_dev = float(max(float(np.max(c)) for c in ana_dev_candidates))

    if omega > 0.0:
        period = 2.0 * math.pi / omega
        t_dense = np.linspace(0.0, period, 4001)
        n_dense = _bloch_analytic(omega_vec, n0, t_dense)
        pop_1_peak_theory = float(np.max(0.5 * (1.0 - n_dense[:, 2])))
    else:  # 只有 sweep 才可能到达 H=0
        period = 0.0
        pop_1_peak_theory = float(0.5 * (1.0 - n0[2]))
        warnings.append("该参数点哈密顿量为零（Ω=0），没有物理演化；结果仅为恒等演化。")
    pop_1_peak_numeric = float(np.max(pop_1))
    energy_drift = _drift(energy)

    if omega > 0.0 and (t1 - t0) < period * (1.0 - 1e-9):
        warnings.append(
            f"时间窗 [{t0:g}, {t1:g}] 短于一个进动周期 2π/|Ω|={period:.6g}，"
            f"布居峰值可能没有被采样到（解析峰值按完整周期给出）。"
        )
    if analytic_max_dev > float(tol["analytic_abs"]):
        warnings.append(
            f"数值演化与 Bloch 解析解最大偏差 {analytic_max_dev:.3e} 超过容差 "
            f"{float(tol['analytic_abs']):.1e}，请检查时间网格或矩阵求解。"
        )

    max_flip_prob_theory = float((ox * ox + oy * oy) / (omega * omega)) if omega > 0 else 0.0
    scalars = {
        "Omega_eff": omega,
        "E_plus": 0.5 * omega,
        "E_minus": -0.5 * omega,
        "max_flip_prob_theory": max_flip_prob_theory,
        "period": period,
        "pop_1_peak_numeric": pop_1_peak_numeric,
        "pop_1_peak_theory": pop_1_peak_theory,
        "pop_1_peak_time": _peak_time(times, pop_1),
        "analytic_max_abs_dev": analytic_max_dev,
        "energy_mean": float(np.mean(energy)),
        "energy_drift": energy_drift,
    }
    metrics = {
        "analytic_max_abs_dev": analytic_max_dev,
        "energy_drift": energy_drift,
        "pop_1_peak_numeric": pop_1_peak_numeric,
    }
    observables = {
        "pop_0": pop_0,
        "pop_1": pop_1,
        "expect_x": expect_x,
        "expect_y": expect_y,
        "expect_z": expect_z,
        "energy": energy,
        "pop_1_theory": pop_1_theory,
    }
    for name, values in observables.items():
        _check_finite(values, f"spin_half 观察量 {name}", warnings)

    return {
        "params": {"Omega_x": ox, "Omega_y": oy, "Omega_z": oz},
        "hamiltonian": hamiltonian,
        "eigen": {"values": eigen_values, "vectors": eigen_vectors},
        "times": times,
        "states": states if keep_states else None,
        "observables": observables,
        "analytic": analytic,
        "scalars": scalars,
        "metrics": metrics,
        "warnings": warnings,
    }


def _spin_sweep(req):
    """对 Ω 的某一个分量扫描；每个采样点独立求解并记录诊断摘要。"""
    sweep = req["sweep"]
    pname = sweep["param"]
    values = [float(v) for v in sweep["values"]]
    base = dict(req["params"])
    time_block = req["time"]
    tol = req["tolerance"]
    series = []
    warnings = []
    for value in values:
        params = dict(base)
        params[pname] = float(value)
        try:
            core = _spin_core(params, req["initial_state"], time_block, tol,
                              keep_states=False)
        except ValueError as exc:
            warnings.append(f"扫描点 {pname}={value:g} 无法求解：{exc}")
            series.append({"value": float(value),
                           "scalars": {"valid": 0.0},
                           "observables": {}})
            continue
        warnings.extend(core["warnings"])
        scalars = dict(core["scalars"])
        scalars["valid"] = 1.0
        series.append({
            "value": float(value),
            "scalars": scalars,
            "observables": {
                "pop_0": _stats(core["observables"]["pop_0"]),
                "pop_1": _stats(core["observables"]["pop_1"]),
                "expect_z": _stats(core["observables"]["expect_z"]),
            },
        })
    return {
        "param": pname,
        "values": values,
        "series": series,
        "note": ("每个采样点独立求解（初态与时间网格相同）；scalars/observables "
                 "是该点的汇总统计。结论只对采样点成立。"),
    }, warnings


def _spin_figures(core, sweep):
    """自旋模型的声明式图：布居、Bloch 期望值、能量、解析对比、扫描叠加。"""
    times = core["times"]
    obs = core["observables"]
    scalars = core["scalars"]
    t0 = float(times[0])
    t1 = float(times[-1])
    notes = [
        f"参数：Ωx={core['params']['Omega_x']:g}，Ωy={core['params']['Omega_y']:g}，"
        f"Ωz={core['params']['Omega_z']:g}；|Ω|={scalars['Omega_eff']:.6g}（ħ=1）。",
        f"时间窗 [{t0:g}, {t1:g}]，共 {len(times)} 点；图中曲线为等间隔抽稀显示，"
        f"验证使用全部采样点。",
    ]
    figs = [
        _fig("population", "line", "自旋布居数随时间的演化（拉比振荡）",
             "时间 t", "布居数",
             [_series("P↑=|⟨↑|ψ⟩|²", times, obs["pop_0"], "line"),
              _series("P↓=|⟨↓|ψ⟩|²", times, obs["pop_1"], "line"),
              _series("P↓ 解析（Bloch）", times, obs["pop_1_theory"], "dashed")],
             "P↓ 峰值为 Ω⊥²/|Ω|²（Ω⊥²=Ωx²+Ωy²）；纵向分量 Ωz 使最大翻转概率小于 1。",
             notes),
        _fig("bloch_expectation", "line", "Bloch 矢量期望值 ⟨σx⟩、⟨σy⟩、⟨σz⟩ 的进动",
             "时间 t", "⟨σ⟩",
             [_series("⟨σx⟩", times, obs["expect_x"], "line"),
              _series("⟨σy⟩", times, obs["expect_y"], "line"),
              _series("⟨σz⟩", times, obs["expect_z"], "line")],
             "Bloch 矢量以角速度矢量 Ω 进动；|n| 保持为 1（纯态）。", notes),
        _fig("energy", "line", "能量期望值 ⟨H⟩ 随时间守恒",
             "时间 t", "⟨H⟩",
             [_series("⟨H⟩(t)", times, obs["energy"], "line")],
             "封闭系统幺正演化下 ⟨H⟩ 应严格守恒，数值漂移反映求解器精度。",
             notes,
             hlines=[_hline(scalars["E_plus"], "E+ = |Ω|/2"),
                     _hline(scalars["E_minus"], "E− = -|Ω|/2")]),
        _fig("analytic_compare", "error", "数值演化与 Bloch 解析解的偏差",
             "时间 t", "偏差",
             [_series("P↓ 数值 − 解析", times, obs["pop_1"] - obs["pop_1_theory"], "line")],
             f"最大绝对偏差 {scalars['analytic_max_abs_dev']:.3e}；量级应接近机器精度。",
             notes),
    ]
    if sweep:
        valid = [item for item in sweep["series"] if item["scalars"].get("valid", 0.0) > 0.5]
        if valid:
            x = [item["value"] for item in valid]
            figs.append(_fig(
                "sweep_overlay", "line",
                f"扫描 {sweep['param']}：P↓ 峰值与解析峰值",
                f"扫描参数 {sweep['param']}", "P↓ 峰值",
                [_series("数值峰值", x,
                         [item["scalars"]["pop_1_peak_numeric"] for item in valid], "line"),
                 _series("解析峰值", x,
                         [item["scalars"]["pop_1_peak_theory"] for item in valid], "dashed")],
                "每个扫描点独立求解；峰值按完整周期解析给出，数值峰值依赖时间网格。",
                [f"扫描参数 {sweep['param']}，共 {len(sweep['values'])} 点，"
                 f"其中有效点 {len(valid)} 个。",
                 f"初态与时间窗固定，能量单位 ħ=1。"]))
    return figs


def _run_spin(req):
    """执行自旋模型的完整流程（求解 → 扫描 → 汇总 → 图描述）。"""
    params = req["params"]
    tol = req["tolerance"]
    core = _spin_core(params, req["initial_state"], req["time"], tol,
                      keep_states=True)
    sweep, sweep_warnings = (None, [])
    if req.get("sweep"):
        sweep, sweep_warnings = _spin_sweep(req)
    warnings = list(req.get("warnings") or []) + core["warnings"] + sweep_warnings

    scalars = core["scalars"]
    state_text = _SPIN_STATE_TEXT.get((req["initial_state"] or {}).get("kind", "up"),
                                      "自定义初态")
    bullets = [
        f"等效拉比频率 |Ω|=√(Ωx²+Ωy²+Ωz²)={scalars['Omega_eff']:.6g}，"
        f"能级 E±=±{scalars['E_plus']:.6g}，完整进动周期 2π/|Ω|={scalars['period']:.6g}。",
        f"初态 {state_text}；时间窗 [{float(core['times'][0]):g}, "
        f"{float(core['times'][-1]):g}]（{len(core['times'])} 点），相位以 t_start 为零点。",
        f"P↓ 数值峰值 {scalars['pop_1_peak_numeric']:.6g}，解析峰值 "
        f"{scalars['pop_1_peak_theory']:.6g}；从 |↑⟩ 出发的理论上限 "
        f"Ω⊥²/|Ω|²={scalars['max_flip_prob_theory']:.6g}。",
        f"Bloch 解析解最大绝对偏差 {scalars['analytic_max_abs_dev']:.3e}；"
        f"能量期望 ⟨H⟩={scalars['energy_mean']:.6g} 相对漂移 "
        f"{scalars['energy_drift']:.2e}。",
    ]
    result = {
        "model": "spin_half",
        "params": core["params"],
        "hamiltonian": core["hamiltonian"],
        "basis": {"labels": list(_SPIN_LABELS), "dim": 2, "truncated": False,
                  "note": "σz 本征基 |↑⟩=(1,0)ᵀ、|↓⟩=(0,1)ᵀ；ħ=1"},
        "eigen": core["eigen"],
        "times": core["times"],
        "states": core["states"],
        "observables": core["observables"],
        "scalars": scalars,
        "analytic": core["analytic"],
        "sweep": sweep,
        "metrics": core["metrics"],
        "warnings": warnings,
        "summary": {"title": "自旋 1/2 二能级系统的拉比振荡",
                    "bullets": bullets},
    }
    result["figure_specs"] = _spin_figures(core, sweep)
    return result


# ---------------------------------------------------------------------------
# 2) harmonic_oscillator
# ---------------------------------------------------------------------------
def _ho_operators(n_levels):
    """返回 Fock 截断空间中的湮灭算符 a（a[n-1,n]=√n）及其共轭。"""
    n = int(n_levels)
    a = np.zeros((n, n), dtype=complex)
    for k in range(1, n):
        a[k - 1, k] = math.sqrt(float(k))
    return a, a.conj().T


def _ho_initial_state(initial_state, n_levels):
    """构造谐振子初态系数 c（长度 N，已归一化），并给出截断诊断。"""
    n_levels = int(n_levels)
    state = initial_state or {}
    kind = state.get("kind", "superposition")
    coeff = np.zeros(n_levels, dtype=complex)
    tail = 0.0
    note = ""
    if kind == "fock":
        n = int(state.get("n", 0))
        if n >= n_levels:
            raise ValueError(
                f"初态 |{n}⟩ 超出截断能级范围 n=0…{n_levels - 1}；"
                f"请增大 n_levels（当前 {n_levels}）或改用更低的 n。")
        coeff[n] = 1.0
        note = f"Fock 本征态 |{n}⟩"
    elif kind == "coherent":
        alpha = state.get("alpha") or [1.0, 0.0]
        z = complex(float(alpha[0]), float(alpha[1]))
        coeff[0] = complex(math.exp(-0.5 * abs(z) ** 2), 0.0)
        for n in range(1, n_levels):
            coeff[n] = coeff[n - 1] * z / math.sqrt(float(n))
        norm2 = float(np.sum(np.abs(coeff) ** 2))
        tail = max(0.0, 1.0 - norm2)
        if norm2 <= 0.0:
            raise ValueError(
                f"相干态 α={z} 在 N={n_levels} 的截断下振幅全部下溢为 0；"
                f"请增大 n_levels 或减小 |α|。")
        coeff = coeff / math.sqrt(norm2)
        note = (f"相干态 |α⟩，α={z.real:g}{z.imag:+g}j；Fock 展开后在 "
                f"N={n_levels} 内重新归一化，截断尾部概率 {tail:.3e}")
    elif kind == "superposition":
        components = state.get("components") or []
        if not components:
            raise ValueError("superposition 初态需要非空 components")
        for comp in components:
            n = int(comp.get("n", 0))
            if n >= n_levels:
                raise ValueError(
                    f"叠加态分量 |{n}⟩ 超出截断能级范围 n=0…{n_levels - 1}；"
                    f"请增大 n_levels 或改用更低的能级。")
            amp = comp.get("amplitude") or [1.0, 0.0]
            coeff[n] += complex(float(amp[0]), float(amp[1]))
        norm2 = float(np.sum(np.abs(coeff) ** 2))
        if norm2 <= 0.0:
            raise ValueError("叠加态初态不能全为零振幅")
        coeff = coeff / math.sqrt(norm2)
        note = "有限 Fock 叠加态（已在截断空间内归一化）"
    else:  # pragma: no cover - spec 已限制
        raise ValueError(f"不支持的谐振子初态 kind={kind!r}")
    return coeff, float(tail), note


def _ho_solve(params, initial_state, time_block, tol, keep_states=True):
    """谐振子独立求解：谱分解演化、观察量、解析 <x>/<p> 与截断诊断。"""
    n_levels = int(params["n_levels"])
    omega = float(params["omega"])
    warnings = []

    a, a_dag = _ho_operators(n_levels)
    hamiltonian = omega * (a_dag @ a + 0.5 * np.eye(n_levels, dtype=complex))
    eigen_values, eigen_vectors = np.linalg.eigh(hamiltonian)

    coeff0, truncation_tail, state_note = _ho_initial_state(initial_state, n_levels)
    t0 = float(time_block["t_start"])
    t1 = float(time_block["t_end"])
    n_steps = int(time_block["n_steps"])
    times = np.linspace(t0, t1, n_steps)
    dt = times - t0
    coeff_eig = eigen_vectors.conj().T @ coeff0

    n_index = np.arange(n_levels, dtype=float)
    sqrt_shift = np.sqrt(np.arange(1, n_levels, dtype=float))
    pop_count = min(4, n_levels)
    buckets = {f"pop_{k}": [] for k in range(pop_count)}
    buckets.update({"expect_x": [], "expect_p": [], "expect_n": [],
                    "boundary_pop": [], "energy": []})

    states = np.empty((n_steps, n_levels), dtype=complex) if keep_states else None
    # 分块演化，避免 n_steps×N 很大时一次性分配过多中间量。
    chunk = max(1, min(n_steps, 500_000 // max(n_levels, 1)))
    for start in range(0, n_steps, chunk):
        stop = min(start + chunk, n_steps)
        local_dt = dt[start:stop]
        phases = np.exp(-1j * np.outer(local_dt, eigen_values))
        block = (phases * coeff_eig[None, :]) @ eigen_vectors.T
        if keep_states:
            states[start:stop] = block
        pop = np.abs(block) ** 2
        for k in range(pop_count):
            buckets[f"pop_{k}"].append(pop[:, k])
        # A(t)=Σ √(n+1) c_n* c_{n+1} = <a>；<x>=√2 Re A，<p>=√2 Im A。
        a_expect = np.sum(sqrt_shift[None, :] * np.conj(block[:, :-1]) * block[:, 1:],
                          axis=1)
        buckets["expect_x"].append(_SQRT2 * np.real(a_expect))
        buckets["expect_p"].append(_SQRT2 * np.imag(a_expect))
        expect_n = np.sum(pop * n_index[None, :], axis=1)
        buckets["expect_n"].append(expect_n)
        buckets["boundary_pop"].append(pop[:, -1])
        buckets["energy"].append(omega * (expect_n + 0.5))
    observables = {key: np.concatenate(value) for key, value in buckets.items()}

    # 解析参照：A0 = Σ √(n+1) c_n* c_{n+1}，A(t)=A0 e^{-iω(t-t0)}。
    a0 = complex(np.sum(sqrt_shift * np.conj(coeff0[:-1]) * coeff0[1:]))
    phase = np.exp(-1j * omega * dt)
    expect_x_theory = _SQRT2 * np.real(a0 * phase)
    expect_p_theory = _SQRT2 * np.imag(a0 * phase)
    observables["expect_x_theory"] = expect_x_theory
    observables["expect_p_theory"] = expect_p_theory
    analytic = {
        "name": "有限叠加态的 <x>(t)、<p>(t) 解析解",
        "formula": "⟨x⟩(t)=√2 Re[A₀e^{-iωτ}]，⟨p⟩(t)=√2 Im[A₀e^{-iωτ}]，"
                   "A₀=Σₙ cₙ*cₙ₊₁√(n+1)，τ=t-t_start",
        "values": {"expect_x": expect_x_theory, "expect_p": expect_p_theory},
        "note": f"A₀={a0.real:.6g}{a0.imag:+g}j；解析解使用同一 N 维截断内的初态系数。",
    }
    analytic_max_dev = float(max(
        np.max(np.abs(observables["expect_x"] - expect_x_theory)),
        np.max(np.abs(observables["expect_p"] - expect_p_theory)),
    ))
    spectrum_theory = omega * (np.arange(n_levels, dtype=float) + 0.5)
    spectrum_max_dev = float(np.max(np.abs(eigen_values - spectrum_theory)))
    boundary_pop_max = float(np.max(observables["boundary_pop"]))
    energy_drift = _drift(observables["energy"])
    x_amplitude = float(np.max(np.abs(observables["expect_x"])))

    if truncation_tail > float(tol["truncation_pop"]):
        warnings.append(
            f"相干态 Fock 展开的截断尾部概率 {truncation_tail:.3e} 超过容差 "
            f"{float(tol['truncation_pop']):.1e}：N={n_levels} 不足以表示该 |α|，"
            f"请增大 n_levels 或减小 |α|。")
    if boundary_pop_max > float(tol["truncation_pop"]):
        warnings.append(
            f"边界能级 |{n_levels - 1}⟩ 的最大占据 {boundary_pop_max:.3e} 超过容差 "
            f"{float(tol['truncation_pop']):.1e}：截断可能不足，建议增大 n_levels；"
            f"这属于模型截断限制，不等于数值发散。")
    if analytic_max_dev > float(tol["analytic_abs"]):
        warnings.append(
            f"<x>/<p> 与解析解最大偏差 {analytic_max_dev:.3e} 超过容差 "
            f"{float(tol['analytic_abs']):.1e}。")

    scalars = {
        "omega": omega,
        "n_levels": float(n_levels),
        "E0": 0.5 * omega,
        "E1": 1.5 * omega,
        "spectrum": [float(v) for v in eigen_values],
        "energy_mean": float(np.mean(observables["energy"])),
        "energy_drift": energy_drift,
        "expect_n_mean": float(np.mean(observables["expect_n"])),
        "x_amplitude": x_amplitude,
        "boundary_pop_max": boundary_pop_max,
        "truncation_tail_prob": float(truncation_tail),
        "analytic_max_abs_dev": analytic_max_dev,
        "spectrum_max_abs_dev": spectrum_max_dev,
    }
    metrics = {
        "analytic_max_abs_dev": analytic_max_dev,
        "spectrum_max_abs_dev": spectrum_max_dev,
        "boundary_pop_max": boundary_pop_max,
        "truncation_tail_prob": float(truncation_tail),
        "energy_drift": energy_drift,
    }
    for name, values in observables.items():
        _check_finite(values, f"harmonic_oscillator 观察量 {name}", warnings)

    return {
        "params": {"omega": omega, "n_levels": n_levels},
        "hamiltonian": hamiltonian,
        "eigen": {"values": eigen_values, "vectors": eigen_vectors},
        "times": times,
        "states": states,
        "observables": observables,
        "analytic": analytic,
        "scalars": scalars,
        "metrics": metrics,
        "warnings": warnings,
        "state_note": state_note,
    }


def _ho_sweep(req):
    """对 ω 或 n_levels 扫描；记录截断诊断、解析偏差等分组摘要。"""
    sweep = req["sweep"]
    pname = sweep["param"]
    values = sweep["values"]
    base = dict(req["params"])
    time_block = req["time"]
    tol = req["tolerance"]
    series = []
    warnings = []
    for value in values:
        params = dict(base)
        if pname == "n_levels":
            params["n_levels"] = int(round(float(value)))
        else:
            params[pname] = float(value)
        try:
            core = _ho_solve(params, req["initial_state"], time_block, tol,
                             keep_states=False)
        except ValueError as exc:
            warnings.append(f"扫描点 {pname}={value:g} 无法求解：{exc}")
            series.append({"value": float(value), "scalars": {"valid": 0.0},
                           "observables": {}})
            continue
        warnings.extend(core["warnings"])
        scalars = dict(core["scalars"])
        scalars["valid"] = 1.0
        series.append({
            "value": float(value),
            "scalars": scalars,
            "observables": {
                "expect_x": _stats(core["observables"]["expect_x"]),
                "expect_n": _stats(core["observables"]["expect_n"]),
                "boundary_pop": _stats(core["observables"]["boundary_pop"]),
            },
        })
    return {
        "param": pname,
        "values": [float(v) for v in values],
        "series": series,
        "note": ("每个采样点独立求解；scalars 中的截断尾部概率、边界最大占据与"
                 "解析偏差用于经验收敛诊断——边界泄漏小不是收敛性证明。"),
    }, warnings


def _ho_figures(core, sweep):
    """谐振子模型的声明式图。"""
    times = core["times"]
    obs = core["observables"]
    scalars = core["scalars"]
    n_levels = int(core["params"]["n_levels"])
    notes = [
        f"参数：ω={scalars['omega']:g}，N={n_levels}（n=0…{n_levels - 1}）；ħ=1。",
        f"时间窗 [{float(times[0]):g}, {float(times[-1]):g}]，共 {len(times)} 点；"
        f"图中曲线为等间隔抽稀显示，验证使用全部采样点。",
    ]
    pop_series = [_series(f"|n={k}⟩ 占据", times, obs[f"pop_{k}"], "line")
                  for k in range(min(4, n_levels))]
    figs = [
        _fig("population", "line", "低能级占据随时间的演化", "时间 t", "占据概率",
             pop_series,
             "在固定 Fock 截断内，占据概率之和为 1（总概率可由态范数验证）。", notes),
        _fig("position", "line", "位置期望值 ⟨x⟩(t) 与解析解对比",
             "时间 t", "⟨x⟩",
             [_series("⟨x⟩ 数值", times, obs["expect_x"], "line"),
              _series("⟨x⟩ 解析", times, obs["expect_x_theory"], "dashed")],
             f"最大偏差 {scalars['analytic_max_abs_dev']:.3e}；解析式 "
             "⟨x⟩=√2 Re[A₀e^(-iωt)]。", notes),
        _fig("momentum", "line", "动量期望值 ⟨p⟩(t)", "时间 t", "⟨p⟩",
             [_series("⟨p⟩ 数值", times, obs["expect_p"], "line"),
              _series("⟨p⟩ 解析", times, obs["expect_p_theory"], "dashed")],
             "⟨p⟩ 与 ⟨x⟩ 相位相差 π/2，体现谐振子的简谐振荡。", notes),
        _fig("occupation", "line", "平均能级占据 ⟨n⟩(t)", "时间 t", "⟨n⟩",
             [_series("⟨n⟩", times, obs["expect_n"], "line")],
             "对能量本征态 ⟨n⟩ 为常数；对叠加态/相干态可随时间变化。", notes),
        _fig("energy", "line", "能量期望值 ⟨H⟩ 随时间守恒", "时间 t", "⟨H⟩",
             [_series("⟨H⟩(t)", times, obs["energy"], "line")],
             "同一截断子空间内 H 不显含时间，⟨H⟩ 应守恒（相对漂移反映数值精度）。",
             notes,
             hlines=[_hline(scalars["E0"], "E₀=ω/2"),
                     _hline(scalars["E1"], "E₁=3ω/2")]),
        _fig("boundary", "line", "截断边界能级占据（诊断量）", "时间 t",
             f"|c_{{{n_levels - 1}}}|²",
             [_series(f"最高能级 |{n_levels - 1}⟩ 占据", times,
                      obs["boundary_pop"], "line")],
             "边界占据小只说明当前条件下来回泄漏小；**不构成收敛性证明**。", notes,
             hlines=[_hline(float(1e-6), "截断参考 1e-6")],
             ylog=True),
        _fig("spectrum", "spectrum", "能谱：数值对角化 vs 解析 Eₙ=ω(n+½)",
             "能级编号 n", "本征能量 Eₙ",
             [_series("eigh 本征值", np.arange(n_levels), core["eigen"]["values"], "dots"),
              _series("ω(n+½) 解析", np.arange(n_levels),
                      scalars["omega"] * (np.arange(n_levels) + 0.5), "dashed")],
             f"逐项最大偏差 {scalars['spectrum_max_abs_dev']:.3e}（容差 1e-8 量级）。",
             notes),
        _fig("analytic_compare", "error", "数值 ⟨x⟩ 与解析解的偏差", "时间 t", "偏差",
             [_series("⟨x⟩ 数值 − 解析", times,
                      obs["expect_x"] - obs["expect_x_theory"], "line")],
             f"最大绝对偏差 {scalars['analytic_max_abs_dev']:.3e}。", notes),
    ]
    if sweep:
        valid = [item for item in sweep["series"] if item["scalars"].get("valid", 0.0) > 0.5]
        if valid:
            x = [item["value"] for item in valid]
            if sweep["param"] == "n_levels":
                figs.append(_fig(
                    "sweep_overlay", "line",
                    "扫描截断能级数 N：截断诊断与解析偏差",
                    "截断能级数 N", "诊断量（对数）",
                    [_series("边界能级最大占据", x,
                             [item["scalars"]["boundary_pop_max"] for item in valid],
                             "line"),
                     _series("相干态截断尾部概率", x,
                             [max(item["scalars"]["truncation_tail_prob"], 1e-300)
                              for item in valid], "dashed"),
                     _series("⟨x⟩ 解析最大偏差", x,
                             [max(item["scalars"]["analytic_max_abs_dev"], 1e-300)
                              for item in valid], "dots")],
                    "随 N 增大，边界泄漏与截断尾部概率应下降；这是经验收敛检查，"
                    "不是严格收敛证明。",
                    [f"扫描 n_levels 共 {len(sweep['values'])} 点，有效 {len(valid)} 点。",
                     "对数纵轴下限 1e-16。"], ylog=True))
            else:
                figs.append(_fig(
                    "sweep_overlay", "line",
                    "扫描 ω：⟨x⟩ 振幅与平均能量",
                    "角频率 ω", "汇总量",
                    [_series("⟨x⟩ 振幅 (max|⟨x⟩|)", x,
                             [item["scalars"]["x_amplitude"] for item in valid], "line"),
                     _series("平均能量 ⟨H⟩", x,
                             [item["scalars"]["energy_mean"] for item in valid], "dashed")],
                    "相干态 ⟨x⟩ 振幅理论值 √2|α| 与 ω 无关，可用该扫描做交叉检查。",
                    [f"扫描 omega 共 {len(sweep['values'])} 点，有效 {len(valid)} 点。"]))
    return figs


def _run_ho(req):
    """执行谐振子模型的完整流程。"""
    params = req["params"]
    tol = req["tolerance"]
    core = _ho_solve(params, req["initial_state"], req["time"], tol,
                     keep_states=True)
    sweep, sweep_warnings = (None, [])
    if req.get("sweep"):
        sweep, sweep_warnings = _ho_sweep(req)
    warnings = list(req.get("warnings") or []) + core["warnings"] + sweep_warnings

    scalars = core["scalars"]
    state_note = core["state_note"]
    bullets = [
        f"谐振子 ω={scalars['omega']:g}，Fock 截断 N={int(scalars['n_levels'])}"
        f"（n=0…{int(scalars['n_levels']) - 1}），Eₙ=ω(n+½)，E₀={scalars['E0']:.6g}。",
        f"初态：{state_note}。",
        f"⟨H⟩={scalars['energy_mean']:.6g}（相对漂移 {scalars['energy_drift']:.2e}），"
        f"⟨n⟩ 均值 {scalars['expect_n_mean']:.6g}，⟨x⟩ 振幅 {scalars['x_amplitude']:.6g}。",
        f"⟨x⟩/⟨p⟩ 与解析式最大偏差 {scalars['analytic_max_abs_dev']:.3e}；"
        f"能谱与 ω(n+½) 最大偏差 {scalars['spectrum_max_abs_dev']:.3e}。",
        f"截断诊断：边界能级最大占据 {scalars['boundary_pop_max']:.3e}，"
        f"相干态展开尾部概率 {scalars['truncation_tail_prob']:.3e}；"
        f"占据小只说明当前条件下泄漏小，不是收敛性证明。",
    ]
    result = {
        "model": "harmonic_oscillator",
        "params": core["params"],
        "hamiltonian": core["hamiltonian"],
        "basis": {"labels": [f"|{k}⟩" for k in range(int(scalars["n_levels"]))],
                  "dim": int(scalars["n_levels"]), "truncated": True,
                  "note": f"Fock 截断 n=0…{int(scalars['n_levels']) - 1}；"
                          f"ħ=1，Eₙ=ω(n+½)"},
        "eigen": core["eigen"],
        "times": core["times"],
        "states": core["states"],
        "observables": core["observables"],
        "scalars": scalars,
        "analytic": core["analytic"],
        "sweep": sweep,
        "metrics": core["metrics"],
        "warnings": warnings,
        "summary": {"title": "量子谐振子的能谱与含时演化（有限 Fock 截断）",
                    "bullets": bullets},
    }
    result["figure_specs"] = _ho_figures(core, sweep)
    return result


# ---------------------------------------------------------------------------
# 3) perturbation
# ---------------------------------------------------------------------------
def _perturbation_theory(delta, lam):
    """非简并微扰的二阶展开：返回精确值、一阶、二阶与误差指标。"""
    delta = float(delta)
    lam = float(lam)
    exact0 = -0.5 * math.sqrt(delta * delta + lam * lam)
    exact1 = +0.5 * math.sqrt(delta * delta + lam * lam)
    pt1 = -0.5 * delta
    pt2 = pt1 - lam * lam / (4.0 * delta)
    abs_err = abs(pt2 - exact0)
    rel_err = abs_err / abs(exact0) if exact0 != 0.0 else 0.0
    return {
        "E0_exact": exact0,
        "E1_exact": exact1,
        "E0_pt1": pt1,
        "E0_pt2": pt2,
        "abs_err": abs_err,
        "rel_err": rel_err,
    }


def _perturbation_core(params, sweep_values, tol):
    """微扰模型：精确对角化 + 闭式解交叉校验 + 二阶微扰误差与适用范围。"""
    delta = float(params["delta"])
    lam = float(params["lambda"])
    warnings = []
    hamiltonian = 0.5 * delta * _SIGMA_Z + 0.5 * lam * _SIGMA_X
    eigen_values, eigen_vectors = np.linalg.eigh(hamiltonian)

    theory = _perturbation_theory(delta, lam)
    solver_dev = float(np.max(np.abs(eigen_values - np.array(
        [theory["E0_exact"], theory["E1_exact"]]))))
    if solver_dev > float(tol["analytic_abs"]):
        warnings.append(
            f"精确对角化与闭式解 -½√(Δ²+λ²) 的最大偏差 {solver_dev:.3e} 超过容差 "
            f"{float(tol['analytic_abs']):.1e}：这是求解器层面的问题。")

    # λ 扫描：找出满足相对误差阈值的最大 λ（只对采样点成立）。
    lambda_valid_max = None
    series = []
    in_scope_flags = []
    for value in sweep_values:
        point = _perturbation_theory(delta, value)
        within = 1.0 if point["rel_err"] <= float(tol["perturbation_rel"]) else 0.0
        in_scope_flags.append(bool(within))
        series.append({
            "value": float(value),
            "scalars": {
                "lambda": float(value),
                "E0_pt1": point["E0_pt1"],
                "E0_pt2": point["E0_pt2"],
                "E0_exact": point["E0_exact"],
                "E1_exact": point["E1_exact"],
                "abs_err": point["abs_err"],
                "rel_err": point["rel_err"],
                "within_threshold": within,
            },
            "observables": {},
        })
    for value, within in zip(sweep_values, in_scope_flags):
        if within:
            lambda_valid_max = float(value) if lambda_valid_max is None else max(
                lambda_valid_max, float(value))
    if lambda_valid_max is None:
        warnings.append(
            "λ 扫描中没有任何采样点满足二阶微扰相对误差阈值，"
            "说明给出的 λ/Δ 区间完全超出二阶近似适用范围。")
    elif lambda_valid_max < float(max(sweep_values)):
        warnings.append(
            f"λ 扫描中仅有 λ≤{lambda_valid_max:g} 的采样点满足相对误差阈值 "
            f"{float(tol['perturbation_rel']):.1e}；更大 λ 时二阶近似偏离精确值，"
            f"这是近似适用范围限制而非数值错误。")

    scalars = {
        "E0_exact": theory["E0_exact"],
        "E1_exact": theory["E1_exact"],
        "E0_pt1": theory["E0_pt1"],
        "E0_pt2": theory["E0_pt2"],
        "abs_err": theory["abs_err"],
        "rel_err": theory["rel_err"],
        "within_threshold": 1.0 if theory["rel_err"] <= float(tol["perturbation_rel"]) else 0.0,
        "solver_closed_form_dev": solver_dev,
    }
    if lambda_valid_max is not None:
        scalars["lambda_valid_max"] = float(lambda_valid_max)
    metrics = {
        "solver_closed_form_dev": solver_dev,
        "perturbation_rel_err": theory["rel_err"],
    }
    analytic = {
        "name": "精确能谱闭式解",
        "formula": "E± = ±½√(Δ²+λ²)",
        "values": {"E0_exact": theory["E0_exact"], "E1_exact": theory["E1_exact"]},
        "note": f"Δ={delta:g}，λ={lam:g}；一阶 E0⁽¹⁾=-Δ/2，二阶 E0⁽²⁾=-λ²/(4Δ)。",
    }
    sweep = {
        "param": "lambda",
        "values": [float(v) for v in sweep_values],
        "series": series,
        "note": ("每点用 Δ 固定、λ 变化重新计算精确对角化与二阶微扰；"
                 "within_threshold=1 表示该点 rel_err ≤ tolerance.perturbation_rel。"
                 "适用范围只对采样点与当前 Δ 成立。"),
    }
    return {
        "params": {"delta": delta, "lambda": lam},
        "hamiltonian": hamiltonian,
        "eigen": {"values": eigen_values, "vectors": eigen_vectors},
        "observables": {},
        "analytic": analytic,
        "scalars": scalars,
        "metrics": metrics,
        "sweep": sweep,
        "warnings": warnings,
        "lambda_valid_max": lambda_valid_max,
    }


def _perturbation_figures(core):
    """微扰模型的声明式图：能谱对比、λ 扫描叠加、相对误差（对数轴）。"""
    scalars = core["scalars"]
    delta = core["params"]["delta"]
    lam = core["params"]["lambda"]
    sweep = core["sweep"]
    values = sweep["values"]
    series = sweep["series"]
    exact = [item["scalars"]["E0_exact"] for item in series]
    pt1 = [item["scalars"]["E0_pt1"] for item in series]
    pt2 = [item["scalars"]["E0_pt2"] for item in series]
    rel = [max(item["scalars"]["rel_err"], 1e-18) for item in series]
    valid_max = scalars.get("lambda_valid_max")
    notes = [
        f"参数：Δ={delta:g}，λ={lam:g}；H=(Δ/2)σz+(λ/2)σx，ħ=1。",
        f"扫描 λ 共 {len(values)} 点（{min(values):g} … {max(values):g}）；"
        f"相对误差阈值 {1e-2:g}。",
    ]
    figs = [
        _fig("spectrum", "spectrum", "基态能量：精确对角化 vs 二阶微扰",
             "能级编号（0=基态，1=激发态）", "本征能量 E",
             [_series("eigh 精确本征值", [0.0, 1.0], core["eigen"]["values"], "dots"),
              _series("闭式解 ±½√(Δ²+λ²)", [0.0, 1.0],
                      [scalars["E0_exact"], scalars["E1_exact"]], "dashed")],
             f"精确对角化与闭式解最大偏差 "
             f"{scalars['solver_closed_form_dev']:.3e}（应接近机器精度）。", notes),
        _fig("sweep_overlay", "line", "扫描 λ：基态能量的精确值与微扰近似",
             "扰动强度 λ", "基态能量 E₀",
             [_series("E₀ 精确", values, exact, "line"),
              _series("E₀⁽¹⁾ 一阶", values, pt1, "dashed"),
              _series("E₀⁽²⁾ 二阶", values, pt2, "dots")],
             "λ/Δ 增大时二阶近似逐渐偏离精确值；一阶结果因 V 无对角元而不随 λ 变化。",
             notes),
        _fig("perturbation_error", "error", "二阶微扰的相对误差随 λ 的变化",
             "扰动强度 λ", "|E₀⁽²⁾−E₀|/|E₀|（对数）",
             [_series("相对误差", values, rel, "line")],
             "相对误差按 λ² 增长；阈值内的区域才可以使用二阶微扰结论。",
             notes + [
                 (f"满足阈值的最大采样 λ = {valid_max:g}" if valid_max is not None
                  else "没有采样点满足阈值。"),
                 "对数纵轴下限 1e-16。",
             ],
             hlines=[_hline(1e-2, "容差 1e-2")], ylog=True),
    ]
    return figs


def _run_perturbation(req):
    """执行微扰模型的完整流程。"""
    tol = req["tolerance"]
    sweep_req = req.get("sweep")
    if sweep_req:
        values = [float(v) for v in sweep_req["values"]]
    else:  # spec 对 perturbation 默认给出扫描；这里防御性回退。
        values = [round(0.01 * i, 4) for i in range(1, 26)]
        values += [0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0]
    core = _perturbation_core(req["params"], values, tol)
    warnings = list(req.get("warnings") or []) + core["warnings"]

    scalars = core["scalars"]
    delta = core["params"]["delta"]
    lam = core["params"]["lambda"]
    valid_max = core["lambda_valid_max"]
    bullets = [
        f"H=(Δ/2)σz+(λ/2)σx，Δ={delta:g}，λ={lam:g}：精确基态 "
        f"E₀=-½√(Δ²+λ²)={scalars['E0_exact']:.12g}。",
        f"一阶 E₀⁽¹⁾=-Δ/2={scalars['E0_pt1']:.12g}（V 无对角元），"
        f"二阶 E₀⁽²⁾=-Δ/2-λ²/(4Δ)={scalars['E0_pt2']:.12g}；"
        f"与精确值相差 {scalars['abs_err']:.3e}（相对 {scalars['rel_err']:.3e}）。",
        f"精确对角化与闭式解最大偏差 {scalars['solver_closed_form_dev']:.3e}"
        f"（求解器交叉校验）。",
        (f"λ 扫描 {len(values)} 点：满足二阶微扰相对误差阈值（1e-2）的最大采样 "
         f"λ={valid_max:g}；结论只对当前 Δ 与这些采样点成立。"
         if valid_max is not None else
         f"λ 扫描 {len(values)} 点：没有采样点满足二阶微扰相对误差阈值，"
         f"当前 λ 区间完全超出二阶近似适用范围（物理预期，非数值错误）。"),
    ]
    result = {
        "model": "perturbation",
        "params": core["params"],
        "hamiltonian": core["hamiltonian"],
        "basis": {"labels": ["|0⟩=|↓⟩", "|1⟩=|↑⟩"], "dim": 2, "truncated": False,
                  "note": "σz 本征基；H₀=(Δ/2)σz，V=(λ/2)σx；ħ=1"},
        "eigen": core["eigen"],
        "times": None,
        "states": None,
        "observables": {},
        "scalars": scalars,
        "analytic": core["analytic"],
        "sweep": core["sweep"],
        "metrics": core["metrics"],
        "warnings": warnings,
        "summary": {"title": "非简并微扰的二阶能量与精确解对比",
                    "bullets": bullets},
    }
    result["figure_specs"] = _perturbation_figures(core)
    return result


# ---------------------------------------------------------------------------
# 对外入口
# ---------------------------------------------------------------------------
def run_model(req: dict) -> dict:
    """按 canonical request 求解模型，返回契约第 2 节的 result dict。

    :param req: :func:`qllab.spec.normalize_request` 的输出。
    :raises ValueError: 请求虽然通过规范化、但在模型层面不可表示时
        （例如 Fock 初态 |n⟩ 超出截断维数）。
    """
    if not isinstance(req, dict):
        raise ValueError("run_model 需要 canonical request（dict）")
    model = req.get("model")
    if model == "spin_half":
        return _run_spin(req)
    if model == "harmonic_oscillator":
        return _run_ho(req)
    if model == "perturbation":
        return _run_perturbation(req)
    raise ValueError(f"不支持的模型 {model!r}；可选：{list(spec.MODELS)}")
