# -*- coding: utf-8 -*-
"""请求规范化：白名单、别名、默认值、类型与范围检查。

任何用户/AI 给出的自然语言请求，都必须先经过 :func:`normalize_request`
变成受限的 canonical request，才允许进入求解器。这里也是唯一允许出现
``SpecError`` 的地方之一（另一个是参数扫描展开）。
"""
from __future__ import annotations

import math

SCHEMA = "quantumlab/1"

# ---------------------------------------------------------------- 规模上限
LIMITS = {
    "n_levels": (2, 400),
    "n_steps": (2, 20000),
    "sweep_points": 60,
    "abs_param": 1e6,
}

# ------------------------------------------------------------ 容差默认值
DEFAULT_TOL = {
    "hermiticity_abs": 1e-10,     # max |H - H†|
    "eigen_residual_abs": 1e-8,   # max |H v - E v|
    "orthonormality_abs": 1e-8,   # max |V†V - I|
    "norm_dev": 1e-9,             # 演化态范数最大偏差
    "probability_sum": 1e-9,      # 占据概率和与 1 的偏差
    "energy_imag": 1e-10,         # 能量虚部
    "energy_drift": 1e-8,         # 能量相对漂移
    "analytic_abs": 1e-6,         # 与解析解的最大绝对偏差
    "truncation_pop": 1e-6,       # 截断边界能级最大占据
    "spectrum_abs": 1e-8,         # 能谱与解析值偏差
    "perturbation_rel": 1e-2,     # 微扰近似相对误差阈值（适用范围判定）
}

MODELS = ("spin_half", "harmonic_oscillator", "perturbation")

MODEL_ALIASES = {
    "spin_half": "spin_half", "spin": "spin_half", "spin1/2": "spin_half",
    "spin-1/2": "spin_half", "twolevel": "spin_half", "two_level": "spin_half",
    "qubit": "spin_half", "自旋": "spin_half", "二能级": "spin_half", "两能级": "spin_half",
    "harmonic_oscillator": "harmonic_oscillator", "oscillator": "harmonic_oscillator",
    "harmonic": "harmonic_oscillator", "ho": "harmonic_oscillator",
    "谐振子": "harmonic_oscillator", "量子谐振子": "harmonic_oscillator",
    "perturbation": "perturbation", "perturb": "perturbation", "微扰": "perturbation",
    "非简并微扰": "perturbation",
}

PARAM_KEYS = {
    "omega_x": "Omega_x", "omegax": "Omega_x", "ω_x": "Omega_x",
    "omega_y": "Omega_y", "omegay": "Omega_y",
    "omega_z": "Omega_z", "omegaz": "Omega_z",
    "omega": "omega", "freq": "omega", "frequency": "omega", "ω": "omega",
    "n_levels": "n_levels", "nlevels": "n_levels", "nlevel": "n_levels",
    "dim": "n_levels", "dimension": "n_levels", "truncation": "n_levels",
    "delta": "delta", "gap": "delta", "detuning": "delta", "Δ": "delta",
    "lambda": "lambda", "lam": "lambda", "coupling": "lambda", "λ": "lambda",
}

# 对外统一名 -> 可接受别名（比较前统一经 _key 规范化：去空格/下划线/连字符、
# 转小写，因此 Ω/ω、Δ/δ、Λ/λ 这类大小写差异会自动归并）。
_PARAM_ALIASES = {
    "Omega_x": {"omegax", "ωx", "ω_x", "x", "驱动x"},
    "Omega_y": {"omegay", "ωy", "ω_y", "y"},
    "Omega_z": {"omegaz", "ωz", "ω_z", "z", "纵向", "失谐"},
    "omega": {"omega", "freq", "ω", "频率"},
    "n_levels": {"nlevels", "nlevel", "dim", "dimension", "截断", "维度"},
    "delta": {"delta", "gap", "Δ", "间隙"},
    "lambda": {"lambda", "lam", "λ", "扰动强度"},
}

MODEL_PARAMS = {
    "spin_half": ("Omega_x", "Omega_y", "Omega_z"),
    "harmonic_oscillator": ("omega", "n_levels"),
    "perturbation": ("delta", "lambda"),
}

PARAM_DEFAULTS = {
    "Omega_x": 0.0, "Omega_y": 0.0, "Omega_z": 0.0,
    "omega": 1.0, "n_levels": 40,
    "delta": 1.0, "lambda": 0.1,
}

TOP_KEYS = {
    "schema", "model", "params", "initial_state", "time", "units", "solver",
    "sweep", "tolerance", "output", "question", "provenance", "warnings",
}
_TOP_ALIASES = {
    "模型": "model", "参数": "params", "初态": "initial_state", "初始态": "initial_state",
    "state": "initial_state", "initial": "initial_state", "psi0": "initial_state",
    "问题": "question", "原始问题": "question", "notes": "question", "note": "question",
    "时间": "time", "单位": "units", "扫描": "sweep", "容差": "tolerance",
    "输出": "output", "求解器": "solver",
}

STATE_KINDS_SPIN = ("up", "down", "plus_x", "plus_y", "custom")
STATE_KINDS_HO = ("fock", "coherent", "superposition")
FORMATS = ("json", "csv", "png", "html")

HAMILTONIAN_TEXT = {
    "spin_half": "H = (ħ/2)(Ωx σx + Ωy σy + Ωz σz)",
    "harmonic_oscillator": "H = ħω(a†a + 1/2)，截断到 N 个 Fock 能级",
    "perturbation": "H = (Δ/2)σz + (λ/2)σx",
}


class SpecError(ValueError):
    """请求不合法。消息应当可以直接转述给用户。"""


# ------------------------------------------------------------------ helpers
def _num(name, value, lo=None, hi=None, integer=False, allow_zero=True):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SpecError(f"参数 {name} 必须是实数，收到 {value!r}")
    if integer:
        if isinstance(value, float) and not float(value).is_integer():
            raise SpecError(f"参数 {name} 必须是整数，收到 {value!r}")
        value = int(value)
    value = float(value) if not integer else int(value)
    if not math.isfinite(float(value)):
        raise SpecError(f"参数 {name} 必须是有限实数，收到 {value!r}")
    if lo is not None and value < lo:
        raise SpecError(f"参数 {name} 超出范围：要求 ≥ {lo}，收到 {value}")
    if hi is not None and value > hi:
        raise SpecError(f"参数 {name} 超出范围：要求 ≤ {hi}，收到 {value}")
    if not allow_zero and value == 0:
        raise SpecError(f"参数 {name} 不能为 0")
    return value


def _key(name):
    return str(name).strip().replace(" ", "").replace("_", "").replace("-", "").lower()


def _canon_param(name):
    k = _key(name)
    for canon, aliases in _PARAM_ALIASES.items():
        if k == _key(canon) or k in {_key(a) for a in aliases}:
            return canon
    return None


def _canon_model(name):
    if not isinstance(name, str):
        raise SpecError(f"model 必须是字符串，收到 {name!r}")
    k = _key(name)
    if k in {_key(x) for x in MODEL_ALIASES}:
        for alias, canon in MODEL_ALIASES.items():
            if _key(alias) == k:
                return canon
    raise SpecError(
        f"不支持的模型 {name!r}。当前支持：spin_half（自旋 1/2 二能级）、"
        f"harmonic_oscillator（量子谐振子）、perturbation（非简并微扰）。"
    )


def _complex_pair(name, value):
    """把 [re, im] / [re] / 复数 规范为 (re, im)。"""
    if isinstance(value, complex):
        return (float(value.real), float(value.imag))
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return (_num(name, value[0]), 0.0)
        if len(value) == 2:
            return (_num(name, value[0]), _num(name, value[1]))
    raise SpecError(f"{name} 需要写成 [实部, 虚部] 两元组，收到 {value!r}")


def _canon_state(model, raw):
    if raw is None:
        raw = {}
    if not isinstance(raw, (dict, str)):
        raise SpecError("initial_state 必须是对象（如 {\"kind\": \"up\"}）或简写字符串")
    if isinstance(raw, str):
        raw = {"kind": raw}
    raw = dict(raw)
    kind = raw.get("kind", raw.get("type", raw.get("态")))
    if kind is not None:
        if not isinstance(kind, str):
            raise SpecError(f"initial_state.kind 必须是字符串，收到 {kind!r}")
        kind = _key(kind)

    if model == "spin_half":
        mapping = {"up": "up", "0": "up", "自旋向上": "up", "激发态0": "up",
                   "down": "down", "1": "down", "自旋向下": "down",
                   "plusx": "plus_x", "plus_x": "plus_x", "x+": "plus_x", "+x": "plus_x",
                   "plusy": "plus_y", "plus_y": "plus_y", "y+": "plus_y", "+y": "plus_y",
                   "custom": "custom", "自定义": "custom"}
        kind = mapping.get(kind, kind) if kind else "up"
        if kind not in STATE_KINDS_SPIN:
            raise SpecError(f"自旋初态 kind 仅支持 {STATE_KINDS_SPIN}，收到 {kind!r}")
        vec = raw.get("vector", raw.get("psi", raw.get("振幅")))
        if kind == "custom":
            if vec is None:
                raise SpecError("kind=custom 时必须提供 vector=[[re,im],[re,im]]")
            if not isinstance(vec, (list, tuple)) or len(vec) != 2:
                raise SpecError("custom 初态 vector 必须是两个复振幅的列表")
            v = [_complex_pair("vector[0]", vec[0]), _complex_pair("vector[1]", vec[1])]
            norm = math.hypot(*v[0]) ** 2 + math.hypot(*v[1]) ** 2
            if norm <= 0:
                raise SpecError("custom 初态不能是零向量")
            if abs(norm - 1.0) > 1e-9:
                f = 1.0 / math.sqrt(norm)
                v = [(v[0][0] * f, v[0][1] * f), (v[1][0] * f, v[1][1] * f)]
                state = {"kind": "custom", "vector": [list(v[0]), list(v[1])],
                         "renormalized": True}
                return state
            return {"kind": "custom", "vector": [list(v[0]), list(v[1])], "renormalized": False}
        return {"kind": kind, "vector": None}

    if model == "harmonic_oscillator":
        mapping = {"fock": "fock", "n": "fock", "能级": "fock", "本征态": "fock",
                   "coherent": "coherent", "相干": "coherent", "相干态": "coherent",
                   "superposition": "superposition", "叠加": "superposition",
                   "叠加态": "superposition"}
        kind = mapping.get(kind, kind) if kind else "superposition"
        if kind not in STATE_KINDS_HO:
            raise SpecError(f"谐振子初态 kind 仅支持 {STATE_KINDS_HO}，收到 {kind!r}")
        if kind == "fock":
            n = raw.get("n", 0)
            n = _num("initial_state.n", n, lo=0, integer=True)
            return {"kind": "fock", "n": int(n)}
        if kind == "coherent":
            alpha = _complex_pair("initial_state.alpha", raw.get("alpha", [1.0, 0.0]))
            if math.hypot(*alpha) > 12:
                raise SpecError("相干态 |α| 过大（>12），截断空间不足，请减小 α 或增大 n_levels")
            return {"kind": "coherent", "alpha": [alpha[0], alpha[1]]}
        comps = raw.get("components")
        if comps is None:
            comps = [{"n": 0, "amplitude": [1.0, 0.0]}, {"n": 1, "amplitude": [1.0, 0.0]}]
        if not isinstance(comps, (list, tuple)) or not comps:
            raise SpecError("superposition 初态需要 components=[{n, amplitude}, ...]")
        out, total = [], 0.0
        for i, c in enumerate(comps):
            if not isinstance(c, dict):
                raise SpecError(f"components[{i}] 必须是对象")
            n = _num(f"components[{i}].n", c.get("n", 0), lo=0, integer=True)
            amp = _complex_pair(f"components[{i}].amplitude", c.get("amplitude", [1.0, 0.0]))
            total += math.hypot(*amp) ** 2
            out.append({"n": int(n), "amplitude": [amp[0], amp[1]]})
        if total <= 0:
            raise SpecError("superposition 初态不能全为零振幅")
        if abs(total - 1.0) > 1e-9:
            f = 1.0 / math.sqrt(total)
            for c in out:
                c["amplitude"] = [c["amplitude"][0] * f, c["amplitude"][1] * f]
            return {"kind": "superposition", "components": out, "renormalized": True}
        return {"kind": "superposition", "components": out}

    return None  # perturbation 无初态


def _check_state_bounds(model, state, params):
    """初态分量必须落在模型允许的空间内（否则求解器只能报底层错误）。"""
    if model != "harmonic_oscillator" or not state:
        return
    n_max = int(params["n_levels"]) - 1
    if state["kind"] == "fock" and state["n"] > n_max:
        raise SpecError(
            f"初态 |{state['n']}⟩ 超出截断空间：n_levels={params['n_levels']} 只包含 "
            f"n=0…{n_max}；请增大 n_levels 或改用更低能级。"
        )
    if state["kind"] == "superposition":
        over = [c["n"] for c in state["components"] if c["n"] > n_max]
        if over:
            raise SpecError(
                f"叠加态分量 n={sorted(set(over))} 超出截断空间：n_levels="
                f"{params['n_levels']} 只包含 n=0…{n_max}；请增大 n_levels。"
            )


def _canon_time(model, raw, params, warnings):
    if model == "perturbation":
        if raw not in (None, {}):
            warnings.append("微扰分析在本技能中是静态能谱计算，已忽略 time 配置")
        return None
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise SpecError("time 必须是对象，如 {\"t_start\": 0, \"t_end\": 6.28, \"n_steps\": 401}")
    t_start = _num("time.t_start", raw.get("t_start", 0.0))
    n_steps = _num("time.n_steps", raw.get("n_steps", 401),
                   lo=LIMITS["n_steps"][0], hi=LIMITS["n_steps"][1], integer=True)
    if model == "spin_half":
        om = math.sqrt(params["Omega_x"] ** 2 + params["Omega_y"] ** 2 + params["Omega_z"] ** 2)
        default_end = 2.0 * math.pi / om
    else:
        default_end = 2.0 * math.pi / params["omega"]
    t_end = raw.get("t_end")
    if t_end is None:
        t_end = t_start + default_end
        warnings.append(
            f"未提供 time.t_end，采用默认 2π/Ωeff = {default_end:.6g}（一个完整振荡周期）"
        )
    t_end = _num("time.t_end", t_end)
    if not t_end > t_start:
        raise SpecError(f"time.t_end 必须大于 t_start，收到 t_start={t_start}, t_end={t_end}")
    return {"t_start": float(t_start), "t_end": float(t_end), "n_steps": int(n_steps)}


def _canon_sweep(model, raw, params_block, warnings):
    if raw in (None, {}):
        if model == "perturbation":
            vals = [round(0.01 * i, 4) for i in range(1, 26)]   # 0.01 .. 0.25
            vals += [0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0]
            warnings.append(
                f"未指定扫描，默认扫描扰动强度 λ ∈ [0.01, 2.0]（共 {len(vals)} 点）"
            )
            return {"param": "lambda", "values": vals, "auto": True}
        return None
    if not isinstance(raw, dict):
        raise SpecError("sweep 必须是对象，如 {\"param\": \"lambda\", \"values\": [0.1, 0.2]}")
    pname = raw.get("param", raw.get("参数"))
    if pname is None:
        raise SpecError("sweep 需要 param 字段指明扫描变量")
    canon = _canon_param(pname)
    if canon is None or canon not in MODEL_PARAMS[model]:
        raise SpecError(
            f"sweep.param={pname!r} 不属于模型 {model} 的可扫描参数 "
            f"{list(MODEL_PARAMS[model])}"
        )
    if "values" in raw:
        vals = raw["values"]
        if not isinstance(vals, (list, tuple)) or not vals:
            raise SpecError("sweep.values 必须是非空列表")
        if len(vals) > LIMITS["sweep_points"]:
            raise SpecError(
                f"扫描点数 {len(vals)} 超过上限 {LIMITS['sweep_points']}；"
                f"请先说明规模并拆分任务"
            )
        out = []
        for v in vals:
            out.append(_num(f"sweep.values[{len(out)}]", v,
                            lo=(LIMITS["n_levels"][0] if canon == "n_levels" else 0.0),
                            hi=LIMITS["abs_param"], integer=(canon == "n_levels")))
        return {"param": canon, "values": out}
    if {"start", "stop"}.issubset(raw.keys()):
        start = _num("sweep.start", raw["start"])
        stop = _num("sweep.stop", raw["stop"])
        num = _num("sweep.num", raw.get("num", 15),
                   lo=2, hi=LIMITS["sweep_points"], integer=True)
        if stop <= start:
            raise SpecError("sweep.stop 必须大于 sweep.start")
        step = (stop - start) / (num - 1)
        vals = [start + i * step for i in range(int(num))]
        if canon == "n_levels":
            vals = sorted({max(2, int(round(v))) for v in vals})
        return {"param": canon, "values": vals}
    raise SpecError("sweep 需要 values，或 start/stop/num 三件套")


def _canon_output(raw, warnings):
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise SpecError("output 必须是对象")
    formats = raw.get("formats", list(FORMATS))
    if not isinstance(formats, (list, tuple)):
        raise SpecError("output.formats 必须是列表")
    fmts = []
    for f in formats:
        if not isinstance(f, str) or f.lower() not in FORMATS:
            raise SpecError(f"不支持的输出格式 {f!r}，可选 {list(FORMATS)}")
        if f.lower() not in fmts:
            fmts.append(f.lower())
    if not fmts:
        raise SpecError("output.formats 不能为空")
    figures = bool(raw.get("figures", "png" in fmts))
    if figures and "png" not in fmts:
        fmts.append("png")
    return {"figures": figures, "formats": fmts}


# ------------------------------------------------------------------- public
def normalize_request(raw: dict) -> dict:
    """把任意请求规范化为 canonical request；非法时抛 :class:`SpecError`。"""
    if not isinstance(raw, dict):
        raise SpecError("请求必须是 JSON 对象（dict）")
    raw = dict(raw)
    warnings = []

    # 顶层别名
    for k in list(raw.keys()):
        kk = _key(k)
        for alias, canon in _TOP_ALIASES.items():
            if kk == _key(alias) and canon not in raw:
                raw[canon] = raw.pop(k)
                break
    unknown = [k for k in raw if k not in TOP_KEYS and _canon_param(k) is None]
    if unknown:
        raise SpecError(
            f"未知请求字段 {unknown}。允许的顶层字段：{sorted(TOP_KEYS)}"
            f"；模型参数请放进 params 或直接写 Omega_x/omega/n_levels/delta/lambda。"
        )

    if "model" not in raw:
        raise SpecError("请求缺少 model 字段（spin_half / harmonic_oscillator / perturbation）")
    model = _canon_model(raw["model"])

    # 参数收集：params 子对象 + 顶层裸参数
    pblock = raw.get("params") or {}
    if not isinstance(pblock, dict):
        raise SpecError("params 必须是对象")
    collected, provenance = {}, {}
    for src, block in (("params", pblock), ("top", {k: v for k, v in raw.items()
                                                    if k in MODEL_PARAMS[model] or _canon_param(k)})):
        for k, v in block.items():
            canon = _canon_param(k)
            if canon is None:
                raise SpecError(
                    f"未知参数 {k!r}。模型 {model} 支持：{list(MODEL_PARAMS[model])}"
                )
            if canon not in MODEL_PARAMS[model]:
                raise SpecError(
                    f"参数 {k!r} 不属于模型 {model}；该模型支持：{list(MODEL_PARAMS[model])}"
                )
            if canon in collected:
                raise SpecError(f"参数 {canon} 被重复指定（params 与顶层各出现一次）")
            collected[canon] = v
            provenance[canon] = src

    params = {}
    for pname in MODEL_PARAMS[model]:
        if pname in collected:
            v = collected[pname]
            if pname == "n_levels":
                params[pname] = _num(pname, v, lo=LIMITS["n_levels"][0],
                                     hi=LIMITS["n_levels"][1], integer=True)
            elif pname == "omega":
                params[pname] = _num(pname, v, lo=0.0, hi=LIMITS["abs_param"], allow_zero=False)
            elif pname == "delta":
                params[pname] = _num(pname, v, lo=0.0, hi=LIMITS["abs_param"], allow_zero=False)
            elif pname == "lambda":
                params[pname] = _num(pname, v, lo=0.0, hi=LIMITS["abs_param"])
            else:
                params[pname] = _num(pname, v, lo=-LIMITS["abs_param"], hi=LIMITS["abs_param"])
        else:
            params[pname] = PARAM_DEFAULTS[pname]
            provenance[pname] = "default"
            if model == "spin_half" and pname in ("Omega_x", "Omega_y", "Omega_z"):
                warnings.append(f"未提供 {pname}，采用默认值 0（未驱动方向）")
            elif model != "spin_half":
                warnings.append(f"未提供 {pname}，采用默认值 {PARAM_DEFAULTS[pname]}")

    if model == "spin_half":
        om = math.sqrt(params["Omega_x"] ** 2 + params["Omega_y"] ** 2 + params["Omega_z"] ** 2)
        if om == 0:
            raise SpecError("Ωx=Ωy=Ωz=0 时哈密顿量为零，无物理演化；请至少给出一个非零分量")
        if om > LIMITS["abs_param"]:
            raise SpecError(f"总驱动强度 |Ω|={om:g} 超出上限 {LIMITS['abs_param']}")
    if model == "perturbation" and params["delta"] <= 0:
        raise SpecError("非简并微扰要求 Δ>0")

    initial_state = _canon_state(model, raw.get("initial_state"))
    _check_state_bounds(model, initial_state, params)
    time_block = _canon_time(model, raw.get("time"), params, warnings)
    sweep = _canon_sweep(model, raw.get("sweep"), pblock, warnings)

    units_raw = raw.get("units") or {}
    if not isinstance(units_raw, dict):
        raise SpecError("units 必须是对象")
    hbar = _num("units.hbar", units_raw.get("hbar", 1.0))
    if abs(hbar - 1.0) > 1e-12:
        raise SpecError(
            "本技能统一采用 ħ=1 的无量纲约定（能量与角频率同单位），"
            f"暂不支持 units.hbar={hbar}；请改用 ħ=1 下的数值或自行换算。"
        )
    units = {"hbar": 1.0, "note": "ħ=1，Ω/ω/Δ/λ 与角频率同量纲"}

    solver_raw = raw.get("solver") or {}
    if not isinstance(solver_raw, dict):
        raise SpecError("solver 必须是对象")
    method = solver_raw.get("method", "eigendecomposition")
    if method not in ("eigendecomposition",):
        raise SpecError(
            f"暂不支持 solver.method={method!r}；当前仅提供厄米矩阵谱分解"
            "（H=VΛV†，演化 U=e^{-iHt}）"
        )
    solver = {"method": "eigendecomposition",
              "note": "H 厄米 → np.linalg.eigh 谱分解，U(t)=V exp(-iΛt) V†"}

    tol = dict(DEFAULT_TOL)
    tol_raw = raw.get("tolerance") or {}
    if not isinstance(tol_raw, dict):
        raise SpecError("tolerance 必须是对象")
    for k, v in tol_raw.items():
        if k not in DEFAULT_TOL:
            raise SpecError(f"未知容差项 {k!r}，可选：{sorted(DEFAULT_TOL)}")
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not v > 0:
            raise SpecError(f"容差 {k} 必须是正实数，收到 {v!r}")
        tol[k] = float(v)

    question = raw.get("question", "")
    if question is None:
        question = ""
    if not isinstance(question, str):
        raise SpecError("question 必须是字符串")
    if len(question) > 4000:
        question = question[:4000]
        warnings.append("原始问题超过 4000 字，已截断记录")

    return {
        "schema": SCHEMA,
        "model": model,
        "params": params,
        "initial_state": initial_state,
        "time": time_block,
        "units": units,
        "solver": solver,
        "sweep": sweep,
        "tolerance": tol,
        "output": _canon_output(raw.get("output"), warnings),
        "question": question,
        "provenance": provenance,
        "warnings": warnings,
    }


def describe_request(req: dict) -> str:
    """人类可读的模型/参数确认块，执行前展示。"""
    model = req["model"]
    p = req["params"]
    prov = req.get("provenance", {})
    lines = ["【模型与参数确认】",
             f"模型：{model}",
             f"哈密顿量：{HAMILTONIAN_TEXT[model]}"]

    def tag(name):
        src = prov.get(name, "given")
        return "（默认值）" if src == "default" else ""

    if model == "spin_half":
        lines.append(f"参数：Ωx={p['Omega_x']:g}{tag('Omega_x')}，"
                     f"Ωy={p['Omega_y']:g}{tag('Omega_y')}，"
                     f"Ωz={p['Omega_z']:g}{tag('Omega_z')}，单位 rad/time（ħ=1）")
        om = math.sqrt(sum(p[k] ** 2 for k in ("Omega_x", "Omega_y", "Omega_z")))
        lines.append(f"能级：E± = ±{om / 2:g}；等效拉比频率 |Ω|={om:g}")
        st = req["initial_state"]
        label = {"up": "|↑⟩（σz 本征态，默认）", "down": "|↓⟩",
                 "plus_x": "|+x⟩（σx 本征态）", "plus_y": "|+y⟩",
                 "custom": "自定义复向量"}[st["kind"]]
        lines.append(f"初态：{label}")
    elif model == "harmonic_oscillator":
        lines.append(f"参数：ω={p['omega']:g}{tag('omega')}，截断能级数 N={p['n_levels']}"
                     f"{tag('n_levels')}（Fock 空间 n=0…{p['n_levels'] - 1}）")
        st = req["initial_state"]
        if st["kind"] == "fock":
            lines.append(f"初态：|{st['n']}⟩")
        elif st["kind"] == "coherent":
            a = st["alpha"]
            lines.append(f"初态：相干态 |α⟩，α={a[0]:g}{a[1]:+g}j")
        else:
            comp = " + ".join(
                f"({c['amplitude'][0]:g}{c['amplitude'][1]:+g}j)|{c['n']}⟩"
                for c in st["components"])
            lines.append(f"初态：{comp}（已归一化）")
        lines.append("截断说明：有限维 Fock 截断，最高能级仍需检查占据；"
                     "边界占据小不等于已证明收敛。")
    else:
        lines.append(f"参数：Δ={p['delta']:g}{tag('delta')}，"
                     f"λ={p['lambda']:g}{tag('lambda')}（H 的本征能量精确值 "
                     f"±√(Δ²+λ²)/2）")

    if req["time"]:
        t = req["time"]
        lines.append(f"时间：t ∈ [{t['t_start']:g}, {t['t_end']:g}]，{t['n_steps']} 个采样点")
    if req["sweep"]:
        sw = req["sweep"]
        lines.append(f"参数扫描：{sw['param']} 共 {len(sw['values'])} 点"
                     f"（{sw['values'][0]:g} … {sw['values'][-1]:g}）")
    lines.append(f"单位约定：{req['units']['note']}")
    lines.append(f"求解方法：{req['solver']['note']}")
    lines.append(f"输出：{', '.join(req['output']['formats'])}")
    if req.get("warnings"):
        lines.append("提示：")
        lines.extend(f"  - {w}" for w in req["warnings"])
    return "\n".join(lines)
