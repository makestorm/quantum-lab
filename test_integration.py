# -*- coding: utf-8 -*-
"""QuantumLab red-team integration / cross-validation suite.

Independent of the product implementation's *algorithms*: Hamiltonians and
time evolution are recomputed with the from-scratch routines in
``tests/reference_physics.py`` (Taylor expm with scaling/squaring, fixed-step
RK4, closed-form analytic solutions).  The product code is only imported to
obtain its outputs.

Usage (repository root):
  python tests/test_integration.py                # run everything available
  python tests/test_integration.py --section ref,spec,physics
  QUANTUMLAB_ROOT=<skill dir> python tests/test_integration.py   # 指定技能根目录

Sections: ref, static, spec, physics, validation, cli, artifacts, models_artifacts,
e2e, regressions.
Exit code 0 only if no FAIL record was produced.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import traceback

try:  # Windows console default is GBK; keep check messages readable
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def _find_skill_root():
    """定位含 scripts/quantum_lab.py 的技能根目录。

    支持两种布局：仓库根即技能目录（GitHub 仓库），或 tests/ 位于
    ``<workspace>/.dsh/skills/quantum-lab/tests/`` 的 DSH 安装布局。
    可用环境变量 ``QUANTUMLAB_ROOT`` 显式指定。
    """
    env = os.environ.get("QUANTUMLAB_ROOT")
    if env and os.path.isfile(os.path.join(env, "scripts", "quantum_lab.py")):
        return os.path.abspath(env)
    cur = HERE
    for _ in range(6):
        if os.path.isfile(os.path.join(cur, "scripts", "quantum_lab.py")):
            return cur
        nested = os.path.join(cur, ".dsh", "skills", "quantum-lab")
        if os.path.isfile(os.path.join(nested, "scripts", "quantum_lab.py")):
            return nested
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return ROOT


SKILL = _find_skill_root()
SCRIPTS = os.path.join(SKILL, "scripts")
QL_DIR = os.path.join(SCRIPTS, "qllab")
CLI = os.path.join(SCRIPTS, "quantum_lab.py")
PY = sys.executable
ART = os.path.join(HERE, "_artifacts")

sys.path.insert(0, HERE)
sys.path.insert(0, SCRIPTS)

import numpy as np  # noqa: E402

import reference_physics as ref  # noqa: E402

RECORDS = []
_ARTIFACT_DIR = None


# --------------------------------------------------------------------------
# harness
# --------------------------------------------------------------------------
def record(area, name, status, evidence="", cmd=""):
    RECORDS.append({"area": area, "name": name, "status": status,
                    "evidence": str(evidence), "cmd": cmd})
    print(f"[{status:7s}] {area}: {name} :: {evidence}")
    return status


def check(area, name, ok, evidence="", cmd=""):
    return record(area, name, "PASS" if ok else "FAIL", evidence, cmd)


def blocked(area, name, why):
    return record(area, name, "BLOCKED", why)


def fmt_max(x):
    return f"{x:.3e}"


def require(result, keys):
    missing = [k for k in keys if k not in result]
    if missing:
        raise KeyError(f"result missing keys {missing}")


def guard(area, label, fn):
    """Run fn(); a crash becomes one FAIL record instead of aborting the section."""
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001
        record(area, f"{label}_crash", "FAIL",
               f"{type(exc).__name__}: {exc} | "
               + traceback.format_exc().strip().splitlines()[-1][:200])
        return None


def wait_for(paths, timeout_s):
    """Poll (5 s cadence) until every path exists or the budget is spent."""
    t0 = time.time()
    missing = [p for p in paths if not os.path.exists(p)]
    print(f"[wait] waiting up to {timeout_s}s for {len(missing)} file(s): "
          f"{[os.path.basename(p) for p in missing]}")
    while missing and time.time() - t0 < timeout_s:
        time.sleep(5)
        missing = [p for p in paths if not os.path.exists(p)]
        print(f"[wait] t+{time.time() - t0:5.0f}s missing="
              f"{[os.path.basename(p) for p in missing]}")
    return [p for p in paths if not os.path.exists(p)]


def module(name):
    try:
        return __import__(f"qllab.{name}", fromlist=["*"])
    except Exception as exc:  # noqa: BLE001
        return exc


def cli_run(args, cwd, timeout=180):
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = SCRIPTS + os.pathsep + env.get("PYTHONPATH", "")
    try:
        p = subprocess.run([PY, CLI] + args, cwd=cwd, capture_output=True,
                           text=True, encoding="utf-8", errors="replace",
                           timeout=timeout, env=env)
        return p.returncode, p.stdout or "", p.stderr or ""
    except subprocess.TimeoutExpired:
        return -9, "", "TIMEOUT"


def parse_json_out(text):
    """Extract the last JSON object from CLI stdout."""
    text = text.strip()
    for start in range(len(text)):
        if text[start] in "{[":
            try:
                return json.loads(text[start:])
            except Exception:  # noqa: BLE001
                continue
    raise ValueError("no JSON object in CLI output")


def fresh_dir(name):
    d = os.path.join(ART, name)
    os.makedirs(d, exist_ok=True)
    # do not delete user data; just use a unique subdirectory per run
    d = os.path.join(d, time.strftime("run-%H%M%S"))
    os.makedirs(d, exist_ok=True)
    return d


def experiment_dirs(root):
    idx = os.path.join(root, "index.json")
    if os.path.exists(idx):
        try:
            data = json.load(open(idx, encoding="utf-8"))
            ids = [e["id"] for e in data.get("experiments", [])]
            return [os.path.join(root, i) for i in ids]
        except Exception:  # noqa: BLE001
            pass
    return [os.path.join(root, d) for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))]


def latest_exp(root):
    dirs = experiment_dirs(root)
    return dirs[0] if dirs else None


# --------------------------------------------------------------------------
# section: reference implementation self-consistency
# --------------------------------------------------------------------------
def section_ref():
    try:
        m = ref.selftest(verbose=False)
    except Exception as exc:  # noqa: BLE001
        record("ref", "selftest", "FAIL", f"{type(exc).__name__}: {exc}")
        return
    analytic = {k: v for k, v in m.items() if "analytic" in k or "bloch" in k}
    numeric = {k: v for k, v in m.items() if "rk4" in k or "vs_rk4" in k}
    for k, v in sorted(m.items()):
        print(f"        ref metric {k:38s} {v:.3e}")
    check("ref", "closed_form_cross_check",
          max(analytic.values()) < 1e-8,
          f"max dev vs analytic = {fmt_max(max(analytic.values()))}")
    check("ref", "taylor_rk4_cross_check",
          max(numeric.values()) < 1e-6,
          f"max Taylor/RK4 dev = {fmt_max(max(numeric.values()))}")
    check("ref", "unitarity_time_reversal", m["spin_Ut_Uminus_t_I"] < 1e-12,
          f"|U(t)U(-t)-I| = {fmt_max(m['spin_Ut_Uminus_t_I'])}")
    check("ref", "norm_preservation", m["spin_norm_dev"] < 1e-12
          and m["ho_norm_dev"] < 1e-12,
          f"spin {fmt_max(m['spin_norm_dev'])}, HO {fmt_max(m['ho_norm_dev'])}")
    check("ref", "ho_spectrum_closed_form",
          m["ho_spectrum_vs_n_plus_half"] < 1e-12,
          f"|E_n-(n+1/2)| = {fmt_max(m['ho_spectrum_vs_n_plus_half'])}")
    pens = ref.pert_energies(1.0, 0.1)
    check("ref", "perturbation_closed_form",
          abs(pens["E0_exact"] + 0.5024937810560445) < 1e-15,
          f"E0_exact={pens['E0_exact']!r}, rel_err={pens['rel_err']:.6e} "
          f"(closed form -sqrt(D^2+l^2)/2, see docs/CONTRACT.md)")


# --------------------------------------------------------------------------
# section: static safety checks on the implementation source
# --------------------------------------------------------------------------
def section_static():
    files = [f for f in os.listdir(QL_DIR) if f.endswith(".py")]
    check("static", "files_present",
          all(os.path.exists(os.path.join(QL_DIR, f))
              for f in ("models.py", "validate.py", "charts.py", "store.py",
                        "report.py", "cli.py")),
          f"present={sorted(files)}")
    bad_imports = []
    dangerous = []
    for f in files:
        src = open(os.path.join(QL_DIR, f), encoding="utf-8").read()
        for lib in ("scipy", "matplotlib", "pandas"):
            if re.search(rf"^\s*(import|from)\s+{lib}\b", src, re.M):
                bad_imports.append(f"{f}:{lib}")
        for pat in (r"(?<![.\w])eval\s*\(", r"(?<![.\w])exec\s*\(",
                    r"\bos\.system\s*\(",
                    r"\bsubprocess\.(run|Popen|call|check_output)",
                    r"(?<![.\w])__import__\s*\(", r"(?<![.\w])compile\s*\("):
            if re.search(pat, src):
                dangerous.append(f"{f}:{pat}")
    check("static", "no_forbidden_libs", not bad_imports, str(bad_imports))
    check("static", "no_arbitrary_code_execution", not dangerous,
          str(dangerous) + "  (cli.py subprocess is only allowed if it is the "
          "verification harness; inspect manually)")
    # traceback must not be echoed to users by cli
    cli_src = ""
    if os.path.exists(os.path.join(QL_DIR, "cli.py")):
        cli_src = open(os.path.join(QL_DIR, "cli.py"), encoding="utf-8").read()
    check("static", "cli_has_no_naked_traceback_print",
          "traceback.print_exc" not in cli_src,
          "cli.py prints user-facing errors, not stack traces")


# --------------------------------------------------------------------------
# section: adversarial normalize_request
# --------------------------------------------------------------------------
def section_spec():
    spec = module("spec")
    if isinstance(spec, Exception):
        return blocked("spec", "import", repr(spec))

    base_spin = {"model": "spin_half", "params": {"Omega_x": 2.0, "Omega_z": 1.0}}
    base_ho = {"model": "harmonic_oscillator", "params": {"omega": 1.0,
                                                          "n_levels": 40}}
    base_pt = {"model": "perturbation", "params": {"delta": 1.0, "lambda": 0.1}}

    cases = [
        ("unknown_param", "spin",
         {"model": "spin_half", "params": {"Omega_x": 1.0, "Omega_q": 2.0}}),
        ("unknown_param_wrong_model", "ho",
         {"model": "harmonic_oscillator", "params": {"Omega_x": 1.0}}),
        ("duplicate_param", "spin",
         {"model": "spin_half", "params": {"Omega_x": 1.0}, "Omega_x": 2.0}),
        ("hbar2", "spin", dict(base_spin, units={"hbar": 2.0})),
        ("hbar_string", "spin", dict(base_spin, units={"hbar": "1"})),
        ("n_steps_1", "spin",
         dict(base_spin, time={"t_start": 0.0, "t_end": 1.0, "n_steps": 1})),
        ("nan_param", "spin",
         {"model": "spin_half", "params": {"Omega_x": float("nan")}}),
        ("inf_param", "spin",
         {"model": "spin_half", "params": {"Omega_x": float("inf")}}),
        ("string_number", "spin",
         {"model": "spin_half", "params": {"Omega_x": "2"}}),
        ("omega_all_zero", "spin",
         {"model": "spin_half", "params": {"Omega_x": 0, "Omega_y": 0,
                                           "Omega_z": 0}}),
        ("omega_negative", "ho",
         {"model": "harmonic_oscillator", "params": {"omega": -1.0}}),
        ("omega_zero", "ho",
         {"model": "harmonic_oscillator", "params": {"omega": 0.0}}),
        ("delta_zero", "pt",
         {"model": "perturbation", "params": {"delta": 0.0}}),
        ("delta_negative", "pt",
         {"model": "perturbation", "params": {"delta": -1.0}}),
        ("n_levels_1", "ho",
         {"model": "harmonic_oscillator", "params": {"n_levels": 1}}),
        ("n_levels_too_big", "ho",
         {"model": "harmonic_oscillator", "params": {"n_levels": 401}}),
        ("sweep_100_points", "pt",
         dict(base_pt, sweep={"param": "lambda",
                              "values": [0.01 * i for i in range(1, 101)]})),
        ("sweep_unknown_param", "pt",
         dict(base_pt, sweep={"param": "Omega_z", "values": [0.1, 0.2]})),
        ("top_unknown_field", "spin", dict(base_spin, banana=1)),
        ("illegal_model", "spin", {"model": "ising", "params": {}}),
        ("model_not_str", "spin", {"model": 3}),
        ("custom_zero_vector", "spin",
         dict(base_spin, initial_state={"kind": "custom",
                                        "vector": [[0, 0], [0, 0]]})),
        ("coherent_alpha_13", "ho",
         dict(base_ho, initial_state={"kind": "coherent", "alpha": [13.0, 0.0]})),
        ("coherent_alpha_12_1", "ho",
         dict(base_ho, initial_state={"kind": "coherent", "alpha": [0.0, 12.1]})),
        ("bad_state_kind", "ho", dict(base_ho, initial_state={"kind": "cat"})),
        ("time_end_le_start", "spin",
         dict(base_spin, time={"t_start": 1.0, "t_end": 1.0, "n_steps": 11})),
        ("bad_solver", "spin", dict(base_spin, solver={"method": "runge_kutta"})),
        ("bad_tolerance_key", "spin", dict(base_spin, tolerance={"foo": 1e-3})),
        ("negative_tolerance", "spin", dict(base_spin, tolerance={"analytic_abs": -1})),
        ("bad_format", "spin", dict(base_spin, output={"formats": ["pdf"]})),
        ("request_not_dict", "spin", [1, 2, 3]),
        ("no_model", "spin", {"params": {"Omega_x": 1.0}}),
        ("superposition_zero", "ho",
         dict(base_ho, initial_state={"kind": "superposition",
                                      "components": [{"n": 0, "amplitude": [0, 0]}]})),
        ("superposition_n_beyond_truncation", "ho",
         dict(base_ho, initial_state={"kind": "superposition",
                                      "components": [{"n": 50, "amplitude": [1, 0]}]})),
        ("fock_negative_n", "ho",
         dict(base_ho, initial_state={"kind": "fock", "n": -1})),
        ("fock_n_beyond_truncation", "ho",
         dict(base_ho, initial_state={"kind": "fock", "n": 40})),
    ]
    unexpected = []
    for name, _kind, raw in cases:
        try:
            spec.normalize_request(raw)
            unexpected.append(name)
            record("spec", name, "FAIL", "NOT REJECTED (unexpected pass)")
        except spec.SpecError as exc:
            msg = str(exc)
            ok = bool(msg.strip()) and "Traceback" not in msg
            record("spec", name, "PASS" if ok else "FAIL",
                   f"SpecError: {msg[:110]}")
        except Exception as exc:  # noqa: BLE001
            record("spec", name, "FAIL",
                   f"wrong exception {type(exc).__name__}: {exc}")
    check("spec", "all_adversarial_rejected", not unexpected,
          f"unexpected passes: {unexpected}")

    # positive controls: valid requests still normalize
    positives = [
        ("spin_alias", {"model": "spin", "params": {"Omega_x": 2.0}}),
        ("top_level_param", {"model": "spin_half", "Omega_x": 2.0,
                             "Omega_z": 1.0}),
        ("hbar_one", dict(base_spin, units={"hbar": 1.0})),
        ("alpha_12_ok", dict(base_ho,
                             initial_state={"kind": "coherent",
                                            "alpha": [12.0, 0.0]})),
        ("sweep_60_ok", dict(base_pt, sweep={
            "param": "lambda", "values": [0.01 * i for i in range(1, 61)]})),
    ]
    for name, raw in positives:
        try:
            req = spec.normalize_request(raw)
            check("spec", "positive_" + name, True,
                  f"model={req['model']} params={req['params']}")
        except Exception as exc:  # noqa: BLE001
            check("spec", "positive_" + name, False, f"{type(exc).__name__}: {exc}")

    # normalization invariants
    req = spec.normalize_request(
        {"model": "spin_half", "params": {"Omega_x": 2.0, "Omega_z": 1.0},
         "initial_state": {"kind": "custom", "vector": [[2.0, 0.0], [0.0, 0.0]]}})
    v = req["initial_state"]["vector"]
    nrm = (v[0][0] ** 2 + v[0][1] ** 2 + v[1][0] ** 2 + v[1][1] ** 2) ** 0.5
    check("spec", "custom_state_renormalized", abs(nrm - 1.0) < 1e-12,
          f"|v|={nrm:.15f}, renormalized flag="
          f"{req['initial_state'].get('renormalized')}")
    check("spec", "default_time_is_period",
          abs(req["time"]["t_end"] - 2 * np.pi / np.sqrt(5)) < 1e-12,
          f"t_end={req['time']['t_end']!r}")
    check("spec", "provenance_and_warnings",
          "provenance" in req and isinstance(req.get("warnings"), list),
          f"provenance={req.get('provenance')}, n_warnings={len(req.get('warnings', []))}")
    # canonical request must be JSON round-trippable via util.to_jsonable
    util = module("util")
    if not isinstance(util, Exception):
        try:
            json.loads(json.dumps(util.to_jsonable(req)))
            check("spec", "canonical_request_json_roundtrip", True, "ok")
        except Exception as exc:  # noqa: BLE001
            check("spec", "canonical_request_json_roundtrip", False, repr(exc))


# --------------------------------------------------------------------------
# section: physics cross-validation against independent reference
# --------------------------------------------------------------------------
def _mk_req(spec, raw):
    return spec.normalize_request(raw)


def _pop(states, idx):
    return np.abs(np.asarray(states, dtype=complex)[:, idx]) ** 2


def section_physics():
    spec = module("spec")
    models = module("models")
    validate = module("validate")
    if isinstance(spec, Exception):
        return blocked("physics", "spec", repr(spec))
    if isinstance(models, Exception):
        return blocked("physics", "models", repr(models))
    if isinstance(validate, Exception):
        return blocked("physics", "validate", repr(validate))

    def run(raw):
        req = spec.normalize_request(raw)
        res = models.run_model(req)
        return req, res

    # ---------------- spin: Omega_x=2, Omega_z=1, |up>
    raw = {"model": "spin_half", "params": {"Omega_x": 2.0, "Omega_z": 1.0},
           "initial_state": {"kind": "up"}}
    req, res = run(raw)
    require(res, ("hamiltonian", "eigen", "times", "states", "observables",
                  "scalars", "metrics", "figure_specs"))
    H = np.asarray(res["hamiltonian"], dtype=complex)
    Href = ref.spin_hamiltonian(2.0, 0.0, 1.0)
    check("physics", "spin_H_matrix", np.max(np.abs(H - Href)) < 1e-12,
          f"max|H-Href|={fmt_max(np.max(np.abs(H - Href)))}")
    ev = np.sort(np.asarray(res["eigen"]["values"]).real)
    evref = np.sort(np.linalg.eigvalsh(Href))
    check("physics", "spin_eigenvalues_vs_analytic",
          np.max(np.abs(ev - evref)) < 1e-12,
          f"E={ev}, |Om|/2={np.sqrt(5) / 2:.12f}, dev={fmt_max(np.max(np.abs(ev - evref)))}")

    t = np.asarray(res["times"], dtype=float)
    psi = np.asarray(res["states"], dtype=complex)
    omega_eff = float(np.sqrt(5.0))
    period = ref.spin_period(2.0, 0.0, 1.0)
    check("physics", "spin_time_grid", t.size == 401
          and abs(t[0]) < 1e-15 and abs(t[-1] - period) < 1e-9,
          f"n={t.size}, t=[{t[0]!r},{t[-1]!r}], period={period!r}")
    check("physics", "spin_scalar_Omega_eff",
          abs(float(res["scalars"].get("Omega_eff", float("nan"))) - omega_eff) < 1e-12,
          f"Omega_eff={res['scalars'].get('Omega_eff')!r}")
    check("physics", "spin_scalar_max_flip_theory",
          abs(float(res["scalars"].get("max_flip_prob_theory", float("nan"))) - 0.8) < 1e-12,
          f"max_flip_prob_theory={res['scalars'].get('max_flip_prob_theory')!r}")

    # independent Taylor expm on the *product* H
    psi_ref = ref.evolve_expm(H, ref.spin_initial_state("up"), t)
    dev_states = np.max(np.abs(psi - psi_ref))
    check("physics", "spin_states_vs_independent_expm", dev_states < 1e-6,
          f"max|psi-psi_ref|={fmt_max(dev_states)}")

    pd = _pop(psi, 1)
    pd_analytic = ref.spin_flip_prob_a1(2.0, 0.0, 1.0, t)
    dev_pop = np.max(np.abs(pd - pd_analytic))
    check("physics", "spin_pop1_vs_analytic", dev_pop < 1e-8,
          f"max|P_down-analytic|={fmt_max(dev_pop)}")
    check("physics", "spin_pop1_peak", abs(pd.max() - 0.8) < 1e-12,
          f"peak P_down={pd.max()!r} (analytic 0.8)")
    i_peak = int(np.argmax(pd))
    check("physics", "spin_peak_time_half_period",
          abs(t[i_peak] - period / 2) < 1e-9,
          f"t_peak={t[i_peak]!r}, T/2={period / 2!r}")
    check("physics", "spin_returns_at_period", abs(pd[-1]) < 1e-12,
          f"P_down(T)={pd[-1]!r}")
    sums = np.sum(np.abs(psi) ** 2, axis=1)
    check("physics", "spin_probability_sum", np.max(np.abs(sums - 1)) < 1e-12,
          f"max|sum-1|={fmt_max(np.max(np.abs(sums - 1)))}")
    # observables consistency with states
    for obs_key, idx, pauli in (("pop_1", 1, None),):
        if obs_key in res["observables"]:
            o = np.asarray(res["observables"][obs_key], dtype=float)
            check("physics", f"spin_observable_{obs_key}_consistent",
                  np.max(np.abs(o - pd)) < 1e-12,
                  f"max dev={fmt_max(np.max(np.abs(o - pd)))}")
    if "expect_z" in res["observables"]:
        ez = np.asarray(res["observables"]["expect_z"], dtype=float)
        check("physics", "spin_expect_z_vs_bloch",
              np.max(np.abs(ez - ref.spin_bloch(2, 0, 1, t, [0, 0, 1])[:, 2])) < 1e-8,
              "expect_z matches closed-form precession")

    # ---------------- spin: Omega_y != 0
    raw = {"model": "spin_half", "params": {"Omega_x": 0.7, "Omega_y": -1.3,
                                            "Omega_z": 0.4},
           "initial_state": {"kind": "plus_x"}}
    req, res = run(raw)
    H = np.asarray(res["hamiltonian"], dtype=complex)
    check("physics", "spin_H_oY_matrix",
          np.max(np.abs(H - ref.spin_hamiltonian(0.7, -1.3, 0.4))) < 1e-12,
          f"max dev={fmt_max(np.max(np.abs(H - ref.spin_hamiltonian(0.7, -1.3, 0.4))))}")
    t = np.asarray(res["times"], dtype=float)
    psi = np.asarray(res["states"], dtype=complex)
    n0 = [1.0, 0.0, 0.0]
    nz_ref = ref.spin_bloch(0.7, -1.3, 0.4, t, n0)[:, 2]
    nz = np.array([float(np.real(np.conj(p) @ ref.SZ @ p)) for p in psi])
    phigh = float((0.7 ** 2 + 1.3 ** 2) / (0.7 ** 2 + 1.3 ** 2 + 0.4 ** 2) *
                  (0.0 + 1.3 ** 2) / (0.7 ** 2 + 1.3 ** 2) + 0)  # unused
    check("physics", "spin_oY_expect_z_vs_bloch",
          np.max(np.abs(nz - nz_ref)) < 1e-8,
          f"max dev={fmt_max(np.max(np.abs(nz - nz_ref)))}")
    # analytic P_down for |+x> = (1 - nz)/2
    dev = np.max(np.abs(_pop(psi, 1) - (1 - nz_ref) / 2))
    check("physics", "spin_oY_pop_vs_analytic", dev < 1e-8,
          f"max|P_down-analytic|={fmt_max(dev)}")
    check("physics", "spin_oY_states_vs_independent_expm",
          np.max(np.abs(psi - ref.evolve_expm(H, ref.spin_initial_state("plus_x"), t))) < 1e-6,
          "states match independent Taylor expm")

    # ---------------- spin: custom initial state
    raw = {"model": "spin_half", "params": {"Omega_x": 1.1, "Omega_z": -0.4},
           "initial_state": {"kind": "custom",
                             "vector": [[0.6, 0.0], [0.0, 0.8]]}}
    req, res = run(raw)
    H = np.asarray(res["hamiltonian"], dtype=complex)
    t = np.asarray(res["times"], dtype=float)
    psi = np.asarray(res["states"], dtype=complex)
    psi0 = ref.spin_initial_state("custom", [[0.6, 0.0], [0.0, 0.8]])
    dev = np.max(np.abs(psi - ref.evolve_expm(H, psi0, t)))
    n0 = [float(np.real(np.conj(psi0) @ s @ psi0)) for s in (ref.SX, ref.SY, ref.SZ)]
    nz = np.array([float(np.real(np.conj(p) @ ref.SZ @ p)) for p in psi])
    dev_analytic = np.max(np.abs(nz - ref.spin_bloch(1.1, 0.0, -0.4, t, n0)[:, 2]))
    check("physics", "spin_custom_states_vs_expm", dev < 1e-6,
          f"max dev={fmt_max(dev)}")
    check("physics", "spin_custom_vs_bloch", dev_analytic < 1e-8,
          f"max dev={fmt_max(dev_analytic)}")
    check("physics", "spin_custom_H_built",
          np.max(np.abs(H - ref.spin_hamiltonian(1.1, 0.0, -0.4))) < 1e-12, "ok")

    # ---------------- spin: large Omega_z suppresses flip peak
    raws = {"small": {"model": "spin_half",
                      "params": {"Omega_x": 1.0, "Omega_z": 0.1},
                      "initial_state": {"kind": "up"}},
            "large": {"model": "spin_half",
                      "params": {"Omega_x": 1.0, "Omega_z": 20.0},
                      "initial_state": {"kind": "up"}}}
    peaks = {}
    for k, raw in raws.items():
        req, res = run(raw)
        peaks[k] = float(_pop(res["states"], 1).max())
    expected = {"small": 1.0 / (1.0 + 0.01), "large": 1.0 / (1.0 + 400.0)}
    check("physics", "spin_large_Omega_z_peak_suppressed",
          peaks["large"] < peaks["small"]
          and abs(peaks["large"] - expected["large"]) < 1e-12
          and abs(peaks["small"] - expected["small"]) < 1e-12,
          f"peak(small)={peaks['small']:.12f} vs {expected['small']:.12f}; "
          f"peak(large)={peaks['large']:.12e} vs {expected['large']:.12e}")

    # ---------------- spin: time reversal on the product Hamiltonian
    Href = ref.spin_hamiltonian(2.0, 0.0, 1.0)
    U_p = ref.expm(-1j * Href * 0.731)
    U_m = ref.expm(1j * Href * 0.731)
    check("physics", "spin_time_reversal_U(t)U(-t)",
          np.max(np.abs(U_p @ U_m - np.eye(2))) < 1e-12,
          f"|U(t)U(-t)-I|={fmt_max(np.max(np.abs(U_p @ U_m - np.eye(2))))}")

    try:
        # ---------------- harmonic oscillator: omega=1, N=40, |0>+|1>
        raw = {"model": "harmonic_oscillator",
               "params": {"omega": 1.0, "n_levels": 40},
               "initial_state": {"kind": "superposition",
                                 "components": [{"n": 0, "amplitude": [1.0, 0.0]},
                                                {"n": 1, "amplitude": [1.0, 0.0]}]}}
        req, res = run(raw)
        H = np.asarray(res["hamiltonian"], dtype=complex)
        Href = ref.ho_hamiltonian(1.0, 40)
        check("physics", "ho_H_matrix", np.max(np.abs(H - Href)) < 1e-12,
              f"max|H-Href|={fmt_max(np.max(np.abs(H - Href)))}")
        ev = np.sort(np.asarray(res["eigen"]["values"]).real)
        check("physics", "ho_spectrum_n_plus_half",
              np.max(np.abs(ev - (np.arange(40) + 0.5))) < 1e-8,
              f"max|E_n-(n+1/2)|={fmt_max(np.max(np.abs(ev - (np.arange(40) + 0.5))))}")
        t = np.asarray(res["times"], dtype=float)
        psi = np.asarray(res["states"], dtype=complex)
        psi0 = ref.ho_superposition(40, [(0, (1.0, 0.0)), (1, (1.0, 0.0))])
        dev = np.max(np.abs(psi - ref.evolve_expm(H, psi0, t)))
        check("physics", "ho_states_vs_independent_expm", dev < 1e-6,
              f"max|psi-psi_ref|={fmt_max(dev)}")
        xop = ref.ho_position(40)
        xt_num = np.array([float(np.real(np.conj(p) @ (xop @ p))) for p in psi])
        xt_analytic = np.cos(t) / np.sqrt(2.0)
        dev_x = np.max(np.abs(xt_num - xt_analytic))
        check("physics", "ho_expect_x_vs_cos(t)/sqrt2", dev_x < 1e-8,
              f"max|⟨x⟩-cos(t)/√2|={fmt_max(dev_x)}")
        if "expect_x" in res["observables"]:
            ox = np.asarray(res["observables"]["expect_x"], dtype=float)
            check("physics", "ho_expect_x_observable_consistent",
                  np.max(np.abs(ox - xt_analytic)) < 1e-8,
                  f"max dev={fmt_max(np.max(np.abs(ox - xt_analytic)))}")
        # independent RK4 on product H
        psi_rk4 = ref.rk4_evolve(H, psi0, t, substeps=4)
        check("physics", "ho_rk4_vs_product_states",
              np.max(np.abs(psi_rk4 - psi)) < 1e-6,
              f"max|RK4-product|={fmt_max(np.max(np.abs(psi_rk4 - psi)))}")

        # coherent state alpha=1
        raw = {"model": "harmonic_oscillator",
               "params": {"omega": 1.0, "n_levels": 40},
               "initial_state": {"kind": "coherent", "alpha": [1.0, 0.0]}}
        req, res = run(raw)
        t = np.asarray(res["times"], dtype=float)
        psi = np.asarray(res["states"], dtype=complex)
        psi_c, leak = ref.ho_coherent(40, (1.0, 0.0))
        xc = np.array([float(np.real(np.conj(p) @ (xop @ p))) for p in psi])
        dev = np.max(np.abs(xc - np.sqrt(2.0) * np.cos(t)))
        check("physics", "ho_coherent_expect_x_sqrt2cos", dev < 1e-8,
              f"max|⟨x⟩-√2cos(t)|={fmt_max(dev)}")
        check("physics", "ho_coherent_leak_reported",
              leak < 1e-12,
              f"reference truncation leak 1-Σ|c_n|²={leak:.3e} (α=1,N=40)")
        check("physics", "ho_coherent_energy_geq_half",
              float(res["scalars"].get("E_mean", res["scalars"].get("energy", 1.0))) >= 0.5 - 1e-12,
              f"scalars keys={sorted(res['scalars'])}")
    except Exception as exc:
        record("physics", "ho_block", "FAIL", f"{type(exc).__name__}: {exc}")

    try:
        # ---------------- perturbation: delta=1, lambda=0.1
        raw = {"model": "perturbation", "params": {"delta": 1.0, "lambda": 0.1},
               "sweep": {"param": "lambda", "values": [0.05, 0.1, 0.2]}}
        req, res = run(raw)
        H = np.asarray(res["hamiltonian"], dtype=complex)
        Href = ref.pert_hamiltonian(1.0, 0.1)
        check("physics", "pert_H_matrix", np.max(np.abs(H - Href)) < 1e-12,
              f"max|H-Href|={fmt_max(np.max(np.abs(H - Href)))}")
        pe = ref.pert_energies(1.0, 0.1)
        sc = res["scalars"]
        got_exact = float(sc.get("E0_exact", float("nan")))
        check("physics", "pert_E0_exact_vs_closed_form",
              abs(got_exact - pe["E0_exact"]) < 1e-8,
              f"E0_exact={got_exact!r} vs -½√(Δ²+λ²)={pe['E0_exact']!r} "
              f"dev={fmt_max(abs(got_exact - pe['E0_exact']))}")
        check("physics", "pert_E0_pt2", abs(float(sc.get("E0_pt2", float("nan")))
                                            - pe["E0_pt2"]) < 1e-8,
              f"E0_pt2={sc.get('E0_pt2')!r} vs {pe['E0_pt2']!r}")
        check("physics", "pert_rel_err_vs_closed_form",
              abs(float(sc.get("rel_err", float("nan"))) - pe["rel_err"]) < 1e-8,
              f"rel_err={sc.get('rel_err')!r} vs {pe['rel_err']!r} "
              f"(lead brief claimed 0.0024937 - flagged as inconsistent)")
        check("physics", "pert_E0_pt1_zero",
              abs(float(sc.get("E0_pt1", float("nan"))) + 0.5) < 1e-12,
              f"E0_pt1={sc.get('E0_pt1')!r}")
    except Exception as exc:
        record("physics", "pert_block", "FAIL", f"{type(exc).__name__}: {exc}")


    # ---------------- perturbation: large lambda -> scope skipped, solver pass
    raw = {"model": "perturbation", "params": {"delta": 1.0, "lambda": 5.0}}
    req, res = run(raw)
    val = validate.validate(req, res)
    by_id = {c["id"]: c for c in val["checks"]}
    scope = by_id.get("perturbation_scope", {})
    solver = by_id.get("solver_reference", {})
    check("physics", "pert_large_lambda_scope_skipped",
          scope.get("status") == "skipped",
          f"status={scope.get('status')!r}, detail={str(scope.get('detail'))[:160]}")
    check("physics", "pert_large_lambda_solver_pass",
          solver.get("status") == "pass",
          f"status={solver.get('status')!r}, metric={solver.get('metric')}")
    check("physics", "pert_large_lambda_overall_not_fail",
          val.get("overall") in ("pass", "partial"),
          f"overall={val.get('overall')!r}")
    check("physics", "pert_lambda_valid_max_present",
          "lambda_valid_max" in str(res.get("scalars"))
          or "lambda_valid_max" in str(res.get("sweep")),
          f"scalars keys={sorted(res['scalars'])}")

    try:
        # ---------------- HO: small N -> truncation diagnostic must fail
        raw = {"model": "harmonic_oscillator",
               "params": {"omega": 1.0, "n_levels": 2},
               "initial_state": {"kind": "coherent", "alpha": [1.0, 0.0]}}
        req, res = run(raw)
        val = validate.validate(req, res)
        by_id = {c["id"]: c for c in val["checks"]}
        tr = by_id.get("truncation_diagnostic", {})
        check("physics", "ho_small_N_truncation_fails",
              tr.get("status") == "fail",
              f"N=2,α=1 status={tr.get('status')!r}, metric={tr.get('metric')}, "
              f"detail={str(tr.get('detail'))[:140]}")
        fail_ids = [c["id"] for c in val["checks"] if c["status"] == "fail"]
        check("physics", "ho_small_N_overall_partial",
              val.get("overall") == "partial" and "truncation_diagnostic" in fail_ids,
              f"overall={val.get('overall')!r}, failed={fail_ids}")
    except Exception as exc:
        record("physics", "ho_truncation_block", "FAIL", f"{type(exc).__name__}: {exc}")



# --------------------------------------------------------------------------
# section: validation report semantics
# --------------------------------------------------------------------------
def section_validation():
    spec = module("spec")
    models = module("models")
    validate = module("validate")
    if any(isinstance(m, Exception) for m in (spec, models, validate)):
        return blocked("validation", "imports",
                       str([repr(m) for m in (spec, models, validate)
                            if isinstance(m, Exception)]))
    rng = np.random.default_rng(20261112)
    cases = []
    cases.append({"model": "spin_half",
                  "params": {"Omega_x": float(rng.uniform(0.2, 4)),
                             "Omega_y": float(rng.uniform(-2, 2)),
                             "Omega_z": float(rng.uniform(-3, 3))},
                  "initial_state": {"kind": "up"},
                  "time": {"t_start": 0.0, "t_end": 3.0, "n_steps": 97}})
    cases.append({"model": "harmonic_oscillator",
                  "params": {"omega": float(rng.uniform(0.3, 3)),
                             "n_levels": int(rng.integers(5, 120))},
                  "initial_state": {"kind": "coherent",
                                    "alpha": [float(rng.uniform(-2, 2)),
                                              float(rng.uniform(-2, 2))]},
                  "time": {"t_start": 0.0, "t_end": 2.0, "n_steps": 61}})
    cases.append({"model": "perturbation",
                  "params": {"delta": float(rng.uniform(0.5, 3)),
                             "lambda": float(rng.uniform(0.01, 4))},
                  "sweep": {"param": "lambda",
                            "values": [0.01 * i for i in range(1, 21)]}})
    for i, raw in enumerate(cases):
        req = spec.normalize_request(raw)
        res = models.run_model(req)
        val = validate.validate(req, res)
        checks = val.get("checks", [])
        ids = [c.get("id") for c in checks]
        counts = val.get("counts", {})
        ok_states = all(c.get("status") in ("pass", "fail", "skipped") for c in checks)
        skipped_reason = all(isinstance(c.get("detail"), str) and c["detail"].strip()
                             for c in checks if c.get("status") == "skipped")
        limits_ok = isinstance(val.get("limits"), list) and len(val["limits"]) > 0 \
            and all(isinstance(x, str) and x.strip() for x in val["limits"])
        counts_ok = (counts.get("pass", 0) + counts.get("fail", 0)
                     + counts.get("skipped", 0)) == len(checks)
        overall_ok = val.get("overall") in ("pass", "partial", "skipped")
        check("validation", f"case{i}_{req['model']}_overall_tristate", overall_ok,
              f"overall={val.get('overall')!r}")
        check("validation", f"case{i}_{req['model']}_states", ok_states,
              f"n_checks={len(checks)}, counts={counts}")
        check("validation", f"case{i}_{req['model']}_skipped_reason", skipped_reason,
              "every skipped check has detail text")
        check("validation", f"case{i}_{req['model']}_limits_nonempty", limits_ok,
              f"limits={val.get('limits')}")
        check("validation", f"case{i}_{req['model']}_counts_consistent", counts_ok,
              f"counts={counts}, len(checks)={len(checks)}")
        check("validation", f"case{i}_{req['model']}_failed_list",
              isinstance(val.get("failed"), list)
              and set(val["failed"]) == {c["id"] for c in checks
                                         if c["status"] == "fail"},
              f"failed={val.get('failed')}")
        # failure semantics: overall==partial iff there is a fail
        has_fail = any(c["status"] == "fail" for c in checks)
        expect = "partial" if has_fail else ("pass" if any(
            c["status"] == "pass" for c in checks) else "skipped")
        check("validation", f"case{i}_{req['model']}_overall_matches_checks",
              val.get("overall") == expect,
              f"overall={val.get('overall')!r}, expected={expect!r}")
        print(f"        validation case{i} ids={ids}")


# --------------------------------------------------------------------------
# section: adversarial CLI
# --------------------------------------------------------------------------
def section_cli(skip_heavy=False):
    if not os.path.exists(CLI):
        return blocked("cli", "entry", "quantum_lab.py missing")
    if not os.path.exists(os.path.join(QL_DIR, "cli.py")):
        return blocked("cli", "cli.py", "missing")
    work = fresh_dir("cli")
    bad_specs = {
        "nan": '{"model":"spin_half","params":{"Omega_x":NaN}}',
        "inf": '{"model":"spin_half","params":{"Omega_x":Infinity}}',
        "string2": '{"model":"spin_half","params":{"Omega_x":"2"}}',
        "hbar2": '{"model":"spin_half","params":{"Omega_x":2},"units":{"hbar":2}}',
        "n_steps1": '{"model":"spin_half","params":{"Omega_x":2},"time":{"t_start":0,"t_end":1,"n_steps":1}}',
        "zero_omega": '{"model":"spin_half","params":{"Omega_x":0,"Omega_y":0,"Omega_z":0}}',
        "omega_neg": '{"model":"harmonic_oscillator","params":{"omega":-1}}',
        "delta0": '{"model":"perturbation","params":{"delta":0}}',
        "n_levels1": '{"model":"harmonic_oscillator","params":{"n_levels":1}}',
        "sweep100": '{"model":"perturbation","params":{"delta":1,"lambda":0.1},"sweep":{"param":"lambda","values":'
                    + json.dumps([0.01 * i for i in range(1, 101)]) + '}}',
        "unknown_top": '{"model":"spin_half","params":{"Omega_x":2},"banana":1}',
        "unknown_param": '{"model":"spin_half","params":{"Omega_x":2,"Omega_q":1}}',
        "bad_model": '{"model":"ising","params":{}}',
        "zero_vector": '{"model":"spin_half","params":{"Omega_x":2},"initial_state":{"kind":"custom","vector":[[0,0],[0,0]]}}',
        "alpha13": '{"model":"harmonic_oscillator","params":{"n_levels":40},"initial_state":{"kind":"coherent","alpha":[13,0]}}',
        "bad_json": '{"model":',
    }
    for name, spec_txt in bad_specs.items():
        code, out, err = cli_run(["run", "--spec", spec_txt], work)
        blob = out + err
        leaks = any(x in blob for x in ("Traceback (most recent call last)",
                                        "File \"", "site-packages"))
        check("cli", "reject_" + name,
              code == 2 and not leaks,
              f"exit={code}, leaks_traceback={leaks}, "
              f"msg={blob.strip().splitlines()[-1][:130] if blob.strip() else '<empty>'}")

    # --set injection tests (baseline must live in the *default* root, because
    # derive has no --out-root flag)
    code, out, err = cli_run(["run", "--spec",
                              '{"model":"spin_half","params":{"Omega_x":2}}'], work)
    base_ok = code == 0
    check("cli", "baseline_run_ok", base_ok,
          f"exit={code}, tail={ (out+err).strip().splitlines()[-1][:120] if (out+err).strip() else ''}")
    ids = []
    root = os.path.join(work, "quantumlab-experiments")
    for d in experiment_dirs(root) if os.path.isdir(root) else []:
        ids.append(os.path.basename(d))
    if ids:
        # A canary file proves whether any injected payload was *executed*;
        # the CLI legitimately echoes the offending string back in its error.
        canary = os.path.join(work, "PWNED_CANARY.txt")
        injections = [
            (f"params.Omega_z=__import__('os').system('echo x > \"{canary}\"')",
             "os_system"),
            (f"params.Omega_z=__import__('pathlib').Path(r'{canary}').write_text('x')",
             "pathlib_write"),
            ("params.Omega_z=2; import os", "semicolon"),
            ("params.Nope=1", "unknown_path"),
            ("params.Omega_z=abc", "nonnumeric"),
            (f"__import__('os').system('echo x > \"{canary}\"')", "bare_payload_path"),
            ("params.Omega_z=1e999", "overflow"),
        ]
        for inj, tag in injections:
            code, out, err = cli_run(["derive", ids[0], "--set", inj], work)
            blob = out + err
            executed = os.path.exists(canary)
            traceback_leak = "Traceback (most recent call last)" in blob
            check("cli", "set_injection_" + tag,
                  code == 2 and not executed and not traceback_leak,
                  f"exit={code}, canary_created={executed}, "
                  f"traceback={traceback_leak}, "
                  f"msg={blob.strip().splitlines()[-1][:110] if blob.strip() else '<empty>'}")
    else:
        blocked("cli", "set_injection", "no baseline experiment recorded")


# --------------------------------------------------------------------------
# section: artifact / report file checks
# --------------------------------------------------------------------------
SECTION_KEYWORDS = ["研究问题", "模型", "参数", "数值方法", "结果", "验证",
                    "解释", "复现", "限制"]


def section_artifacts():
    if not all(os.path.exists(os.path.join(QL_DIR, f))
               for f in ("models.py", "validate.py", "charts.py", "store.py",
                         "report.py", "cli.py")):
        return blocked("artifacts", "stack", "full stack not on disk yet")
    work = fresh_dir("artifacts")
    root = os.path.join(work, "quantumlab-experiments")
    spec_txt = json.dumps({
        "model": "spin_half",
        "params": {"Omega_x": 2.0, "Omega_z": 1.0},
        "initial_state": {"kind": "up"},
        "question": "verifier artifact check: spin Rabi oscillation",
    })
    code, out, err = cli_run(["run", "--spec", spec_txt, "--out-root", root], work)
    check("artifacts", "run_exit0", code == 0,
          f"exit={code}; tail={(out + err).strip().splitlines()[-1][:160] if (out + err).strip() else ''}")
    exp = latest_exp(root) if os.path.isdir(root) else None
    if not exp:
        return blocked("artifacts", "experiment_dir", "no experiment created")
    exp_id = os.path.basename(exp)
    print(f"        artifact experiment: {exp}")
    expected = ["request.json", "result.json", "validation.json", "meta.json",
                "data.csv", "report.html", "figures"]
    for f in expected:
        check("artifacts", f"file_{f}", os.path.exists(os.path.join(exp, f)),
              os.path.join(exp, f))

    fig_dir = os.path.join(exp, "figures")
    pngs = sorted(f for f in os.listdir(fig_dir)) if os.path.isdir(fig_dir) else []
    check("artifacts", "figures_nonempty", len(pngs) >= 1, f"figures={pngs}")
    try:
        from PIL import Image
        sizes, bad = [], []
        for f in pngs:
            p = os.path.join(fig_dir, f)
            with Image.open(p) as im:
                im.verify()
            with Image.open(p) as im:
                sizes.append((f, im.size, im.format))
                if im.size != (3000, 1760) or im.format != "PNG":
                    bad.append((f, im.size, im.format))
        check("artifacts", "png_openable_3000x1760", not bad,
              f"sizes={sizes}" + (f" BAD={bad}" if bad else ""))
    except Exception as exc:  # noqa: BLE001
        check("artifacts", "png_openable_3000x1760", False, repr(exc))

    try:
        html = open(os.path.join(exp, "report.html"), encoding="utf-8").read()
        ext = re.findall(r"""https?://[^\s"'<>]+""", html)
        check("artifacts", "html_no_external_refs", not ext,
              f"n_external={len(ext)}, sample={ext[:3]}")
        n_b64 = html.count("data:image/png;base64,")
        check("artifacts", "html_inline_base64_png", n_b64 >= len(pngs),
              f"inline_png={n_b64}, figure_files={len(pngs)}")
        n_h2 = len(re.findall(r"<h2[\s>]", html))
        missing = [k for k in SECTION_KEYWORDS if k not in html]
        check("artifacts", "html_9_sections",
              n_h2 >= 9 and not missing,
              f"h2_count={n_h2}, missing_keywords={missing}")
        status_ok = bool(re.search(r"(background(-color)?\s*:\s*#[0-9a-fA-F]{3,8})", html)) \
            and any(s in html for s in ("通过", "部分验证未通过", "执行失败"))
        check("artifacts", "html_status_color_block", status_ok,
              "found background-color rule + status wording")
        check("artifacts", "html_utf8_declared",
              'charset="utf-8"' in html.lower() or "charset=utf-8" in html.lower(), "ok")
    except Exception as exc:  # noqa: BLE001
        check("artifacts", "html_checks", False, repr(exc))

    try:
        raw = open(os.path.join(exp, "data.csv"), "rb").read()
        bom = raw.startswith(b"\xef\xbb\xbf")
        text = raw.decode("utf-8-sig")
        header = text.splitlines()[0].strip()
        cols = [c.strip() for c in header.split(",")]
        need = ["t", "pop_0", "pop_1", "expect_x", "expect_y", "expect_z",
                "energy", "pop_1_theory"]
        miss = [c for c in need if c not in cols]
        check("artifacts", "csv_utf8_sig_bom", bom, f"bom={bom}")
        check("artifacts", "csv_columns_complete", not miss,
              f"header={header}" + (f" missing={miss}" if miss else ""))
        check("artifacts", "csv_rows", len(text.splitlines()) >= 400,
              f"lines={len(text.splitlines())}")
    except Exception as exc:  # noqa: BLE001
        check("artifacts", "csv_checks", False, repr(exc))

    # result.json / validation.json structural key checks
    try:
        result = json.load(open(os.path.join(exp, "result.json"), encoding="utf-8"))
        check("artifacts", "result_json_keys",
              all(k in result for k in ("model", "params", "hamiltonian", "eigen",
                                        "times", "states", "observables",
                                        "scalars", "metrics", "figure_specs",
                                        "summary")),
              f"keys={sorted(result)}")
        val = json.load(open(os.path.join(exp, "validation.json"), encoding="utf-8"))
        check("artifacts", "validation_json_keys",
              all(k in val for k in ("checks", "overall", "counts", "failed",
                                     "limits")),
              f"overall={val.get('overall')!r}, keys={sorted(val)}")
        meta = json.load(open(os.path.join(exp, "meta.json"), encoding="utf-8"))
        check("artifacts", "meta_json_keys",
              all(k in meta for k in ("id", "created_at", "model", "params",
                                      "parent_id", "relation", "duration_ms",
                                      "solver", "question")),
              f"id={meta.get('id')!r}, relation={meta.get('relation')!r}")
    except Exception as exc:  # noqa: BLE001
        check("artifacts", "json_structure", False, repr(exc))


# --------------------------------------------------------------------------
# section: end-to-end history workflow
# --------------------------------------------------------------------------
def numeric_maxdiff(a, b):
    """max abs diff over nested numeric JSON structures (lists/floats)."""
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b))
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return float("inf")
        return max([numeric_maxdiff(x, y) for x, y in zip(a, b)] or [0.0])
    if isinstance(a, dict) and isinstance(b, dict):
        return max([numeric_maxdiff(a[k], b[k]) for k in a if k in b] or [0.0])
    return 0.0 if a == b else float("inf")


def section_e2e():
    if not all(os.path.exists(os.path.join(QL_DIR, f))
               for f in ("models.py", "validate.py", "charts.py", "store.py",
                         "report.py", "cli.py")):
        return blocked("e2e", "stack", "full stack not on disk yet")
    work = fresh_dir("e2e")
    root = os.path.join(work, "quantumlab-experiments")
    spec_txt = json.dumps({"model": "spin_half",
                           "params": {"Omega_x": 2.0, "Omega_z": 1.0},
                           "initial_state": {"kind": "up"},
                           "question": "verifier e2e"})
    code, out, err = cli_run(["run", "--spec", spec_txt, "--json"], work)
    check("e2e", "run_json_exit0", code == 0, f"exit={code}")
    payload = None
    if code == 0:
        try:
            payload = parse_json_out(out)
        except Exception as exc:  # noqa: BLE001
            check("e2e", "run_json_parse", False, repr(exc))
    if payload is None:
        return blocked("e2e", "run", f"exit={code}, out={out[-300:]}, err={err[-300:]}")
    check("e2e", "run_json_summary_fields",
          all(k in payload for k in ("id", "dir", "overall", "checks",
                                     "scalars", "figures", "report")),
          f"keys={sorted(payload)}")
    exp1 = payload["id"]
    exp1_dir = payload.get("dir") or os.path.join(root, exp1)
    result1_path = os.path.join(exp1_dir, "result.json")
    check("e2e", "experiment_dir_created", os.path.isdir(exp1_dir), exp1_dir)
    check("e2e", "overall_reported", payload.get("overall") in
          ("pass", "partial", "skipped"), f"overall={payload.get('overall')!r}")

    # list
    code, out, err = cli_run(["list", "--json"], work)
    ok = code == 0
    entries = []
    if ok:
        try:
            obj = parse_json_out(out)
            entries = obj.get("experiments", obj if isinstance(obj, list) else [])
        except Exception:  # noqa: BLE001
            ok = False
    check("e2e", "list_json", ok and any(e.get("id") == exp1 for e in entries),
          f"exit={code}, n={len(entries)}")

    # show
    code, out, err = cli_run(["show", exp1, "--json"], work)
    ok = code == 0
    shown = {}
    if ok:
        try:
            shown = parse_json_out(out)
        except Exception:  # noqa: BLE001
            ok = False
    check("e2e", "show_json", ok and bool(shown),
          f"exit={code}, keys={sorted(shown) if isinstance(shown, dict) else shown}")

    # snapshot parent before rerun/derive
    parent_snapshot = {}
    for f in ("request.json", "result.json", "validation.json"):
        p = os.path.join(exp1_dir, f)
        if os.path.exists(p):
            parent_snapshot[f] = open(p, "rb").read()

    # rerun
    code, out, err = cli_run(["rerun", exp1, "--json"], work)
    ok = code == 0
    rerun = {}
    if ok:
        try:
            rerun = parse_json_out(out)
        except Exception:  # noqa: BLE001
            ok = False
    exp2 = rerun.get("id")
    check("e2e", "rerun_new_id", ok and exp2 and exp2 != exp1,
          f"exit={code}, id={exp2!r} (parent {exp1})")
    if exp2:
        meta2 = {}
        try:
            meta2 = json.load(open(os.path.join(root, exp2, "meta.json"),
                                   encoding="utf-8"))
        except Exception:  # noqa: BLE001
            pass
        check("e2e", "rerun_relation", meta2.get("relation") == "rerun",
              f"relation={meta2.get('relation')!r}, parent_id={meta2.get('parent_id')!r}")
        # identical results to 1e-12
        try:
            r1 = json.load(open(result1_path, encoding="utf-8"))
            r2 = json.load(open(os.path.join(root, exp2, "result.json"),
                                encoding="utf-8"))
            keys = ["times", "states"]
            devs = {k: numeric_maxdiff(r1.get(k), r2.get(k)) for k in keys}
            for k in ("observables", "scalars", "metrics"):
                devs[k] = numeric_maxdiff(r1.get(k, {}), r2.get(k, {}))
            devs["eigen.values"] = numeric_maxdiff(r1["eigen"]["values"],
                                                   r2["eigen"]["values"])
            worst = max(devs.values())
            check("e2e", "rerun_results_identical_1e-12", worst < 1e-12,
                  f"max diff={worst:.3e} per_key={ {k: f'{v:.2e}' for k, v in devs.items()} }")
        except Exception as exc:  # noqa: BLE001
            check("e2e", "rerun_results_identical_1e-12", False, repr(exc))

    # parent not overwritten
    same = True
    for f, blob in parent_snapshot.items():
        p = os.path.join(exp1_dir, f)
        if not os.path.exists(p) or open(p, "rb").read() != blob:
            same = False
    check("e2e", "parent_not_overwritten_after_rerun", same,
          f"files={sorted(parent_snapshot)}")

    # derive with --set
    code, out, err = cli_run(["derive", exp1, "--set", "params.Omega_z=0", "--json"],
                             work)
    ok = code == 0
    derived = {}
    if ok:
        try:
            derived = parse_json_out(out)
        except Exception:  # noqa: BLE001
            ok = False
    exp3 = derived.get("id")
    check("e2e", "derive_new_id", ok and exp3 and exp3 != exp1,
          f"exit={code}, id={exp3!r}")
    if exp3:
        try:
            req3 = json.load(open(os.path.join(root, exp3, "request.json"),
                                  encoding="utf-8"))
            meta3 = json.load(open(os.path.join(root, exp3, "meta.json"),
                                   encoding="utf-8"))
            check("e2e", "derive_set_applied",
                  abs(float(req3["params"]["Omega_z"])) < 1e-15,
                  f"Omega_z={req3['params']['Omega_z']!r}")
            check("e2e", "derive_relation",
                  meta3.get("relation") == "derived"
                  and meta3.get("parent_id") == exp1,
                  f"relation={meta3.get('relation')!r}, parent={meta3.get('parent_id')!r}")
        except Exception as exc:  # noqa: BLE001
            check("e2e", "derive_set_applied", False, repr(exc))
    # parent unchanged after derive
    same = all(os.path.exists(os.path.join(exp1_dir, f))
               and open(os.path.join(exp1_dir, f), "rb").read() == blob
               for f, blob in parent_snapshot.items())
    check("e2e", "parent_not_overwritten_after_derive", same, "ok")

    # compare
    ids = [exp1] + ([exp3] if exp3 else [])
    code, out, err = cli_run(["compare"] + ids + ["--json"], work)
    ok = code == 0
    if ok:
        try:
            cmp_obj = parse_json_out(out)
            ok = "rows" in cmp_obj or "diff" in cmp_obj or "ids" in cmp_obj
        except Exception:  # noqa: BLE001
            ok = False
    check("e2e", "compare_json", ok, f"exit={code}, out={out[:200]}")

    # export html
    code, out, err = cli_run(["export", exp1, "--format", "html"], work)
    rep = os.path.join(exp1_dir, "report.html")
    check("e2e", "export_html", code == 0 and os.path.exists(rep),
          f"exit={code}, report_exists={os.path.exists(rep)}")

    # selfcheck
    code, out, err = cli_run(["selfcheck"], work, timeout=300)
    blob = out + err
    n_pass = blob.count("PASS")
    n_fail = blob.count("FAIL")
    check("e2e", "selfcheck_all_pass", code == 0 and n_fail == 0 and n_pass > 0,
          f"exit={code}, PASS={n_pass}, FAIL={n_fail}, tail={blob.strip().splitlines()[-3:] if blob.strip() else []}")


# --------------------------------------------------------------------------
# section: per-model artifact checks (all three models through the CLI)
# --------------------------------------------------------------------------
MODEL_CSV_COLUMNS = {
    "spin_half": ["t", "pop_0", "pop_1", "expect_x", "expect_y", "expect_z",
                  "energy", "pop_1_theory"],
    "harmonic_oscillator": ["t", "pop_0", "pop_1", "pop_2", "pop_3",
                            "expect_x", "expect_p", "expect_n", "boundary_pop",
                            "expect_x_theory"],
    "perturbation": ["lambda", "E0_pt1", "E0_pt2", "E0_exact", "abs_err",
                     "rel_err", "within_threshold"],
}


def section_model_artifacts():
    if not all(os.path.exists(os.path.join(QL_DIR, f))
               for f in ("models.py", "validate.py", "charts.py", "store.py",
                         "report.py", "cli.py")):
        return blocked("models_artifacts", "stack", "full stack not on disk yet")
    work = fresh_dir("models")
    root = os.path.join(work, "quantumlab-experiments")
    specs = {
        "spin_half": {"model": "spin_half",
                      "params": {"Omega_x": 1.7, "Omega_y": 0.4},
                      "initial_state": {"kind": "plus_y"}},
        "harmonic_oscillator": {
            "model": "harmonic_oscillator",
            "params": {"omega": 1.5, "n_levels": 60},
            "initial_state": {"kind": "coherent", "alpha": [0.7, 0.3]}},
        "perturbation": {"model": "perturbation",
                         "params": {"delta": 1.0, "lambda": 0.2},
                         "sweep": {"param": "lambda",
                                   "values": [0.05, 0.1, 0.2, 0.5]}},
    }
    try:
        from PIL import Image
    except Exception as exc:  # noqa: BLE001
        return blocked("models_artifacts", "PIL", repr(exc))
    for model, raw in specs.items():
        code, out, err = cli_run(["run", "--spec", json.dumps(raw)], work)
        check("models_artifacts", f"{model}_run_exit0", code == 0,
              f"exit={code}, tail={(out + err).strip().splitlines()[-1][:140] if (out + err).strip() else ''}")
        exp = latest_exp(root) if os.path.isdir(root) else None
        if not exp:
            blocked("models_artifacts", f"{model}_dir", "no experiment")
            continue
        exp_id = os.path.basename(exp)
        fig_dir = os.path.join(exp, "figures")
        pngs = sorted(os.listdir(fig_dir)) if os.path.isdir(fig_dir) else []
        bad = []
        for f in pngs:
            p = os.path.join(fig_dir, f)
            try:
                with Image.open(p) as im:
                    im.load()
                    if im.size != (3000, 1760):
                        bad.append((f, im.size))
            except Exception as exc:  # noqa: BLE001
                bad.append((f, repr(exc)))
        check("models_artifacts", f"{model}_figures_3000x1760",
              bool(pngs) and not bad, f"n={len(pngs)}, bad={bad}")
        try:
            html = open(os.path.join(exp, "report.html"), encoding="utf-8").read()
            ext = re.findall(r"""https?://[^\s"'<>]+""", html)
            n_h2 = len(re.findall(r"<h2[\s>]", html))
            n_b64 = html.count("data:image/png;base64,")
            check("models_artifacts", f"{model}_html_inline_no_external",
                  not ext and n_b64 >= len(pngs) and n_h2 >= 9,
                  f"external={len(ext)}, h2={n_h2}, inline_png={n_b64}")
        except Exception as exc:  # noqa: BLE001
            check("models_artifacts", f"{model}_html_inline_no_external", False,
                  repr(exc))
        try:
            raw_bytes = open(os.path.join(exp, "data.csv"), "rb").read()
            text = raw_bytes.decode("utf-8-sig")
            header = [c.strip() for c in text.splitlines()[0].split(",")]
            need = MODEL_CSV_COLUMNS[model]
            miss = [c for c in need if c not in header]
            check("models_artifacts", f"{model}_csv_columns",
                  raw_bytes.startswith(b"\xef\xbb\xbf") and not miss,
                  f"bom={raw_bytes.startswith(b'\xef\xbb\xbf')}, "
                  f"header={header}" + (f" missing={miss}" if miss else ""))
        except Exception as exc:  # noqa: BLE001
            check("models_artifacts", f"{model}_csv_columns", False, repr(exc))
        try:
            val = json.load(open(os.path.join(exp, "validation.json"),
                                 encoding="utf-8"))
            check("models_artifacts", f"{model}_validation_not_fail",
                  val.get("overall") in ("pass", "partial", "skipped")
                  and val.get("limits"),
                  f"overall={val.get('overall')!r}, "
                  f"failed={val.get('failed')}")
        except Exception as exc:  # noqa: BLE001
            check("models_artifacts", f"{model}_validation_not_fail", False,
                  repr(exc))
        print(f"        model artifacts {model}: {exp_id} figures={pngs}")

    # CLI `sweep` subcommand (contract lists it)
    code, out, err = cli_run(["sweep", "--spec",
                              '{"model":"spin_half","params":{"Omega_x":2}}',
                              "--param", "Omega_z", "--values", "0,1,2"], work)
    check("models_artifacts", "cli_sweep_command", code == 0,
          f"exit={code}, tail={(out + err).strip().splitlines()[-1][:140] if (out + err).strip() else ''}")


# --------------------------------------------------------------------------
# section: targeted regression checks requested by the Lead
# --------------------------------------------------------------------------
def section_regressions():
    spec = module("spec")
    models = module("models")
    validate = module("validate")
    if any(isinstance(m, Exception) for m in (spec, models, validate)):
        return blocked("regressions", "imports",
                       str([repr(m) for m in (spec, models, validate)
                            if isinstance(m, Exception)]))

    # 1) a warning from request.json must be printed exactly once by `run`
    work = fresh_dir("regressions")
    root = os.path.join(work, "quantumlab-experiments")
    code, out, err = cli_run(["run", "--spec",
                              '{"model":"spin_half","params":{"Omega_x":2.0}}',
                              "--out-root", root], work)
    exp = latest_exp(root) if os.path.isdir(root) else None
    if code != 0 or not exp:
        record("regressions", "warning_dedup", "FAIL",
               f"run failed: exit={code}, out={out[-200:]}, err={err[-200:]}")
    else:
        try:
            req = json.load(open(os.path.join(exp, "request.json"),
                                 encoding="utf-8"))
            warns = [w for w in (req.get("warnings") or []) if str(w).strip()]
            mult = {w: out.count(w) for w in warns}
            check("regressions", "warning_dedup",
                  warns and all(v == 1 for v in mult.values()),
                  f"n_warnings={len(warns)}, appearances={mult}")
        except Exception as exc:  # noqa: BLE001
            check("regressions", "warning_dedup", False, repr(exc))

    # 2) Omega_z = 0: metrics / scalars / validation must report the same drift
    try:
        req = spec.normalize_request({"model": "spin_half",
                                      "params": {"Omega_x": 2.0}})
        res = models.run_model(req)
        val = validate.validate(req, res)
        m = float(res["metrics"].get("energy_drift", float("nan")))
        s = float(res["scalars"].get("energy_drift", float("nan")))
        ec = [c for c in val["checks"] if c["id"] == "energy_conservation"]
        v = float(ec[0]["metric"]["value"]) if ec else float("nan")
        check("regressions", "omega_z0_energy_drift_consistent",
              abs(m - s) < 1e-15 and abs(m - v) < 1e-15 and m < 1e-12,
              f"metrics={m:.3e}, scalars={s:.3e}, validation={v:.3e}, "
              f"status={ec[0]['status'] if ec else None!r}")
    except Exception as exc:  # noqa: BLE001
        check("regressions", "omega_z0_energy_drift_consistent", False, repr(exc))


    # 3) result-side (new) warnings must still be surfaced, in "补充提示"
    try:
        work3 = fresh_dir("regressions_ho")
        root3 = os.path.join(work3, "quantumlab-experiments")
        ho_spec = json.dumps({"model": "harmonic_oscillator",
                              "params": {"omega": 1.0, "n_levels": 8},
                              "initial_state": {"kind": "coherent",
                                                "alpha": [1.0, 0.0]}})
        code3, out3, err3 = cli_run(["run", "--spec", ho_spec,
                                     "--out-root", root3], work3)
        exp3 = latest_exp(root3) if os.path.isdir(root3) else None
        if code3 not in (0, 1) or not exp3:
            record("regressions", "new_warnings_surfaced", "FAIL",
                   f"run failed: exit={code3}, err={err3[-160:]}")
        else:
            req3 = json.load(open(os.path.join(exp3, "request.json"),
                                  encoding="utf-8"))
            res3 = json.load(open(os.path.join(exp3, "result.json"),
                                  encoding="utf-8"))
            val3 = json.load(open(os.path.join(exp3, "validation.json"),
                                  encoding="utf-8"))
            req_w = [w for w in (req3.get("warnings") or []) if str(w).strip()]
            new_w = [w for w in (res3.get("warnings") or [])
                     if str(w).strip() and w not in req_w]
            mult_req = {w: out3.count(w) for w in req_w}
            mult_new = {w: out3.count(w) for w in new_w}
            check("regressions", "new_warnings_surfaced",
                  all(v == 1 for v in mult_req.values())
                  and new_w and all(v == 1 for v in mult_new.values())
                  and "补充提示" in out3,
                  f"req_each_once={all(v == 1 for v in mult_req.values())}, "
                  f"n_new={len(new_w)}, new_appearances={mult_new}, "
                  f"has_buchong_tishi={'补充提示' in out3}")
            check("regressions", "partial_reported_as_partial",
                  val3.get("overall") == "partial"
                  and "truncation_diagnostic" in val3.get("failed", []),
                  f"overall={val3.get('overall')!r}, failed={val3.get('failed')}, "
                  f"CLI exit={code3} (note: exit 1 is also used for hard "
                  f"execution failure - semantics worth documenting)")
    except Exception as exc:  # noqa: BLE001
        check("regressions", "new_warnings_surfaced", False, repr(exc))


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
SECTIONS = {
    "ref": (section_ref, []),
    "static": (section_static, []),
    "spec": (section_spec, ["spec.py"]),
    "physics": (section_physics, ["models.py", "validate.py"]),
    "validation": (section_validation, ["models.py", "validate.py"]),
    "cli": (section_cli, ["cli.py", "store.py"]),
    "artifacts": (section_artifacts,
                  ["models.py", "validate.py", "charts.py", "store.py",
                   "report.py", "cli.py"]),
    "models_artifacts": (section_model_artifacts,
                         ["models.py", "validate.py", "charts.py", "store.py",
                          "report.py", "cli.py"]),
    "regressions": (section_regressions, ["models.py", "validate.py"]),
    "e2e": (section_e2e,
            ["models.py", "validate.py", "charts.py", "store.py",
             "report.py", "cli.py"]),
}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait", type=int, default=0,
                    help="seconds to poll for teammate files before running")
    ap.add_argument("--section", default=",".join(SECTIONS))
    args = ap.parse_args(argv)
    os.makedirs(ART, exist_ok=True)
    global _ARTIFACT_DIR
    _ARTIFACT_DIR = ART
    chosen = [s.strip() for s in args.section.split(",") if s.strip()]
    unknown = [s for s in chosen if s not in SECTIONS]
    if unknown:
        print(f"unknown sections {unknown}; valid={list(SECTIONS)}")
        return 2

    need = []
    for s in chosen:
        for f in SECTIONS[s][1]:
            p = os.path.join(QL_DIR, f)
            if p not in need:
                need.append(p)
    if need and args.wait > 0:
        missing = wait_for(need, args.wait)
    else:
        missing = [p for p in need if not os.path.exists(p)]

    for s in chosen:
        print(f"\n===== section: {s} =====")
        fn, deps = SECTIONS[s]
        missing_deps = [f for f in deps if not os.path.exists(os.path.join(QL_DIR, f))]
        if missing_deps:
            blocked(s, "deps", f"missing files {missing_deps}")
            continue
        try:
            fn()
        except Exception:  # noqa: BLE001
            record(s, "section_crash", "FAIL", traceback.format_exc()[-1500:])

    n = {"PASS": 0, "FAIL": 0, "BLOCKED": 0, "SKIP": 0}
    for r in RECORDS:
        n[r["status"]] = n.get(r["status"], 0) + 1
    out_path = os.path.join(ART, time.strftime("results-%Y%m%d-%H%M%S.json"))
    json.dump({"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "records": RECORDS, "summary": n},
              open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n===== SUMMARY: {n} =====")
    print(f"results saved to {out_path}")
    if n.get("FAIL"):
        print("FAILURES:")
        for r in RECORDS:
            if r["status"] == "FAIL":
                print(f"  - {r['area']}/{r['name']} :: {r['evidence'][:300]}")
    return 0 if n.get("FAIL", 0) == 0 and n.get("BLOCKED", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
