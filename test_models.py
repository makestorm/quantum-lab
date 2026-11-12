# -*- coding: utf-8 -*-
"""物理内核（models.py + validate.py）单元测试。

运行方式（在 skill 目录下）::

    python -m unittest discover -s tests -p "test_models.py" -v

覆盖点：

- 自旋：Ω=√(Ωx²+Ωz²)、P↓ 峰值 Ω⊥²/|Ω|²、周期 2π/|Ω|、Bloch 解析对比；
- 谐振子：叠加态/相干态/Fock 态的解析 <x>(t)、能量守恒、能谱、截断诊断；
- 微扰：精确值 vs 一阶/二阶（Δ=1, λ=0.1 的基准值按 CONTRACT 的
  E0_exact=-½√(Δ²+λ²) 校正为 -0.5024937810560445）；
- 验证器：14 项检查齐全、状态语义正确、微扰超范围时 perturbation_scope
  必须 skipped（不是 fail）、solver_reference 必须 pass；
- 输出质量：图描述结构、数值有限性、JSON 可序列化、禁止 eval/exec。
"""
from __future__ import annotations

import json
import math
import re
import sys
import unittest
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from qllab import models, spec, util, validate  # noqa: E402

SQRT2 = math.sqrt(2.0)
FIGURE_KINDS = {"line", "spectrum", "error"}
STYLES = {"line", "dashed", "dots", "scatter"}


def run_request(raw):
    """规范化 → 求解 → 验证，返回 (req, result, report)。"""
    req = spec.normalize_request(raw)
    result = models.run_model(req)
    report = validate.validate(req, result)
    return req, result, report


def checks_by_id(report):
    return {item["id"]: item for item in report["checks"]}


def assert_finite_array(testcase, values, label=""):
    arr = np.asarray(values)
    if arr.dtype.kind == "c":
        testcase.assertTrue(np.all(np.isfinite(arr.real)), f"{label} 实部非有限")
        testcase.assertTrue(np.all(np.isfinite(arr.imag)), f"{label} 虚部非有限")
    elif arr.dtype.kind in "fiu":
        testcase.assertTrue(np.all(np.isfinite(arr)), f"{label} 含非有限值")


def walk_arrays(obj, path=""):
    """递归遍历 result，产出所有数值数组（用于有限性断言）。"""
    if obj is None or isinstance(obj, str):
        return
    if isinstance(obj, (bool, np.bool_)):
        return
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind in "fcui" and obj.size:
            yield path, obj
        return
    if isinstance(obj, (int, float, np.integer, np.floating, complex)):
        yield path, np.asarray([obj])
        return
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield from walk_arrays(value, f"{path}.{key}")
        return
    if isinstance(obj, (list, tuple)):
        try:
            arr = np.asarray(obj)
        except (TypeError, ValueError):
            arr = None
        if arr is not None and arr.dtype.kind in "fcui" and arr.size:
            yield path, arr
            return
        for index, item in enumerate(obj):
            yield from walk_arrays(item, f"{path}[{index}]")


class TestSpinHalf(unittest.TestCase):
    """自旋 1/2：拉比振荡与 Bloch 解析参照。"""

    def test_rabi_theory_values(self):
        """Ωx=2, Ωz=1, |↑⟩：Ω=√5，P↓ 峰值 0.8，周期 2π/Ω。"""
        _req, res, report = run_request({
            "model": "spin_half",
            "params": {"Omega_x": 2.0, "Omega_z": 1.0},
            "initial_state": {"kind": "up"},
        })
        omega = math.sqrt(5.0)
        self.assertAlmostEqual(res["scalars"]["Omega_eff"], omega, places=12)
        self.assertAlmostEqual(res["scalars"]["E_plus"], omega / 2.0, places=12)
        self.assertAlmostEqual(res["scalars"]["E_minus"], -omega / 2.0, places=12)
        self.assertAlmostEqual(res["scalars"]["period"], 2.0 * math.pi / omega,
                               places=12)

        pop1 = res["observables"]["pop_1"]
        self.assertAlmostEqual(float(np.max(pop1)), 0.8, places=9)
        self.assertAlmostEqual(res["scalars"]["pop_1_peak_theory"], 0.8, places=9)
        self.assertAlmostEqual(res["scalars"]["max_flip_prob_theory"], 0.8, places=12)
        # t=0 时为 |↑⟩，P↓=0；t=T/2（Ωt=π）时达到峰值。
        self.assertAlmostEqual(float(pop1[0]), 0.0, places=12)
        half = len(pop1) // 2
        self.assertAlmostEqual(float(pop1[half]), 0.8, places=9)
        # 一个完整周期后回到初态。
        self.assertAlmostEqual(float(res["observables"]["expect_z"][-1]), 1.0,
                               places=9)

        self.assertLess(res["scalars"]["analytic_max_abs_dev"], 1e-12)
        eigen = np.sort(res["eigen"]["values"])
        np.testing.assert_allclose(eigen, [-omega / 2.0, omega / 2.0], atol=1e-12)
        self.assertEqual(report["overall"], "pass")
        self.assertEqual(report["failed"], [])

        by_id = checks_by_id(report)
        for check_id in ("input_legality", "finite_values", "hermiticity",
                         "eigen_residual", "orthonormality", "state_norm",
                         "probability", "energy_real", "energy_conservation",
                         "analytic_agreement"):
            self.assertEqual(by_id[check_id]["status"], "pass", check_id)
        # 与自旋无关的检查必须 skipped 且有原因。
        for check_id in ("truncation_diagnostic", "solver_reference",
                         "perturbation_scope", "spectrum_agreement"):
            self.assertEqual(by_id[check_id]["status"], "skipped", check_id)
            self.assertTrue(by_id[check_id]["detail"].strip())

    def test_plus_x_precession_and_start_phase(self):
        """t_start 作为参考相位：非零 t_start 下 Bloch 解析仍应吻合。"""
        omega = 2.0
        raw = {
            "model": "spin_half",
            "params": {"Omega_z": omega},
            "initial_state": {"kind": "plus_x"},
            "time": {"t_start": 1.0, "t_end": 1.0 + 2 * math.pi / omega,
                     "n_steps": 201},
        }
        _req, res, report = run_request(raw)
        self.assertLess(res["scalars"]["analytic_max_abs_dev"], 1e-12)
        # σz=0 时 |+x⟩ 在 σx/σy 平面进动，<σx>(t_start)=1。
        self.assertAlmostEqual(float(res["observables"]["expect_x"][0]), 1.0,
                               places=12)
        self.assertAlmostEqual(
            float(res["observables"]["expect_x"][-1]), 1.0, places=9)
        self.assertEqual(report["failed"], [])

    def test_sweep_records_every_point(self):
        """扫描 Ωx：每个采样点都进入 result["sweep"]，且图里有叠加图。"""
        raw = {
            "model": "spin_half",
            "params": {"Omega_x": 2.0, "Omega_z": 1.0},
            "initial_state": {"kind": "up"},
            "sweep": {"param": "Omega_x", "values": [1.0, 2.0, 3.0]},
        }
        _req, res, _report = run_request(raw)
        sweep = res["sweep"]
        self.assertIsNotNone(sweep)
        self.assertEqual(sweep["param"], "Omega_x")
        self.assertEqual(sweep["values"], [1.0, 2.0, 3.0])
        self.assertEqual(len(sweep["series"]), 3)
        for item, value in zip(sweep["series"], [1.0, 2.0, 3.0]):
            self.assertAlmostEqual(item["value"], value, places=12)
            self.assertIn("scalars", item)
            self.assertIn("observables", item)
            self.assertGreater(item["scalars"]["valid"], 0.5)
            # 从 |↑⟩ 出发：峰值理论值 = Ωx²/(Ωx²+Ωz²)。
            expected = value ** 2 / (value ** 2 + 1.0)
            self.assertAlmostEqual(item["scalars"]["pop_1_peak_theory"], expected,
                                   places=9)
            self.assertAlmostEqual(item["scalars"]["pop_1_peak_numeric"], expected,
                                   delta=1e-3)
            self.assertIn("pop_1", item["observables"])
            self.assertIn("peak", item["observables"]["pop_1"])
        names = [fig["name"] for fig in res["figure_specs"]]
        self.assertIn("sweep_overlay", names)
        overlay = next(f for f in res["figure_specs"] if f["name"] == "sweep_overlay")
        self.assertEqual(overlay["series"][0]["x"], [1.0, 2.0, 3.0])


class TestHarmonicOscillator(unittest.TestCase):
    """量子谐振子：有限 Fock 截断下的能谱与含时演化。"""

    def test_superposition_energy_and_position(self):
        """ω=1, N=40, (|0⟩+|1⟩)/√2：⟨H⟩≡1，⟨x⟩=cos(t)/√2。"""
        _req, res, report = run_request({
            "model": "harmonic_oscillator",
            "params": {"omega": 1.0, "n_levels": 40},
            "initial_state": {"kind": "superposition", "components": [
                {"n": 0, "amplitude": [1.0, 0.0]},
                {"n": 1, "amplitude": [1.0, 0.0]},
            ]},
        })
        times = res["times"]
        energy = res["observables"]["energy"]
        np.testing.assert_allclose(energy, 1.0, atol=1e-12)
        self.assertAlmostEqual(res["scalars"]["energy_mean"], 1.0, places=12)

        expected_x = np.cos(times - times[0]) / SQRT2
        np.testing.assert_allclose(res["observables"]["expect_x"], expected_x,
                                   atol=1e-12)
        np.testing.assert_allclose(res["observables"]["expect_x_theory"],
                                   expected_x, atol=1e-12)
        self.assertLess(res["scalars"]["analytic_max_abs_dev"], 1e-12)
        self.assertLess(res["scalars"]["spectrum_max_abs_dev"], 1e-12)
        self.assertLess(res["scalars"]["boundary_pop_max"], 1e-12)
        self.assertEqual(report["overall"], "pass")
        self.assertEqual(report["failed"], [])
        by_id = checks_by_id(report)
        for check_id in ("state_norm", "probability", "energy_conservation",
                         "analytic_agreement", "spectrum_agreement",
                         "truncation_diagnostic"):
            self.assertEqual(by_id[check_id]["status"], "pass", check_id)

    def test_coherent_state(self):
        """相干态 α=1：⟨x⟩=√2 cos(t)，⟨H⟩=1.5，截断尾部概率可忽略。"""
        _req, res, report = run_request({
            "model": "harmonic_oscillator",
            "params": {"omega": 1.0, "n_levels": 40},
            "initial_state": {"kind": "coherent", "alpha": [1.0, 0.0]},
        })
        times = res["times"]
        np.testing.assert_allclose(res["observables"]["expect_x"],
                                   SQRT2 * np.cos(times), atol=1e-12)
        self.assertAlmostEqual(res["scalars"]["energy_mean"], 1.5, places=12)
        self.assertLess(res["scalars"]["truncation_tail_prob"], 1e-12)
        self.assertLess(res["scalars"]["boundary_pop_max"], 1e-12)
        self.assertLess(res["scalars"]["analytic_max_abs_dev"], 1e-12)
        self.assertEqual(report["failed"], [])

    def test_fock_eigenstate(self):
        """Fock 本征态 |5⟩：⟨H⟩=5.5 且边界占据为 0。"""
        _req, res, _report = run_request({
            "model": "harmonic_oscillator",
            "params": {"omega": 1.0, "n_levels": 40},
            "initial_state": {"kind": "fock", "n": 5},
        })
        np.testing.assert_allclose(res["observables"]["energy"], 5.5, atol=1e-12)
        np.testing.assert_allclose(res["observables"]["expect_n"], 5.0, atol=1e-12)
        self.assertLess(res["scalars"]["boundary_pop_max"], 1e-12)
        self.assertAlmostEqual(float(res["observables"]["pop_0"][0]), 0.0,
                               places=12)
        self.assertAlmostEqual(float(res["observables"]["pop_0"][0]
                                     + res["observables"]["pop_1"][0]), 0.0,
                               places=12)

    def test_spectrum_matches_analytic(self):
        """Eₙ 与 ω(n+½) 逐项一致。"""
        _req, res, _report = run_request({
            "model": "harmonic_oscillator",
            "params": {"omega": 1.7, "n_levels": 12},
            "initial_state": {"kind": "fock", "n": 0},
        })
        expected = 1.7 * (np.arange(12) + 0.5)
        np.testing.assert_allclose(res["eigen"]["values"], expected, atol=1e-12)
        np.testing.assert_allclose(np.asarray(res["scalars"]["spectrum"]),
                                   expected, atol=1e-12)

    def test_sweep_n_levels(self):
        """扫描 N：每组都有截断诊断，截断尾部概率随 N 单调下降。"""
        _req, res, _report = run_request({
            "model": "harmonic_oscillator",
            "params": {"omega": 1.0, "n_levels": 40},
            "initial_state": {"kind": "coherent", "alpha": [3.0, 0.0]},
            "sweep": {"param": "n_levels", "values": [10, 20, 40]},
        })
        sweep = res["sweep"]
        self.assertEqual(sweep["param"], "n_levels")
        self.assertEqual(len(sweep["series"]), 3)
        tails = []
        for item in sweep["series"]:
            self.assertGreater(item["scalars"]["valid"], 0.5)
            self.assertIn("boundary_pop", item["observables"])
            tails.append(item["scalars"]["truncation_tail_prob"])
        self.assertGreater(tails[0], tails[1])
        self.assertGreater(tails[1], tails[2])
        self.assertGreaterEqual(tails[2], 0.0)
        names = [fig["name"] for fig in res["figure_specs"]]
        self.assertIn("sweep_overlay", names)

    def test_fock_state_out_of_range_is_rejected(self):
        """初态 |n⟩ 超出截断维数时给出明确错误，而不是静默截断。"""
        bad = {
            "model": "harmonic_oscillator",
            "params": {"omega": 1.0, "n_levels": 4},
            "initial_state": {"kind": "fock", "n": 10},
        }
        # 1) spec 规范化阶段就会拒绝（SpecError 是 ValueError 子类）。
        with self.assertRaises(spec.SpecError) as ctx:
            spec.normalize_request(dict(bad))
        self.assertIn("n_levels", str(ctx.exception))
        # 2) models 层保留防御性检查：即便绕过 spec，也不能静默截断。
        req = {
            "model": "harmonic_oscillator",
            "params": {"omega": 1.0, "n_levels": 4},
            "initial_state": {"kind": "fock", "n": 10},
            "time": {"t_start": 0.0, "t_end": 1.0, "n_steps": 3},
            "tolerance": dict(spec.DEFAULT_TOL),
            "sweep": None,
            "warnings": [],
        }
        with self.assertRaises(ValueError) as ctx:
            models.run_model(req)
        self.assertIn("超出截断能级范围", str(ctx.exception))


class TestPerturbation(unittest.TestCase):
    """非简并微扰：精确对角化 vs 一阶/二阶展开。"""

    # Δ=1, λ=0.1 的正确基准值（任务书早期版本写错，已按 Lead 更正）。
    BENCH = {
        "E0_pt1": -0.5,
        "E0_pt2": -0.5025,
        "E0_exact": -0.5024937810560445,
        "abs_err": 6.2189439554716e-06,
        "rel_err": 1.2376161039052e-05,
    }

    def test_reference_values(self):
        """Δ=1, λ=0.1：E0_exact=-½√1.01，E0_pt2=-0.5025，rel_err≈1.2376e-5。"""
        _req, res, report = run_request({
            "model": "perturbation",
            "params": {"delta": 1.0, "lambda": 0.1},
            "sweep": {"param": "lambda", "values": [0.1]},
        })
        scalars = res["scalars"]
        for key, expected in self.BENCH.items():
            self.assertAlmostEqual(scalars[key], expected, delta=1e-12,
                                   msg=key)
        # 精确能谱与闭式解交叉校验。
        exact = -0.5 * math.sqrt(1.0 + 0.1 ** 2)
        self.assertAlmostEqual(exact, self.BENCH["E0_exact"], delta=1e-15)
        np.testing.assert_allclose(np.sort(res["eigen"]["values"]),
                                   [-abs(exact), abs(exact)], atol=1e-12)
        # 二阶近似在该 λ 下仍处于适用范围。
        by_id = checks_by_id(report)
        self.assertEqual(by_id["solver_reference"]["status"], "pass")
        self.assertEqual(by_id["perturbation_scope"]["status"], "pass")
        self.assertNotIn("perturbation_scope", report["failed"])

    def test_large_lambda_scope_skipped_not_failed(self):
        """λ=5（λ/Δ≫1）：perturbation_scope 必须 skipped，精确对比必须 pass。"""
        _req, res, report = run_request({
            "model": "perturbation",
            "params": {"delta": 1.0, "lambda": 5.0},
            "sweep": {"param": "lambda", "values": [1.0, 5.0]},
        })
        by_id = checks_by_id(report)
        self.assertEqual(by_id["solver_reference"]["status"], "pass")
        self.assertEqual(by_id["perturbation_scope"]["status"], "skipped")
        self.assertNotIn("perturbation_scope", report["failed"])
        self.assertIn("物理预期", by_id["perturbation_scope"]["detail"])
        self.assertGreater(res["scalars"]["rel_err"], 1.0)
        self.assertLess(res["scalars"]["solver_closed_form_dev"], 1e-12)
        # overall 的语义：没有 fail 项，所以不算 partial。
        self.assertEqual(report["overall"], "pass")
        self.assertEqual(report["counts"]["fail"], 0)

    def test_default_sweep_and_validity_max(self):
        """默认扫描 32 点；lambda_valid_max 只对采样点成立。"""
        _req, res, report = run_request({
            "model": "perturbation",
            "params": {"delta": 1.0, "lambda": 0.1},
        })
        sweep = res["sweep"]
        self.assertEqual(sweep["param"], "lambda")
        # spec.normalize_request 的默认 λ 列表：0.01…0.25（25 点）+ 7 点 = 32。
        self.assertEqual(len(sweep["values"]), 32)
        self.assertEqual(len(sweep["series"]), len(sweep["values"]))
        valid_max = res["scalars"]["lambda_valid_max"]
        self.assertIsNotNone(valid_max)
        self.assertGreater(valid_max, 0.0)
        # 每个采样点都带 E0_exact / rel_err / within_threshold 等诊断量。
        for item in sweep["series"]:
            for key in ("E0_exact", "E0_pt1", "E0_pt2", "abs_err", "rel_err",
                        "within_threshold"):
                self.assertIn(key, item["scalars"])
        by_id = checks_by_id(report)
        self.assertEqual(by_id["solver_reference"]["status"], "pass")


class TestValidationSemantics(unittest.TestCase):
    """验证器的检查项完整性、状态语义与失败识别。"""

    EXPECTED_IDS = [
        "input_legality", "finite_values", "hermiticity", "eigen_residual",
        "orthonormality", "state_norm", "probability", "energy_real",
        "energy_conservation", "analytic_agreement", "spectrum_agreement",
        "truncation_diagnostic", "solver_reference", "perturbation_scope",
    ]

    def _sample_results(self):
        cases = {
            "spin_half": {
                "model": "spin_half",
                "params": {"Omega_x": 2.0, "Omega_z": 1.0},
                "initial_state": {"kind": "up"},
            },
            "harmonic_oscillator": {
                "model": "harmonic_oscillator",
                "params": {"omega": 1.0, "n_levels": 20},
                "initial_state": {"kind": "coherent", "alpha": [1.0, 0.0]},
            },
            "perturbation": {
                "model": "perturbation",
                "params": {"delta": 1.0, "lambda": 0.1},
            },
        }
        return {name: run_request(raw) for name, raw in cases.items()}

    def test_checks_complete_and_well_formed(self):
        """14 项检查齐全；字段完整；计数与 failed 自洽；skipped 必写原因。"""
        for model, (_req, _res, report) in self._sample_results().items():
            ids = [item["id"] for item in report["checks"]]
            self.assertEqual(ids, self.EXPECTED_IDS, model)
            for item in report["checks"]:
                for key in ("id", "name", "status", "metric", "tolerance",
                            "detail", "scope"):
                    self.assertIn(key, item, f"{model}/{item['id']}")
                self.assertIn(item["status"], ("pass", "fail", "skipped"))
                self.assertIn("name", item["metric"])
                self.assertIsInstance(item["metric"]["value"], float)
                self.assertTrue(math.isfinite(item["metric"]["value"]),
                                f"{model}/{item['id']} 指标非有限")
                self.assertIsInstance(item["tolerance"], float)
                self.assertTrue(item["detail"].strip())
                if item["status"] == "skipped":
                    self.assertTrue(item["detail"].strip(),
                                    f"{model}/{item['id']} skipped 未写原因")
            counts = report["counts"]
            self.assertEqual(sum(counts.values()), len(report["checks"]))
            expected_failed = [item["id"] for item in report["checks"]
                               if item["status"] == "fail"]
            self.assertEqual(report["failed"], expected_failed)
            self.assertIn(report["overall"], ("pass", "partial", "skipped"))
            self.assertTrue(report["limits"])
            self.assertTrue(all(isinstance(text, str) for text in report["limits"]))

    def test_model_specific_skips(self):
        """不适用项必须 skipped 并写明原因（含模型名或物理原因）。"""
        samples = self._sample_results()
        spin_report = samples["spin_half"][2]
        spin = checks_by_id(spin_report)
        self.assertEqual(spin["state_norm"]["status"], "pass")
        self.assertEqual(spin["probability"]["status"], "pass")
        self.assertEqual(spin["truncation_diagnostic"]["status"], "skipped")
        self.assertEqual(spin["perturbation_scope"]["status"], "skipped")

        ho_report = samples["harmonic_oscillator"][2]
        ho = checks_by_id(ho_report)
        self.assertEqual(ho["truncation_diagnostic"]["status"], "pass")
        self.assertEqual(ho["spectrum_agreement"]["status"], "pass")
        self.assertEqual(ho["solver_reference"]["status"], "skipped")

        perturb_report = samples["perturbation"][2]
        perturb = checks_by_id(perturb_report)
        for check_id in ("state_norm", "probability", "energy_conservation",
                         "analytic_agreement", "truncation_diagnostic",
                         "spectrum_agreement"):
            self.assertEqual(perturb[check_id]["status"], "skipped", check_id)
        self.assertEqual(perturb["solver_reference"]["status"], "pass")

    def test_energy_drift_matches_validation(self):
        """energy_drift 口径必须与 validate 一致（分母 max(|⟨H⟩|,1)，非 1e-12）。

        回归用例：Ωz=0 时 ⟨H⟩ 恒为 0（纯数值噪声），旧实现在 models 里
        会把它放大成 1e-4 量级，而 validate 给出 1e-16。
        """
        cases = {
            "spin_omega_z_zero": {
                "model": "spin_half",
                "params": {"Omega_x": 2.0},
                "initial_state": {"kind": "up"},
            },
            "spin_sweep": {
                "model": "spin_half",
                "params": {"Omega_x": 2.0, "Omega_z": 1.0},
                "initial_state": {"kind": "up"},
                "sweep": {"param": "Omega_z", "values": [0.0, 1.0]},
            },
            "ho_sweep": {
                "model": "harmonic_oscillator",
                "params": {"omega": 1.0, "n_levels": 40},
                "initial_state": {"kind": "coherent", "alpha": [1.0, 0.0]},
                "sweep": {"param": "n_levels", "values": [10, 40]},
            },
        }
        for name, raw in cases.items():
            with self.subTest(case=name):
                _req, res, report = run_request(raw)
                check = checks_by_id(report)["energy_conservation"]
                checked = check["metric"]["value"]
                self.assertEqual(check["status"], "pass", name)
                self.assertLess(checked, 1e-12, f"{name}: validate 漂移应接近机器精度")
                self.assertAlmostEqual(
                    res["metrics"]["energy_drift"], checked, delta=1e-14,
                    msg=f"{name}: metrics.energy_drift 与 validate 不一致")
                self.assertAlmostEqual(
                    res["scalars"]["energy_drift"], checked, delta=1e-14,
                    msg=f"{name}: scalars.energy_drift 与 validate 不一致")
                self.assertLess(res["scalars"]["energy_drift"], 1e-10,
                                f"{name}: 漂移不应被放大")

    def test_non_hermitian_hamiltonian_is_caught(self):
        """篡改 H 使它非厄米：hermiticity 必须 fail，overall 变 partial。"""
        req, res, _report = run_request({
            "model": "spin_half",
            "params": {"Omega_x": 2.0, "Omega_z": 1.0},
            "initial_state": {"kind": "up"},
        })
        tampered = dict(res)
        hamiltonian = np.array(res["hamiltonian"], dtype=complex, copy=True)
        hamiltonian[0, 1] += 0.3j
        tampered["hamiltonian"] = hamiltonian
        report = validate.validate(req, tampered)
        by_id = checks_by_id(report)
        self.assertEqual(by_id["hermiticity"]["status"], "fail")
        self.assertIn("hermiticity", report["failed"])
        self.assertEqual(report["overall"], "partial")

    def test_nonfinite_values_are_caught(self):
        """结果里注入 NaN：finite_values 必须 fail。"""
        req, res, _report = run_request({
            "model": "spin_half",
            "params": {"Omega_x": 1.0},
            "initial_state": {"kind": "up"},
        })
        tampered = dict(res)
        obs = dict(res["observables"])
        bad = np.array(obs["pop_1"], dtype=float, copy=True)
        bad[3] = float("nan")
        obs["pop_1"] = bad
        tampered["observables"] = obs
        report = validate.validate(req, tampered)
        by_id = checks_by_id(report)
        self.assertEqual(by_id["finite_values"]["status"], "fail")
        self.assertIn("finite_values", report["failed"])


class TestOutputQuality(unittest.TestCase):
    """输出结构、图描述、有限性与可 JSON 化。"""

    def _cases(self):
        return {
            "spin_half": {
                "model": "spin_half",
                "params": {"Omega_x": 2.0, "Omega_z": 1.0},
                "initial_state": {"kind": "up"},
                "sweep": {"param": "Omega_z", "values": [0.0, 1.0]},
            },
            "harmonic_oscillator": {
                "model": "harmonic_oscillator",
                "params": {"omega": 1.0, "n_levels": 20},
                "initial_state": {"kind": "coherent", "alpha": [1.0, 0.0]},
                "sweep": {"param": "n_levels", "values": [10, 20]},
            },
            "perturbation": {
                "model": "perturbation",
                "params": {"delta": 1.0, "lambda": 0.1},
            },
        }

    def test_result_contract_keys(self):
        """result 必须包含契约第 2 节的字段。"""
        required = ("model", "params", "hamiltonian", "basis", "eigen", "times",
                    "states", "observables", "scalars", "analytic", "sweep",
                    "metrics", "figure_specs", "warnings", "summary")
        for model, raw in self._cases().items():
            _req, res, _report = run_request(raw)
            for key in required:
                self.assertIn(key, res, f"{model} 缺少 result[{key!r}]")
            self.assertEqual(res["model"], model)
            self.assertIn("title", res["summary"])
            self.assertTrue(res["summary"]["bullets"])
            self.assertTrue(all(isinstance(b, str) for b in res["summary"]["bullets"]))
            self.assertIsInstance(res["basis"]["labels"], list)
            self.assertEqual(res["basis"]["dim"], int(res["basis"]["dim"]))
            self.assertIsInstance(res["basis"]["truncated"], bool)

    def test_all_numeric_values_finite_and_jsonable(self):
        """所有返回数值有限，且经 util.to_jsonable 后能 json.dumps。"""
        for model, raw in self._cases().items():
            _req, res, _report = run_request(raw)
            for path, array in walk_arrays(res):
                assert_finite_array(self, array, f"{model}:{path}")
            payload = util.to_jsonable(res)
            text = json.dumps(payload, ensure_ascii=False)
            self.assertGreater(len(text), 0)
            report = validate.validate(
                spec.normalize_request(raw), res)
            json.dumps(util.to_jsonable(report), ensure_ascii=False)

    def test_figure_specs_well_formed(self):
        """figspec 名称/坐标轴/曲线长度/有限性合法；扫描模型有叠加图。"""
        for model, raw in self._cases().items():
            _req, res, _report = run_request(raw)
            self.assertTrue(res["figure_specs"], f"{model} 没有 figure_specs")
            for fig in res["figure_specs"]:
                self.assertRegex(fig["name"], r"^[a-z0-9_]+$")
                self.assertIn(fig["kind"], FIGURE_KINDS)
                self.assertTrue(fig["title"].strip())
                self.assertTrue(fig["xlabel"].strip())
                self.assertTrue(fig["ylabel"].strip())
                self.assertTrue(fig["caption"].strip())
                self.assertIsInstance(fig["notes"], list)
                self.assertTrue(fig["notes"])
                self.assertIsInstance(fig["hlines"], list)
                self.assertIsInstance(fig["vlines"], list)
                self.assertTrue(fig["ylim"] is None or len(fig["ylim"]) == 2)
                for series in fig["series"]:
                    self.assertIn(series["style"], STYLES)
                    self.assertEqual(len(series["x"]), len(series["y"]),
                                     f"{model}/{fig['name']} x/y 长度不一致")
                    assert_finite_array(self, np.asarray(series["x"]),
                                        f"{model}/{fig['name']} x")
                    assert_finite_array(self, np.asarray(series["y"]),
                                        f"{model}/{fig['name']} y")
            names = [fig["name"] for fig in res["figure_specs"]]
            self.assertEqual(len(names), len(set(names)), f"{model} 图名重复")
            if raw.get("sweep") is not None:
                self.assertIn("sweep_overlay", names, model)

    def test_no_eval_or_exec_in_physics_sources(self):
        """物理内核源码不含 eval/exec（安全边界）。"""
        for filename in ("models.py", "validate.py"):
            source = (SCRIPT_DIR / "qllab" / filename).read_text(encoding="utf-8")
            self.assertNotIn("eval(", source, filename)
            self.assertNotIn("exec(", source, filename)

    def test_no_scipy_or_matplotlib_import(self):
        """只允许 numpy + 标准库。"""
        for filename in ("models.py", "validate.py"):
            source = (SCRIPT_DIR / "qllab" / filename).read_text(encoding="utf-8")
            self.assertNotIn("import scipy", source, filename)
            self.assertNotIn("import matplotlib", source, filename)


class TestSmokeAllPaths(unittest.TestCase):
    """3 个模型 × 有/无 sweep = 6 条分支的冒烟测试（防止只在某分支触发的错误）。"""

    CASES = {
        "spin_no_sweep": {
            "model": "spin_half",
            "params": {"Omega_x": 2.0, "Omega_z": 1.0},
            "initial_state": {"kind": "up"},
        },
        "spin_sweep": {
            "model": "spin_half",
            "params": {"Omega_x": 2.0, "Omega_z": 1.0},
            "initial_state": {"kind": "up"},
            "sweep": {"param": "Omega_z", "values": [0.0, 1.0, 2.0]},
        },
        "ho_no_sweep": {
            "model": "harmonic_oscillator",
            "params": {"omega": 1.0, "n_levels": 40},
            "initial_state": {"kind": "superposition", "components": [
                {"n": 0, "amplitude": [1.0, 0.0]},
                {"n": 1, "amplitude": [1.0, 0.0]},
            ]},
        },
        "ho_sweep": {
            "model": "harmonic_oscillator",
            "params": {"omega": 1.0, "n_levels": 40},
            "initial_state": {"kind": "coherent", "alpha": [1.0, 0.0]},
            "sweep": {"param": "n_levels", "values": [10, 40]},
        },
        "perturbation_no_sweep": {
            "model": "perturbation",
            "params": {"delta": 1.0, "lambda": 0.1},
            "sweep": None,
        },
        "perturbation_sweep": {
            "model": "perturbation",
            "params": {"delta": 1.0, "lambda": 0.1},
            "sweep": {"param": "lambda",
                      "values": [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]},
        },
    }

    def test_six_paths(self):
        for name, raw in self.CASES.items():
            with self.subTest(case=name):
                _req, res, report = run_request(raw)
                self.assertEqual(res["model"], raw["model"])
                self.assertTrue(res["figure_specs"])
                self.assertIn(report["overall"], ("pass", "partial", "skipped"))
                for path, array in walk_arrays(res):
                    assert_finite_array(self, array, f"{name}:{path}")
                energy_check = checks_by_id(report)["energy_conservation"]
                self.assertTrue(math.isfinite(energy_check["metric"]["value"]),
                                f"{name}: energy_conservation 指标非有限（可能除零）")

    def test_harmonic_superposition_amplitude_and_spectrum(self):
        """|0⟩+|1⟩：⟨x⟩ 振幅 ≈ 1/√2；能谱为 n+0.5。"""
        _req, res, _report = run_request(dict(self.CASES["ho_no_sweep"]))
        self.assertAlmostEqual(res["scalars"]["x_amplitude"], 1.0 / SQRT2,
                               places=9)
        np.testing.assert_allclose(res["eigen"]["values"],
                                   np.arange(40) + 0.5, atol=1e-12)
        self.assertAlmostEqual(res["scalars"]["energy_mean"], 1.0, places=12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
