# -*- coding: utf-8 -*-
"""charts.py / report.py 的单元测试。

用**手工构造**的 canonical request / result / validation / figspec 驱动，
不依赖 models.py，因此在物理内核完成前也能独立保证图表与报告可用。
覆盖：PNG 尺寸与格式、缺字形兜底、render_all 顺序、三态状态、CSV 列、
HTML 关键段落与 base64 内嵌。
"""
from __future__ import annotations

import io
import json
import math
import os
import shutil
import sys
import unittest
import uuid
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

# 临时文件放在工作区内：DSH 沙箱为 workspace-write，系统 temp 可能不可写。
TESTS_DIR = Path(__file__).resolve().parent
TMP_ROOT = TESTS_DIR / "_tmp"
TMP_ROOT.mkdir(parents=True, exist_ok=True)

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from qllab import charts, report  # noqa: E402

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
TARGET_SIZE = (3000, 1760)
CSV_ORDER = ["lambda", "E0_pt1", "E0_pt2", "E0_exact", "abs_err", "rel_err",
             "within_threshold"]


class _TempDir:
    """工作区内的临时目录。

    不用 ``tempfile.TemporaryDirectory``：其创建方式在 DSH Windows 沙箱下
    会得到不可写的目录（错误报告为 workspace-write 拒绝），而普通
    ``os.makedirs`` 子目录可以正常写入。
    """

    def __init__(self, root: Path):
        self.name = str(root / f"t{uuid.uuid4().hex[:10]}")
        os.makedirs(self.name, exist_ok=True)

    def cleanup(self):
        shutil.rmtree(self.name, ignore_errors=True)


def _make_tempdir() -> _TempDir:
    return _TempDir(TMP_ROOT)


# ---------------------------------------------------------------------------
# 合成数据
# ---------------------------------------------------------------------------
def make_spin_result(n_steps: int = 51) -> dict:
    times = np.linspace(0.0, 2.0 * math.pi / math.sqrt(5.0), n_steps)
    omega = math.sqrt(5.0)
    pop_1 = 0.8 * np.sin(omega * times / 2.0) ** 2
    pop_0 = 1.0 - pop_1
    expect_z = pop_0 - pop_1
    expect_x = np.zeros_like(times)
    expect_y = 0.4 * np.sin(omega * times)
    energy = np.full_like(times, -omega / 2.0)
    return {
        "model": "spin_half",
        "params": {"Omega_x": 2.0, "Omega_y": 0.0, "Omega_z": 1.0},
        "hamiltonian": np.array([[0.5, 1.0 - 0.5j], [1.0 + 0.5j, -0.5]]),
        "basis": {"labels": ["up", "down"], "dim": 2, "truncated": False, "note": ""},
        "eigen": {"values": np.array([-omega / 2, omega / 2]),
                  "vectors": np.eye(2)},
        "times": times,
        "states": None,
        "observables": {
            "pop_0": pop_0, "pop_1": pop_1,
            "expect_x": expect_x, "expect_y": expect_y, "expect_z": expect_z,
            "energy": energy,
        },
        "scalars": {"Omega_eff": omega, "E_plus": omega / 2, "E_minus": -omega / 2,
                    "max_flip_prob_theory": 0.8},
        "analytic": {"name": "Bloch 进动解析解", "formula": "P_down(t)=0.8 sin^2(|Ω|t/2)",
                     "values": {"pop_1": pop_1}, "note": "Ω 恒定"},
        "sweep": None,
        "metrics": {"analytic_abs_dev": 1.2e-15, "norm_max_dev": 3.0e-16},
        "figure_specs": [make_figspec("population"), make_figspec("error", ylog=True)],
        "warnings": ["Ωy 采用默认值 0。"],
        "summary": {"title": "自旋跃迁概率", "bullets": [
            "跃迁概率在 0 与 0.8 之间振荡，周期对应 |Ω|=√5。"],
        },
    }


def make_figspec(name: str, ylog: bool = False) -> dict:
    times = [i * 6.283185307179586 / 40 for i in range(41)]
    return {
        "name": name,
        "kind": "error" if ylog else "line",
        "title": "自旋 1/2 跃迁概率：⟨σz⟩(t) 与 P↓ 的振荡（Ωx=2, Ωz=1）",
        "xlabel": "时间 t（ħ/Ω 单位）",
        "ylabel": "⟨σz⟩ 与占据概率 P",
        "ylog": ylog,
        "series": [
            {"label": "pop_1 数值", "x": times,
             "y": [0.8 * math.sin(math.sqrt(5) * t / 2) ** 2 for t in times],
             "style": "line"},
            {"label": "pop_1 解析解（Bloch 进动）", "x": times,
             "y": [0.8 * math.sin(math.sqrt(5) * t / 2) ** 2 for t in times],
             "style": "line"},
            {"label": "pop_0 数值", "x": times,
             "y": [1 - 0.8 * math.sin(math.sqrt(5) * t / 2) ** 2 for t in times],
             "style": "line"},
            {"label": "抽样点", "x": times[::8],
             "y": [0.8 * math.sin(math.sqrt(5) * t / 2) ** 2 for t in times[::8]],
             "style": "scatter"},
        ],
        "hlines": [{"y": 0.8, "label": "理论峰值 0.8", "style": "dashed"}],
        "vlines": [{"x": math.pi / math.sqrt(5), "label": "半周期"}],
        "xlim": None,
        "ylim": None,
        "caption": "跃迁概率随时间振荡",
        "notes": ["Ωx=2.0，Ωz=1.0，|Ω|=√5≈2.236，初态 |↑⟩，ħ=1。",
                  "⟨x⟩(t)、λ≪Δ、≈、⁻¹、m² 等符号均应正常显示。",
                  "纵轴为对数刻度，下限 1e-16。" if ylog else "采样 41 点。"],
        "caption_extra": "",
    }


def make_spin_req() -> dict:
    return {
        "schema": "quantumlab/1",
        "model": "spin_half",
        "params": {"Omega_x": 2.0, "Omega_y": 0.0, "Omega_z": 1.0},
        "initial_state": {"kind": "up", "vector": None},
        "time": {"t_start": 0.0, "t_end": 2.0 * math.pi / math.sqrt(5.0), "n_steps": 51},
        "units": {"hbar": 1.0, "note": "ħ=1，Ω/ω/Δ/λ 与角频率同量纲"},
        "solver": {"method": "eigendecomposition",
                   "note": "H 厄米 → np.linalg.eigh 谱分解，U(t)=V exp(-iΛt) V†"},
        "sweep": None,
        "tolerance": {"hermiticity_abs": 1e-10, "analytic_abs": 1e-6},
        "output": {"figures": True, "formats": ["json", "csv", "png", "html"]},
        "question": "自旋 1/2 的跃迁概率如何随时间变化？",
        "provenance": {"Omega_x": "params", "Omega_z": "params", "Omega_y": "default"},
        "warnings": ["未提供 Omega_y，采用默认值 0（未驱动方向）"],
    }


def make_perturbation_result() -> dict:
    values = [0.1, 0.2, 0.5]
    series = []
    for lam in values:
        exact = -0.5 * math.sqrt(1.0 + lam * lam)
        pt2 = -0.5 - lam * lam / 4.0
        abs_err = abs(pt2 - exact)
        # 与 physics-core 对齐的真实结构：value + 嵌套 scalars / observables
        series.append({
            "value": lam,
            "scalars": {
                "lambda": lam,
                "E0_pt1": -0.5,
                "E0_pt2": pt2,
                "E0_exact": exact,
                "abs_err": abs_err,
                "rel_err": abs_err / abs(exact),
                "within_threshold": abs_err / abs(exact) <= 1e-2,
                "spectrum": [exact, pt2],       # 列表：不得进入 CSV 单元格
            },
            "observables": {"unused": 1.0},     # 时间序列型：不得进扫描表
        })
    return {
        "model": "perturbation",
        "params": {"delta": 1.0, "lambda": 0.1},
        "hamiltonian": np.array([[0.5, 0.05], [0.05, -0.5]]),
        "basis": {"labels": ["0", "1"], "dim": 2, "truncated": False, "note": ""},
        "eigen": {"values": np.array([-0.501249808, 0.501249808]),
                  "vectors": np.eye(2)},
        "times": None,
        "states": None,
        "observables": {},
        "scalars": {"E0_exact": -0.501249808, "E0_pt1": -0.5, "E0_pt2": -0.5025,
                    "abs_err": 0.001250192, "rel_err": 0.002494,
                    "lambda_valid_max": 0.2},
        "analytic": {"name": "闭式解 -½√(Δ²+λ²)", "formula": "E0=-sqrt(delta^2+lambda^2)/2",
                     "values": {}, "note": "精确对角化交叉校验"},
        "sweep": {"param": "lambda", "values": values, "series": series},
        "metrics": {"rel_err": 0.002494, "abs_err": 0.001250192},
        "figure_specs": [make_figspec("lambda_scan")],
        "warnings": [],
        "summary": {"title": "微扰能量对比", "bullets": [
            "二阶近似在 λ=0.1 时相对误差约 2.5e-3。"]},
    }


def make_perturbation_req() -> dict:
    return {
        "schema": "quantumlab/1",
        "model": "perturbation",
        "params": {"delta": 1.0, "lambda": 0.1},
        "initial_state": None,
        "time": None,
        "units": {"hbar": 1.0, "note": "ħ=1，Ω/ω/Δ/λ 与角频率同量纲"},
        "solver": {"method": "eigendecomposition",
                   "note": "H 厄米 → np.linalg.eigh 谱分解"},
        "sweep": {"param": "lambda", "values": [0.1, 0.2, 0.5]},
        "tolerance": {"perturbation_rel": 1e-2},
        "output": {"figures": True, "formats": ["json", "csv", "png", "html"]},
        "question": "λ 扫描时二阶微扰近似的误差如何变化？",
        "provenance": {"delta": "default", "lambda": "params"},
        "warnings": ["未指定扫描，默认扫描扰动强度 λ ∈ [0.01, 2.0]"],
    }


def make_validation(overall: str = "pass", fail: bool = False) -> dict:
    checks = [
        {"id": "hermiticity", "name": "厄米性", "status": "pass",
         "metric": {"name": "max|H-H†|", "value": 0.0}, "tolerance": 1e-10,
         "detail": "线性代数意义上的 H=H†", "scope": "全局"},
        {"id": "state_norm", "name": "态归一化", "status": "pass",
         "metric": {"name": "max||ψ|-1|", "value": 3.0e-16}, "tolerance": 1e-9,
         "detail": "演化全程范数保持", "scope": "时间网格"},
        {"id": "analytic_agreement", "name": "解析结果对比",
         "status": "fail" if fail else "pass",
         "metric": {"name": "max_abs_dev", "value": 4.0e-3 if fail else 1.2e-15},
         "tolerance": 1e-6,
         "detail": "与 Bloch 进动解析解逐点比较", "scope": "时间网格"},
        {"id": "truncation_diagnostic", "name": "截断诊断", "status": "skipped",
         "metric": None, "tolerance": 1e-6, "detail": "不适用于自旋模型", "scope": "—"},
    ]
    counts = {"pass": 0, "fail": 0, "skipped": 0}
    for check in checks:
        counts[check["status"]] += 1
    return {"checks": checks, "overall": overall, "counts": counts,
            "failed": [c["id"] for c in checks if c["status"] == "fail"],
            "limits": ["仅对 Ωx=2、Ωz=1、|↑⟩ 初态、t∈[0,2π/√5] 成立。",
                       "未验证更大 Ωz、含时驱动、开放系统。"]}


def make_meta() -> dict:
    return {"id": "ql-20261112-110359-a1b2", "created_at": "2026-11-12T11:03:59+08:00",
            "model": "spin_half", "params": {"Omega_x": 2.0, "Omega_z": 1.0},
            "parent_id": None, "relation": None, "duration_ms": 42,
            "solver": "eigendecomposition",
            "question": "自旋 1/2 的跃迁概率如何随时间变化？"}


def make_figure_list(result: dict) -> list:
    return charts.render_all(result)


# ---------------------------------------------------------------------------
# charts
# ---------------------------------------------------------------------------
class TestChartsRender(unittest.TestCase):
    """渲染尺寸、格式与健壮性。"""

    def test_render_default_size_and_format(self):
        png = charts.render(make_figspec("population"))
        self.assertTrue(png.startswith(PNG_MAGIC))
        image = Image.open(io.BytesIO(png))
        image.load()
        self.assertEqual(image.format, "PNG")
        self.assertEqual(image.size, TARGET_SIZE)
        self.assertGreater(len(png), 20000)
        self.assertLess(len(png), 4_000_000)

    def test_render_custom_size_and_scale(self):
        png = charts.render(make_figspec("small"), width=800, height=500, scale=2)
        image = Image.open(io.BytesIO(png))
        image.load()
        self.assertEqual(image.size, (1600, 1000))

    def test_render_ylog(self):
        png = charts.render(make_figspec("logplot", ylog=True))
        image = Image.open(io.BytesIO(png))
        image.load()
        self.assertEqual(image.size, TARGET_SIZE)

    def test_render_limits_and_styles(self):
        fig = make_figspec("limits")
        fig["xlim"] = [0.0, 6.2832]
        fig["ylim"] = [0.0, 1.05]
        fig["series"][0]["style"] = "dashed"
        fig["series"][1]["style"] = "dots"
        png = charts.render(fig)
        self.assertEqual(Image.open(io.BytesIO(png)).size, TARGET_SIZE)

    def test_render_empty_and_single_point(self):
        for fig in ({"title": "空数据", "series": []},
                    {"title": "单点", "series": [{"label": "点", "x": [1], "y": [2]}]},
                    {"title": "缺字段"}):
            png = charts.render(fig)
            self.assertEqual(Image.open(io.BytesIO(png)).size, TARGET_SIZE)

    def test_render_all_order_names_caption(self):
        result = make_spin_result()
        result["figure_specs"][0]["name"] = "Fig 1/Population"
        result["figure_specs"][1]["name"] = "error"
        figures = charts.render_all(result)
        self.assertEqual(len(figures), 2)
        self.assertEqual(figures[0]["name"], "fig_1_population")
        self.assertEqual(figures[1]["name"], "error")
        self.assertIn("跃迁概率随时间振荡", figures[0]["caption"])
        self.assertIn("对数", figures[1]["caption"])
        for figure in figures:
            image = Image.open(io.BytesIO(figure["png"]))
            image.load()
            self.assertEqual(image.size, TARGET_SIZE)

    def test_render_all_safe_stem_dedup(self):
        result = {"figure_specs": [{"name": "same"}, {"name": "same"}]}
        names = [figure["name"] for figure in charts.render_all(result)]
        self.assertEqual(names, ["same", "same_2"])

    def test_glyph_fallback_removes_tofu(self):
        """⟨⟩≪⁻¹ 等字体缺失字形必须被替换成可显示写法。"""
        font = charts._font(28)
        source = "⟨x⟩(t)、|↑⟩、λ≪Δ、≈、⁻¹、m²、10¹²"
        safe = charts._glyph_safe(source, font)
        for missing in ("\u27e8", "\u27e9", "\u226a", "\u207b"):
            self.assertNotIn(missing, safe)
        self.assertIn("\u3008", safe)   # 〈
        self.assertIn("\u3009", safe)   # 〉
        self.assertIn("<<", safe)
        self.assertIn("^-1", safe)
        for char in safe:
            if char in "^\n":
                continue
            self.assertTrue(charts._has_glyph(font, char),
                            f"兜底后仍是缺字形：{char!r}")


# ---------------------------------------------------------------------------
# bundle 与状态
# ---------------------------------------------------------------------------
class TestBundle(unittest.TestCase):
    """bundle 汇总与三态语义。"""

    def test_bundle_pass(self):
        result = make_spin_result()
        data = report.bundle(make_spin_req(), result, make_validation("pass"),
                             make_figure_list(result), make_meta())
        self.assertEqual(data["status"], "pass")
        self.assertEqual(data["status_label"], "通过")
        self.assertEqual(data["status_class"], "ok")
        self.assertEqual(len(data["sections"]), 9)
        self.assertIn("研究问题与实验目的", [s["title"] for s in data["sections"]])
        self.assertEqual(len(data["figures"]), 2)
        self.assertTrue(data["figures"][0]["png_b64"])
        self.assertGreater(len(data["key_values"]), 0)
        self.assertTrue(data["interpretation"]["what"])
        self.assertTrue(data["interpretation"]["why"])
        self.assertTrue(data["limits"])
        # 参数来源标注
        sources = {row["name"]: row["source_text"] for row in data["params"]}
        self.assertEqual(sources["Omega_x"], "用户给定")
        self.assertEqual(sources["Omega_y"], "默认值")
        # 复现 JSON 可解析
        json.loads(data["reproduction"]["json_text"])

    def test_bundle_partial(self):
        result = make_spin_result()
        data = report.bundle(make_spin_req(), result, make_validation("partial", fail=True),
                             [], make_meta())
        self.assertEqual(data["status_label"], "部分验证未通过")
        self.assertEqual(data["status_class"], "warn")
        self.assertEqual(data["validation"]["counts"]["fail"], 1)
        failing = [c for c in data["checks"] if c["status"] == "fail"]
        self.assertEqual(len(failing), 1)
        self.assertEqual(failing[0]["status_label"], "未通过")

    def test_bundle_skipped_validation(self):
        validation = {"checks": [{"id": "x", "name": "不适用", "status": "skipped",
                                  "metric": None, "tolerance": None,
                                  "detail": "本模型不需要", "scope": "—"}],
                      "overall": "skipped", "counts": {"pass": 0, "fail": 0, "skipped": 1},
                      "failed": [], "limits": []}
        data = report.bundle(make_spin_req(), make_spin_result(), validation, [], make_meta())
        self.assertEqual(data["status"], "skipped")
        self.assertIn("跳过", data["status_label"])

    def test_bundle_execution_failure(self):
        data = report.bundle(make_spin_req(), None, None, [], make_meta())
        self.assertFalse(data["executed"])
        self.assertEqual(data["status_label"], "执行失败（没有结果）")
        self.assertEqual(data["status_class"], "bad")
        self.assertEqual(data["figures"], [])
        self.assertEqual(data["key_values"], [])
        self.assertTrue(any("没有" in item for item in data["interpretation"]["what"]))
        self.assertIn("执行失败", data["status_label"])

    def test_bundle_error_result(self):
        data = report.bundle(make_spin_req(), {"error": "数值发散"}, None, [], make_meta())
        self.assertEqual(data["status"], "fail")
        self.assertIn("数值发散", data["status_detail"])

    def test_bundle_perturbation_sweep_table(self):
        result = make_perturbation_result()
        data = report.bundle(make_perturbation_req(), result, make_validation("pass"),
                             make_figure_list(result), make_meta())
        self.assertIsNotNone(data["sweep_table"])
        self.assertEqual(data["sweep_table"]["param"], "lambda")
        self.assertIn("lambda", data["sweep_table"]["columns"])
        self.assertIn("rel_err", data["sweep_table"]["columns"])
        self.assertNotIn("observables", data["sweep_table"]["columns"])
        self.assertNotIn("scalars", data["sweep_table"]["columns"])
        self.assertNotIn("spectrum", data["sweep_table"]["columns"])
        self.assertEqual(data["sweep_table"]["columns"][:7], CSV_ORDER)
        self.assertEqual(len(data["sweep_table"]["rows"]), 3)


# ---------------------------------------------------------------------------
# JSON / CSV
# ---------------------------------------------------------------------------
class TestWriters(unittest.TestCase):
    """写盘格式与列约定。"""

    def setUp(self):
        self.tmp = _make_tempdir()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def test_write_json_roundtrip(self):
        path = self.dir / "report.json"
        result = make_spin_result()
        data = report.bundle(make_spin_req(), result, make_validation("pass"), [], make_meta())
        report.write_json(str(path), data)
        loaded = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(loaded["status_label"], "通过")
        self.assertTrue(loaded["generated_at"])
        self.assertEqual(loaded["schema"], report.SCHEMA)

    def test_write_csv_spin_time_series(self):
        path = self.dir / "data.csv"
        result = make_spin_result(n_steps=51)
        report.write_csv(str(path), result)
        raw = path.read_bytes()
        self.assertTrue(raw.startswith(b"\xef\xbb\xbf"), "CSV 必须是 utf-8-sig")
        text = raw.decode("utf-8-sig")
        lines = text.strip().splitlines()
        header = lines[0].split(",")
        self.assertEqual(header[0], "t")
        for column in ("pop_0", "pop_1", "expect_x", "expect_y", "expect_z",
                       "energy", "pop_1_theory"):
            self.assertIn(column, header)
        self.assertEqual(len(lines), 1 + 51)

    def test_write_csv_perturbation_sweep(self):
        path = self.dir / "data.csv"
        result = make_perturbation_result()
        report.write_csv(str(path), result)
        text = path.read_bytes().decode("utf-8-sig")
        lines = text.strip().splitlines()
        header = lines[0].split(",")
        self.assertEqual(header[0], "lambda")
        for column in ("E0_pt1", "E0_pt2", "E0_exact", "abs_err", "rel_err",
                       "within_threshold"):
            self.assertIn(column, header)
        # 契约：不得出现 dict 字符串单元格 / 嵌套子字典列
        self.assertNotIn("observables", header)
        self.assertNotIn("scalars", header)
        self.assertNotIn("spectrum", header, "list 值不得进入 CSV 单元格")
        self.assertNotIn("{", text)
        self.assertEqual(len(lines), 1 + 3)
        # 列序严格按 references/report-format.md:40
        self.assertEqual([name for name in CSV_ORDER if name in header], CSV_ORDER[:7])

    def test_write_csv_failure_placeholder(self):
        path = self.dir / "data.csv"
        report.write_csv(str(path), None)
        text = path.read_bytes().decode("utf-8-sig")
        self.assertIn("执行失败", text)

    def test_write_csv_sweep_sibling(self):
        path = self.dir / "data.csv"
        result = make_spin_result()
        # 与 physics-core 对齐的嵌套结构：param 列取 value，标量来自 scalars
        result["sweep"] = {
            "param": "Omega_z", "values": [0.0, 1.0],
            "series": [
                {"value": 0.0, "scalars": {"pop_1_max": 1.0, "spectrum": [0.0, 0.5]},
                 "observables": {"pop_1": [0.0, 1.0]}},
                {"value": 1.0, "scalars": {"pop_1_max": 0.8, "spectrum": [0.0, 0.5]},
                 "observables": {"pop_1": [0.0, 0.8]}},
            ],
        }
        report.write_csv(str(path), result)
        sweep_path = self.dir / "sweep.csv"
        self.assertTrue(sweep_path.exists())
        text = sweep_path.read_bytes().decode("utf-8-sig")
        header = text.splitlines()[0].split(",")
        self.assertEqual(header[0], "Omega_z")
        self.assertIn("pop_1_max", header)
        self.assertNotIn("spectrum", header)
        self.assertNotIn("observables", header)
        self.assertNotIn("{", text)
        self.assertEqual(len(text.strip().splitlines()), 3)


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------
class TestHtmlReport(unittest.TestCase):
    """单文件 HTML：base64 内嵌、章节齐全、三态可辨。"""

    def setUp(self):
        self.tmp = _make_tempdir()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def _write(self, status: str = "pass", fail: bool = False, result=None):
        result = make_spin_result() if result is None else result
        data = report.bundle(make_spin_req(), result, make_validation(status, fail),
                             make_figure_list(result), make_meta())
        path = self.dir / "report.html"
        report.write_html(str(path), data)
        return path.read_text(encoding="utf-8")

    def test_html_contains_base64_and_sections(self):
        text = self._write("pass")
        self.assertTrue(text.startswith("<!DOCTYPE html>"))
        self.assertTrue(text.rstrip().endswith("</html>"))
        self.assertIn("data:image/png;base64,", text)
        self.assertNotIn("http://", text.split("<style>")[0])  # 无外部资源
        for title in ("研究问题与实验目的", "模型公式与假设", "参数、初态与单位",
                      "数值方法", "结果图像与关键数据", "物理一致性验证",
                      "结果解释与适用限制", "复现配置", "附录：生成信息与文件清单"):
            self.assertIn(title, text, f"缺少章节：{title}")
        self.assertIn("实验编号", text)
        self.assertIn("生成时间", text)
        self.assertIn("ql-20261112-110359-a1b2", text)
        self.assertIn("自旋 1/2 的跃迁概率如何随时间变化？", text)
        self.assertIn("badge ok", text)
        self.assertIn("通过", text)
        self.assertIn("Ωx（横向驱动）", text)
        self.assertIn("Omega_x", text)
        self.assertIn("用户给定", text)
        self.assertIn("默认值", text)
        self.assertIn("data.csv", text)
        self.assertIn("figures/population.png", text)

    def test_html_partial_status_visible(self):
        text = self._write("partial", fail=True)
        self.assertIn("badge warn", text)
        self.assertIn("部分验证未通过", text)
        self.assertIn("未通过", text)

    def test_html_execution_failure(self):
        data = report.bundle(make_spin_req(), None, None, [], make_meta())
        path = self.dir / "failed.html"
        report.write_html(str(path), data)
        text = path.read_text(encoding="utf-8")
        self.assertIn("执行失败（没有结果）", text)
        self.assertIn("badge bad", text)
        self.assertIn("复现配置", text)

    def test_html_escapes_request_json(self):
        text = self._write("pass")
        self.assertIn("&quot;schema&quot;", text)
        self.assertNotIn("<script>", text)


def tearDownModule():
    shutil.rmtree(TMP_ROOT, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
