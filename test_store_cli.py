# -*- coding: utf-8 -*-
"""qllab.store / qllab.cli 的单元测试。

队友模块（models / validate / charts / report）可能尚未完成，这里用
``sys.modules`` 注入假模块，只验证 store 与 cli 自身的行为：

* 实验目录结构、index 原子写、find/get/compare、派生关系；
* CLI 退出码约定（0/1/2）、``--json`` 摘要、run/derive/rerun/compare/export/selfcheck。
"""
from __future__ import annotations

import contextlib
import io
import json
import math
import os
import pathlib
import re
import secrets
import sys
import tempfile
import types
import unittest
from unittest import mock

SCRIPTS_DIR = pathlib.Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import qllab  # noqa: E402
from qllab import cli, spec, store  # noqa: E402

_ID_RE = re.compile(r"^ql-\d{8}-\d{6}-[0-9a-f]{4}$")
_PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 24
_PATCHED = ("qllab.models", "qllab.validate", "qllab.charts", "qllab.report")


# ------------------------------------------------------------------ 假依赖
def _fake_write_csv(path, result):
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        fh.write("t,pop_0,pop_1\n0,1,0\n1,0,1\n")


def _fake_write_json(path, payload):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)


def _fake_write_html(path, bundle):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("<!doctype html><html><body>fake report</body></html>")


def _fake_bundle(req, result, validation, figures, meta):
    return {"meta": meta, "model": req.get("model"),
            "figure_count": len(figures or [])}


def _fake_run_model(req):
    """给出与三条 selfcheck 基准一致的确定性结果。"""
    model = req["model"]
    params = dict(req["params"])
    common = {
        "model": model,
        "params": params,
        "basis": {"labels": [], "dim": 2, "truncated": False, "note": "测试"},
        "times": [0.0, 1.0],
        "states": None,
        "analytic": None,
        "sweep": None,
        "figure_specs": [{"name": "population"}],
        "warnings": [],
        "summary": {"title": "测试结果", "bullets": ["测试用途"]},
    }
    if model == "spin_half":
        omega = math.sqrt(sum(float(params[k]) ** 2
                              for k in ("Omega_x", "Omega_y", "Omega_z")))
        peak = (float(params["Omega_x"]) ** 2
                + float(params["Omega_y"]) ** 2) / omega ** 2
        common.update({
            "hamiltonian": [[[0, 0], [1, 0]], [[1, 0], [0, 0]]],
            "eigen": {"values": [-omega / 2.0, omega / 2.0],
                      "vectors": [[[1, 0], [0, 0]], [[0, 0], [1, 0]]]},
            "observables": {"pop_0": [1.0, 1.0 - peak],
                            "pop_1": [0.0, peak],
                            "expect_x": [0.0, 1.0]},
            "scalars": {"Omega_eff": omega, "max_flip_prob_theory": peak},
            "analytic": {"name": "Bloch", "formula": "n(t)",
                         "values": {"max_flip_prob_theory": peak},
                         "note": "测试"},
            "metrics": {"analytic_abs": 0.0},
        })
        return common
    if model == "harmonic_oscillator":
        omega = float(params["omega"])
        levels = int(params["n_levels"])
        common.update({
            "hamiltonian": [[[0, 0]]],
            "eigen": {"values": [omega * (i + 0.5) for i in range(levels)],
                      "vectors": []},
            "observables": {"expect_x": [0.0, 1.0 / math.sqrt(2.0)],
                            "pop_0": [0.5, 1.0], "pop_1": [0.5, 0.0]},
            "scalars": {"spectrum_max_dev": 0.0},
            "metrics": {"spectrum_abs": 0.0},
        })
        return common
    if model == "perturbation":
        delta = float(params["delta"])
        lam = float(params["lambda"])
        exact = -0.5 * math.sqrt(delta ** 2 + lam ** 2)
        pt1 = -delta / 2.0
        pt2 = -delta / 2.0 - lam ** 2 / (4.0 * delta)
        abs_err = abs(pt2 - exact)
        rel_err = abs_err / abs(exact)
        common.update({
            "hamiltonian": [[[0, 0], [1, 0]], [[1, 0], [0, 0]]],
            "eigen": {"values": [exact, -exact], "vectors": []},
            "observables": {},
            "scalars": {"E0_pt1": pt1, "E0_pt2": pt2, "E0_exact": exact,
                        "abs_err": abs_err, "rel_err": rel_err,
                        "lambda_valid_max": lam},
            "metrics": {"perturbation_rel": rel_err},
        })
        return common
    raise ValueError(f"未知模型 {model}")


def _fake_validate(req, result):
    forced = "fail" in str(req.get("question") or "").lower()
    checks = [{
        "id": "hermiticity", "name": "厄米性", "status": "pass",
        "metric": {"name": "max|H-H†|", "value": 0.0},
        "tolerance": 1e-10, "detail": "测试", "scope": "全矩阵",
    }]
    if forced:
        checks.append({
            "id": "forced_fail", "name": "强制失败", "status": "fail",
            "metric": {"name": "deviation", "value": 1.0},
            "tolerance": 1e-12, "detail": "测试用失败项", "scope": "测试",
        })
    counts = {"pass": 0, "fail": 0, "skipped": 0}
    for check in checks:
        counts[check["status"]] += 1
    failed = [c["id"] for c in checks if c["status"] == "fail"]
    return {
        "checks": checks,
        "overall": "partial" if failed else "pass",
        "counts": counts,
        "failed": failed,
        "limits": ["测试用途，不代表真实物理验证"],
    }


def _fake_render(figspec, **kwargs):
    return _PNG_BYTES


def _fake_render_all(result):
    specs = result.get("figure_specs") or [{"name": "figure"}]
    return [{"name": str(item.get("name") or f"fig{index}"),
             "png": _PNG_BYTES, "caption": "测试图"}
            for index, item in enumerate(specs)]


def _fake_modules():
    return {
        "qllab.models": _make_module("qllab.models", run_model=_fake_run_model),
        "qllab.validate": _make_module("qllab.validate", validate=_fake_validate),
        "qllab.charts": _make_module("qllab.charts", render=_fake_render,
                                     render_all=_fake_render_all),
        "qllab.report": _make_module(
            "qllab.report",
            write_csv=_fake_write_csv,
            write_json=_fake_write_json,
            write_html=_fake_write_html,
            bundle=_fake_bundle,
        ),
    }


def _make_module(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _install_fakes():
    saved = {"sys": {}, "attrs": {}}
    for name in _PATCHED:
        saved["sys"][name] = sys.modules.get(name)
        attr = name.split(".", 1)[1]
        saved["attrs"][attr] = getattr(qllab, attr, None)
    for name, module in _fake_modules().items():
        sys.modules[name] = module
        setattr(qllab, name.split(".", 1)[1], module)
    return saved


def _restore_fakes(saved):
    for name, old in saved["sys"].items():
        if old is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = old
    for attr, old in saved["attrs"].items():
        if old is None:
            if hasattr(qllab, attr):
                delattr(qllab, attr)
        else:
            setattr(qllab, attr, old)


class FakeDepsCase(unittest.TestCase):
    """注入/还原假依赖模块的基类。"""

    @classmethod
    def setUpClass(cls):
        cls._saved_deps = _install_fakes()
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        _restore_fakes(cls._saved_deps)
        super().tearDownClass()


# ------------------------------------------------------------------ 测试辅助
def _canonical(raw):
    return spec.normalize_request(raw)


def _spin_raw(omega_x=2.0, omega_z=1.0, question="自旋实验"):
    return {"model": "spin_half",
            "params": {"Omega_x": omega_x, "Omega_z": omega_z},
            "initial_state": "up",
            "time": {"t_start": 0.0, "t_end": 2.0 * math.pi / math.sqrt(5.0),
                     "n_steps": 101},
            "question": question}


def _ho_raw(question="谐振子实验"):
    return {"model": "harmonic_oscillator", "params": {"omega": 1.0,
                                                       "n_levels": 8},
            "initial_state": {"kind": "superposition"}, "question": question}


def _create(root, raw, relation=None, parent_id=None):
    req = _canonical(raw)
    result = _fake_run_model(req)
    validation = _fake_validate(req, result)
    figures = [{"name": "population", "png": _PNG_BYTES, "caption": "测试图"}]
    return store.create(root, req, result, validation, figures,
                        {"relation": relation, "parent_id": parent_id,
                         "duration_ms": 7, "solver": "eigendecomposition"})


def _run_cli(argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(argv)
    return code, out.getvalue(), err.getvalue()


def _make_temp_dir(prefix):
    """在技能目录下建普通权限临时目录。

    沙箱下 ``tempfile.mkdtemp`` 的 0o700 目录对受限令牌不可写，故改用
    ``os.makedirs``（普通继承权限）。
    """
    base = str(SCRIPTS_DIR.parent)
    for _ in range(20):
        path = os.path.join(base, f"{prefix}{secrets.token_hex(4)}")
        try:
            os.makedirs(path, exist_ok=False)
            return path
        except FileExistsError:
            continue
    return tempfile.mkdtemp(prefix=prefix, dir=base)


# ------------------------------------------------------------------ store 测试
class TestStore(FakeDepsCase):
    def setUp(self):
        # 临时目录放在工作区内：DSH 文件策略可能禁止写系统临时目录
        self.root = _make_temp_dir("qllab-store-test-")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.root, ignore_errors=True)

    def test_create_writes_expected_layout(self):
        exp_id = _create(self.root, _spin_raw())
        self.assertRegex(exp_id, _ID_RE)
        exp_dir = os.path.join(self.root, exp_id)
        for name in ("request.json", "result.json", "validation.json",
                     "meta.json", "data.csv", "report.html"):
            self.assertTrue(os.path.isfile(os.path.join(exp_dir, name)),
                            f"缺少 {name}")
        figures = os.listdir(os.path.join(exp_dir, "figures"))
        self.assertEqual(figures, ["population.png"])
        self.assertTrue(os.path.isfile(os.path.join(self.root, "index.json")))
        # 不应残留临时文件
        leftovers = [f for f in os.listdir(exp_dir) if ".tmp-" in f]
        self.assertEqual(leftovers, [])

        meta = store._read_json(os.path.join(exp_dir, "meta.json"))
        self.assertEqual(meta["id"], exp_id)
        self.assertEqual(meta["model"], "spin_half")
        self.assertIsNone(meta["relation"])
        self.assertEqual(meta["duration_ms"], 7)

        payload = store._read_json(os.path.join(self.root, "index.json"))
        self.assertEqual(payload["version"], 1)
        entry = payload["experiments"][0]
        self.assertEqual(entry["id"], exp_id)
        self.assertEqual(entry["overall"], "pass")
        self.assertEqual(entry["dir"], exp_id)
        self.assertEqual(entry["figures"], ["population"])

    def test_create_generates_unique_ids(self):
        first = _create(self.root, _spin_raw())
        second = _create(self.root, _spin_raw())
        self.assertNotEqual(first, second)

    def test_find_filters_and_orders(self):
        stamps = ["2026-01-01T00:00:01+08:00",
                  "2026-01-01T00:00:02+08:00",
                  "2026-01-01T00:00:03+08:00"]
        with mock.patch("qllab.util.now_iso", side_effect=stamps):
            exp_a = _create(self.root, _spin_raw(2.0, 1.0, "自旋实验A"))
            exp_b = _create(self.root, _spin_raw(3.0, 0.0, "自旋实验B"))
            exp_c = _create(self.root, _ho_raw("谐振子实验"))

        self.assertEqual([e["id"] for e in store.find(self.root)],
                         [exp_c, exp_b, exp_a])
        self.assertEqual([e["id"] for e in store.find(self.root,
                                                      {"model": "spin_half"})],
                         [exp_b, exp_a])
        # 浮点 1e-9 容差内匹配
        matched = store.find(self.root, {"params": {"Omega_x": 2.0 + 1e-10}})
        self.assertEqual([e["id"] for e in matched], [exp_a])
        self.assertEqual(store.find(self.root, {"params": {"Omega_x": 2.5}}), [])
        # 文本子串（大小写不敏感）
        self.assertEqual([e["id"] for e in store.find(self.root,
                                                      {"text": "自旋"})],
                         [exp_b, exp_a])
        self.assertEqual([e["id"] for e in store.find(self.root, {"limit": 1})],
                         [exp_c])

    def test_get_returns_files_and_rejects_bad_id(self):
        exp_id = _create(self.root, _spin_raw())
        exp = store.get(self.root, exp_id)
        self.assertEqual(exp["meta"]["id"], exp_id)
        self.assertEqual(exp["request"]["model"], "spin_half")
        self.assertEqual(exp["validation"]["overall"], "pass")
        self.assertIn("population", exp["figures"])
        self.assertTrue(os.path.isfile(exp["figures"]["population"]))
        self.assertTrue(exp["report"].endswith("report.html"))
        with self.assertRaises(ValueError):
            store.get(self.root, "../../etc/passwd")
        with self.assertRaises(KeyError):
            store.get(self.root, "ql-20260101-000000-dead")

    def test_load_index_recovers_from_missing_or_corrupt_index(self):
        exp_id = _create(self.root, _spin_raw())
        index_path = os.path.join(self.root, "index.json")

        os.remove(index_path)
        self.assertEqual([e["id"] for e in store.load_index(self.root)],
                         [exp_id])

        with open(index_path, "w", encoding="utf-8") as fh:
            fh.write("{ this is not json")
        self.assertEqual([e["id"] for e in store.load_index(self.root)],
                         [exp_id])

    def test_reads_bom_encoded_json(self):
        """外部工具改过的实验 JSON 带 BOM 时仍应可读。"""
        exp_id = _create(self.root, _spin_raw())
        exp_dir = store.experiment_dir(self.root, exp_id)
        for name in ("meta.json", "validation.json", "request.json"):
            path = os.path.join(exp_dir, name)
            with open(path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            with open(path, "w", encoding="utf-8-sig") as fh:
                json.dump(payload, fh, ensure_ascii=False)
        index_path = os.path.join(self.root, "index.json")
        with open(index_path, "r", encoding="utf-8") as fh:
            index = json.load(fh)
        with open(index_path, "w", encoding="utf-8-sig") as fh:
            json.dump(index, fh, ensure_ascii=False)

        exp = store.get(self.root, exp_id)
        self.assertEqual(exp["meta"]["id"], exp_id)
        self.assertEqual(exp["validation"]["overall"], "pass")
        self.assertEqual([e["id"] for e in store.load_index(self.root)],
                         [exp_id])

    def test_compare_entries(self):
        exp_a = _create(self.root, _spin_raw(2.0, 1.0))
        exp_b = _create(self.root, _spin_raw(3.0, 0.0))
        data = store.compare_entries(self.root, [exp_a, exp_b])
        self.assertEqual(data["ids"], [exp_a, exp_b])
        self.assertEqual(len(data["rows"]), 2)
        param_diff = data["diff"]["params"]
        self.assertTrue(param_diff["Omega_x"]["differs"])
        self.assertEqual(param_diff["Omega_x"]["values"][exp_a], 2.0)
        self.assertTrue(data["diff"]["scalars"]["Omega_eff"]["differs"])
        self.assertFalse(data["diff"]["any_failed"])
        self.assertEqual(data["diff"]["overall"][exp_a], "pass")

    def test_compare_marks_failed_experiment(self):
        ok_id = _create(self.root, _spin_raw())
        bad_id = _create(self.root, _spin_raw(2.0, 1.0, "please fail"))
        data = store.compare_entries(self.root, [ok_id, bad_id])
        self.assertTrue(data["diff"]["any_failed"])
        self.assertEqual(data["diff"]["failed"][bad_id], ["forced_fail"])
        self.assertEqual(data["diff"]["overall"][bad_id], "partial")


# ------------------------------------------------------------------ CLI 测试
class TestCli(FakeDepsCase):
    def setUp(self):
        self.root = _make_temp_dir("qllab-cli-test-")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.root, ignore_errors=True)

    def _run(self, raw, extra=None):
        argv = ["run", "--spec", json.dumps(raw, ensure_ascii=False),
                "--json", "--out-root", self.root]
        return _run_cli(argv + list(extra or []))

    def test_run_json_summary_and_files(self):
        code, out, err = self._run(_spin_raw())
        self.assertEqual(code, 0, err)
        payload = json.loads(out)
        self.assertTrue(payload["ok"])
        self.assertRegex(payload["id"], _ID_RE)
        self.assertEqual(payload["overall"], "pass")
        self.assertEqual(payload["failed"], [])
        self.assertTrue(os.path.isdir(payload["dir"]))
        self.assertTrue(os.path.isfile(payload["report"]))
        self.assertIn("max_flip_prob_theory", payload["scalars"])
        self.assertEqual(payload["figures"], ["population"])
        self.assertIn("checks", payload)
        self.assertIn("confirm", payload)
        self.assertEqual(store.load_index(self.root)[0]["id"], payload["id"])

    def test_run_human_output_contains_confirm_and_summary(self):
        code, out, err = _run_cli(
            ["run", "--spec", json.dumps(_spin_raw(), ensure_ascii=False),
             "--out-root", self.root])
        self.assertEqual(code, 0, err)
        self.assertIn("【模型与参数确认】", out)
        self.assertIn("【实验已记录】", out)
        self.assertIn("实验 ID：", out)
        self.assertIn("总体验证：pass", out)

    def test_run_request_file(self):
        path = os.path.join(self.root, "req.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(_spin_raw(), fh, ensure_ascii=False)
        code, out, err = _run_cli(
            ["run", "--request", path, "--json", "--root", self.root])
        self.assertEqual(code, 0, err)
        self.assertTrue(json.loads(out)["ok"])

    def test_request_file_with_utf8_bom(self):
        """Windows 的 utf8 编码常带 BOM，--request 必须能解析。"""
        path = os.path.join(self.root, "bom-req.json")
        with open(path, "w", encoding="utf-8-sig") as fh:
            json.dump(_spin_raw(), fh, ensure_ascii=False)
        code, out, err = _run_cli(
            ["run", "--request", path, "--json", "--root", self.root])
        self.assertEqual(code, 0, err)
        self.assertTrue(json.loads(out)["ok"])

    def test_run_invalid_request_exit_2(self):
        code, out, err = _run_cli(
            ["run", "--spec", '{"model": "not_a_model"}', "--json",
             "--out-root", self.root])
        self.assertEqual(code, 2)
        payload = json.loads(out)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["code"], 2)
        self.assertIn("请求非法", err)

    def test_run_bad_json_exit_2(self):
        code, out, err = _run_cli(
            ["run", "--spec", "{oops", "--json", "--out-root", self.root])
        self.assertEqual(code, 2)
        self.assertIn("JSON", err)

    def test_run_validation_fail_exit_1(self):
        code, out, err = self._run(_spin_raw(question="please fail"))
        self.assertEqual(code, 1)
        payload = json.loads(out)
        self.assertEqual(payload["overall"], "partial")
        self.assertEqual(payload["failed"], ["forced_fail"])
        self.assertEqual(payload["counts"]["fail"], 1)

    def test_missing_subcommand_exit_2(self):
        code, _out, _err = _run_cli([])
        self.assertEqual(code, 2)
        code, _out, _err = _run_cli(["run"])
        self.assertEqual(code, 2)

    def test_sweep_injects_sweep_block(self):
        code, out, err = _run_cli(
            ["sweep", "--spec", json.dumps(_spin_raw(), ensure_ascii=False),
             "--param", "Omega_z", "--values", "0,0.5,1",
             "--json", "--out-root", self.root])
        self.assertEqual(code, 0, err)
        payload = json.loads(out)
        self.assertIn("参数扫描", payload["confirm"])
        self.assertIn("3", payload["confirm"])

    def test_list_show_json(self):
        first, _, _ = self._run(_spin_raw(2.0, 1.0, "实验一"))
        self.assertEqual(first, 0)
        _, out, _ = self._run(_ho_raw("实验二"))
        code, out, err = _run_cli(["list", "--json", "--out-root", self.root])
        self.assertEqual(code, 0, err)
        payload = json.loads(out)
        self.assertEqual(payload["count"], 2)
        self.assertEqual(len(payload["experiments"]), 2)

        exp_id = payload["experiments"][0]["id"]
        code, out, err = _run_cli(["show", exp_id, "--json",
                                   "--out-root", self.root])
        self.assertEqual(code, 0, err)
        shown = json.loads(out)
        self.assertEqual(shown["meta"]["id"], exp_id)
        self.assertIn("checks", shown["validation"])
        self.assertIn("request", shown)

        code, out, err = _run_cli(["show", "ql-20260101-000000-dead",
                                   "--json", "--out-root", self.root])
        self.assertEqual(code, 2)

    def test_derive_changes_whitelisted_field(self):
        code, out, err = self._run(_spin_raw(2.0, 1.0))
        self.assertEqual(code, 0, err)
        parent_id = json.loads(out)["id"]

        code, out, err = _run_cli(
            ["derive", parent_id, "--set", "params.Omega_z=0",
             "--set", "time.n_steps=801", "--json", "--out-root", self.root])
        self.assertEqual(code, 0, err)
        child = json.loads(out)
        self.assertRegex(child["id"], _ID_RE)
        self.assertNotEqual(child["id"], parent_id)
        self.assertEqual(child["relation"], "derived")
        self.assertEqual(child["parent_id"], parent_id)
        self.assertEqual(child["params"]["Omega_z"], 0.0)

        child_exp = store.get(self.root, child["id"])
        self.assertEqual(child_exp["request"]["params"]["Omega_z"], 0.0)
        self.assertEqual(child_exp["request"]["time"]["n_steps"], 801)
        # 父实验保持不变，且仍可查到
        parent_exp = store.get(self.root, parent_id)
        self.assertEqual(parent_exp["request"]["params"]["Omega_z"], 1.0)
        self.assertEqual(len(store.load_index(self.root)), 2)

    def test_derive_rejects_invalid_set(self):
        code, out, _ = self._run(_spin_raw())
        parent_id = json.loads(out)["id"]
        for bad in ("model=harmonic_oscillator", "params.Omega_x=abc",
                    "params.unknown=1", "question"):
            code, out, err = _run_cli(
                ["derive", parent_id, "--set", bad, "--json",
                 "--out-root", self.root])
            self.assertEqual(code, 2, f"{bad} 应返回 2")
            self.assertFalse(json.loads(out)["ok"])

    def test_derive_requires_a_set(self):
        code, out, _ = self._run(_spin_raw())
        parent_id = json.loads(out)["id"]
        code, out, err = _run_cli(["derive", parent_id, "--json",
                                   "--out-root", self.root])
        self.assertEqual(code, 2)
        self.assertIn("至少需要一个 --set", err)

    def test_rerun_relation(self):
        code, out, _ = self._run(_spin_raw())
        parent_id = json.loads(out)["id"]
        code, out, err = _run_cli(["rerun", parent_id, "--json",
                                   "--out-root", self.root])
        self.assertEqual(code, 0, err)
        rerun = json.loads(out)
        self.assertEqual(rerun["relation"], "rerun")
        self.assertEqual(rerun["parent_id"], parent_id)
        self.assertNotEqual(rerun["id"], parent_id)

    def test_compare_json(self):
        _, out, _ = self._run(_spin_raw(2.0, 1.0))
        id_a = json.loads(out)["id"]
        _, out, _ = self._run(_spin_raw(3.0, 0.0))
        id_b = json.loads(out)["id"]
        code, out, err = _run_cli(["compare", id_a, id_b, "--json",
                                   "--out-root", self.root])
        self.assertEqual(code, 0, err)
        data = json.loads(out)
        self.assertTrue(data["ok"])
        self.assertEqual(data["ids"], [id_a, id_b])
        self.assertTrue(data["diff"]["params"]["Omega_x"]["differs"])

        code, _out, _err = _run_cli(["compare", id_a,
                                     "ql-20260101-000000-dead", "--json",
                                     "--out-root", self.root])
        self.assertEqual(code, 2)

    def test_export_rewrites_artifacts(self):
        code, out, _ = self._run(_spin_raw())
        exp_id = json.loads(out)["id"]
        exp_dir = store.experiment_dir(self.root, exp_id)
        report_html = os.path.join(exp_dir, "report.html")
        os.remove(report_html)

        code, out, err = _run_cli(
            ["export", exp_id, "--format", "all", "--json",
             "--out-root", self.root])
        self.assertEqual(code, 0, err)
        payload = json.loads(out)
        for fmt in ("html", "json", "csv"):
            self.assertIn(fmt, payload["paths"])
            self.assertTrue(os.path.isfile(payload["paths"][fmt]))
        self.assertTrue(os.path.isfile(report_html))

        code, _out, _err = _run_cli(["export", "ql-20260101-000000-dead",
                                     "--json", "--out-root", self.root])
        self.assertEqual(code, 2)

    def test_selfcheck_passes_with_reference_values(self):
        code, out, err = _run_cli(["selfcheck", "--json"])
        self.assertEqual(code, 0, err)
        payload = json.loads(out)
        self.assertTrue(payload["ok"])
        self.assertGreaterEqual(payload["total"], 9)
        self.assertEqual(payload["passed"], payload["total"])

        code, out, err = _run_cli(["selfcheck"])
        self.assertEqual(code, 0, err)
        self.assertIn("全部通过", out)
        self.assertIn("E0_exact", out)

    def test_selfcheck_reports_failure(self):
        bad_models = _make_module(
            "qllab.models",
            run_model=lambda req: (_ for _ in ()).throw(RuntimeError("boom")))
        old = sys.modules["qllab.models"]
        sys.modules["qllab.models"] = bad_models
        try:
            code, out, err = _run_cli(["selfcheck", "--json"])
        finally:
            sys.modules["qllab.models"] = old
        self.assertEqual(code, 1)
        payload = json.loads(out)
        self.assertFalse(payload["ok"])
        self.assertLess(payload["passed"], payload["total"])


if __name__ == "__main__":
    unittest.main()
