# -*- coding: utf-8 -*-
"""跑一遍 examples/requests 里的示例，并演示实验历史操作。

用法（仓库根目录，任何平台）：
    python examples/run_examples.py                 # 产物写到 examples/experiments/
    python examples/run_examples.py --out-dir /tmp/lab
    python examples/run_examples.py --quiet

依次执行：run（5 组请求）→ list → rerun → derive --set params.Omega_z=0 → compare。
任一命令非零退出即返回非零，便于 CI 当冒烟测试用。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CLI = os.path.join(ROOT, "scripts", "quantum_lab.py")
REQUESTS = os.path.join(HERE, "requests")

CASES = [
    "spin_ox2_oz1.json",
    "spin_sweep_oz.json",
    "ho_superposition.json",
    "ho_coherent_truncation.json",
    "perturbation_default.json",
]


def run_cli(args, quiet=False):
    """执行 CLI 并返回 (exit_code, stdout)。子进程只跑本仓库脚本，不接受用户输入拼接。"""
    cmd = [sys.executable, CLI] + args
    print("$", " ".join(os.path.basename(c) if i == 1 else c for i, c in enumerate(cmd)))
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                          errors="replace")
    out = (proc.stdout or "") + (proc.stderr or "")
    if not quiet:
        tail = "\n".join(out.strip().splitlines()[-12:])
        print(tail)
    return proc.returncode, out


def main(argv=None):
    parser = argparse.ArgumentParser(description="QuantumLab 示例冒烟")
    parser.add_argument("--out-dir", default=os.path.join(HERE, "experiments"),
                        help="实验输出根目录（默认 examples/experiments）")
    parser.add_argument("--quiet", action="store_true", help="不打印命令输出尾部")
    args = parser.parse_args(argv)

    out_root = os.path.abspath(args.out_dir)
    os.makedirs(out_root, exist_ok=True)
    print(f"输出根目录：{out_root}\n")

    created = []
    for name in CASES:
        req = os.path.join(REQUESTS, name)
        code, out = run_cli(["run", "--request", req, "--out-root", out_root], args.quiet)
        if code != 0:
            print(f"[FAIL] {name}: exit={code}")
            return 1
        exp_id = ""
        for line in out.splitlines():
            if "实验 ID：" in line:
                exp_id = line.split("实验 ID：", 1)[1].strip()
        created.append((name, exp_id))
        print(f"[ OK ] {name} -> {exp_id}\n")

    first_id = created[0][1]
    steps = [
        (["list", "--out-root", out_root, "--limit", "5"], "list"),
        (["rerun", first_id, "--out-root", out_root], "rerun"),
        (["derive", first_id, "--set", "params.Omega_z=0", "--out-root", out_root], "derive"),
        (["compare", first_id, created[-1][1], "--out-root", out_root], "compare"),
        (["export", created[-1][1], "--format", "html", "--out-root", out_root], "export"),
    ]
    for cmd, label in steps:
        code, _ = run_cli(cmd, args.quiet)
        if code != 0:
            print(f"[FAIL] {label}: exit={code}")
            return 1
        print(f"[ OK ] {label}\n")

    index = os.path.join(out_root, "index.json")
    if os.path.isfile(index):
        with open(index, encoding="utf-8-sig") as fh:
            data = json.load(fh)
        summary = [(e.get("id"), e.get("model"), e.get("overall"), e.get("relation"))
                   for e in data.get("experiments", [])]
        print("实验索引（新→旧）：")
        for row in summary:
            print("  ", row)
    print(f"\n全部示例通过。打开报告：{os.path.join(out_root, created[0][1], 'report.html')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
