# -*- coding: utf-8 -*-
"""打印最近一次集成验证结果的汇总（供 CI 使用）。

用法：python tests/summarize_results.py [结果文件路径]
不带参数时取 tests/_artifacts/results-*.json 中最新的一份。
"""
from __future__ import annotations

import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "_artifacts")


def latest_result():
    files = sorted(glob.glob(os.path.join(ART, "results-*.json")))
    return files[-1] if files else None


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    path = argv[0] if argv else latest_result()
    if not path or not os.path.isfile(path):
        print("未找到验证结果文件（tests/_artifacts/results-*.json）")
        return 0
    with open(path, encoding="utf-8-sig") as fh:
        data = json.load(fh)
    summary = data.get("summary") or {}
    print(f"结果文件：{os.path.relpath(path, os.path.dirname(HERE))}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    records = data.get("records") or []
    failed = [r for r in records if r.get("status") == "FAIL"]
    if failed:
        print(f"\n失败项（{len(failed)}）：")
        for rec in failed:
            print(f"  - {rec.get('area')}:{rec.get('name')} :: {rec.get('evidence')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
