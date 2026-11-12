# -*- coding: utf-8 -*-
"""QuantumLab CLI 入口。

用法（详细见 SKILL.md）：
    python quantum_lab.py run    --request req.json
    python quantum_lab.py run    --request req.json --out-root E:\\lab
    python quantum_lab.py list   [--json]
    python quantum_lab.py show   <exp_id> [--json]
    python quantum_lab.py rerun  <exp_id>
    python quantum_lab.py derive <exp_id> --set params.Omega_z=0
    python quantum_lab.py compare <id1> <id2> [...] [--json]
    python quantum_lab.py export <exp_id> [--format html|json|csv]
    python quantum_lab.py selfcheck
"""
import os
import sys

# Windows 控制台默认 GBK，无法编码 ħ/⟨⟩ 等物理符号；统一按 UTF-8 输出。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # pragma: no cover - 非标准流对象
        pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qllab import cli  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(cli.main(sys.argv[1:]))
