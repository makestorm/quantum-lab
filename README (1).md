# 示例

`requests/` 里是 5 个可直接运行的请求，覆盖三类模型与参数扫描：

| 文件 | 内容 | 关键结论 |
|---|---|---|
| `spin_ox2_oz1.json` | 自旋 `Ωx=2, Ωz=1, |↑⟩` | 跃迁概率峰值 `Ω⊥²/|Ω|² = 0.8`，|Ω|=√5 |
| `spin_sweep_oz.json` | 扫描 `Ωz = 0, 0.5, 1, 2` | Ωz=0 时峰值 1，Ωz 越大峰值越小 |
| `ho_superposition.json` | 谐振子 `(|0⟩+|1⟩)/√2` | `⟨x⟩(t)=cos(ωt)/√2`，相对相位驱动振荡 |
| `ho_coherent_truncation.json` | 相干态 `α=1`，扫描 `n_levels = 8,16,32,64` | N=8 时截断诊断报警（边界占据 7.3e-5） |
| `perturbation_default.json` | `Δ=1, λ=0.1` 默认扫描 | `E₀=-½√(Δ²+λ²)`，相对误差 1.24e-5，适用到 λ≈0.5 |

## 一键跑完

```bash
python examples/run_examples.py                 # 产物写到 examples/experiments/
python examples/run_examples.py --out-dir /tmp/lab
python examples/run_examples.py --keep          # 跑完不打印摘要细节
```

脚本会依次执行 `run`（5 组）→ `list` → `rerun` → `derive --set params.Omega_z=0` → `compare`，
任一命令非零退出即整体失败。

## 目录里已有的成品

`sample-experiment/ql-20261112-121251-fe8b/` 是 `spin_ox2_oz1.json` 的完整产物，
可以直接打开 [report.html](sample-experiment/ql-20261112-121251-fe8b/report.html) 看报告长什么样：

```
sample-experiment/ql-20261112-121251-fe8b/
├── report.html        # 9 章独立报告，图 base64 内嵌，可打印为 PDF
├── request.json       # 规范化后的请求（含参数来源 provenance 与提示）
├── result.json        # 本征值、时间序列、scalars、metrics、figure_specs
├── validation.json    # 14 项检查：指标、容差、状态、适用边界
├── meta.json          # 实验编号、时间、模型、参数、关系、耗时
├── data.csv           # t, pop_0, pop_1, expect_x/y/z, energy, pop_1_theory
└── figures/*.png      # 4 张 3000×1760 图（布居数、Bloch 期望值、能量守恒、解析对比）
```
