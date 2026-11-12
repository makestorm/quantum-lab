# 实验目录、文件与报告格式

## 目录结构

默认根目录：当前工作目录下 `quantumlab-experiments/`（可用 `--out-root` 改）。

```
quantumlab-experiments/
├─ index.json                        # 全部实验索引（倒序）
└─ ql-20261112-110359-a1b2/          # 实验编号 = ql-<时间戳>-<4位随机>
   ├─ request.json                  # 规范化后的 canonical request（含 provenance/warnings）
   ├─ result.json                   # 结果：本征值、时间序列、scalars、metrics、figure_specs
   ├─ validation.json               # 检查表 + overall + limits
   ├─ meta.json                     # id/时间/模型/参数/父子关系/耗时/求解器/原始问题
   ├─ data.csv                      # 时间序列或参数扫描数据（utf-8-sig）
   ├─ figures/*.png                 # 每张图一个文件（3000×1760）
   └─ report.html                   # 独立报告（图 base64 内嵌）
```

`index.json`：

```json
{"version": 1, "experiments": [
  {"id": "ql-20261112-110359-a1b2", "created_at": "2026-11-12T11:03:59+08:00",
   "model": "spin_half", "params": {"Omega_x": 2.0, "Omega_y": 0.0, "Omega_z": 1.0},
   "question": "...", "overall": "pass", "parent_id": null, "relation": null,
   "dir": "ql-20261112-110359-a1b2", "figures": ["population", "analytic_compare", "error"]}
]}
```

`relation`：`null`（首次）| `rerun`（同参数重跑）| `derived`（改参数派生）| `sweep_child`。
重跑/派生一定生成新 ID，绝不覆盖父实验。

## data.csv 列

| 模型 | 列 |
|---|---|
| spin_half | `t, pop_0, pop_1, expect_x, expect_y, expect_z, energy, pop_1_theory` |
| harmonic_oscillator | `t, pop_0, pop_1, pop_2, pop_3, expect_x, expect_p, expect_n, boundary_pop, expect_x_theory` |
| perturbation | `lambda, E0_pt1, E0_pt2, E0_exact, abs_err, rel_err, within_threshold` |

有 `sweep` 时，另附扫描表（`sweep.csv` 或与时间序列合并在同一文件，但 report.html 必须给出同一数据的表格）：

| 模型 | 扫描表的列 |
|---|---|
| spin_half / harmonic_oscillator | `<扫描参数>` + 该组 `scalars` 中标量列（按模型插入顺序；列表值如 `spectrum` 不进 CSV） |
| perturbation | `lambda, E0_pt1, E0_pt2, E0_exact, abs_err, rel_err, within_threshold`（其余标量可置后） |

扫描表的列由 `report` 侧统一展开 `series[].scalars` 生成（`observables`、与参数列重复的键、列表/空值一律丢弃），
CSV 与 report.html 中的表格必须同源。

## result.json 关键字段

| 字段 | 说明 |
|---|---|
| `hamiltonian` | 复矩阵（`[re, im]` 表示复数，见 util.to_jsonable） |
| `eigen.values/vectors` | 本征值与本征矢量 |
| `times`, `states` | 时间网格与复振幅 (n_steps×dim) |
| `observables` | 随时间的量，见上表列名 |
| `scalars` | 标量结论：`Omega_eff`、`E_plus/minus`、`max_flip_prob_theory`、`spectrum`、`E0_exact`、`lambda_valid_max` 等 |
| `analytic` | 解析解名称、公式、数值 |
| `sweep` | 扫描参数、每点结果与诊断 |
| `metrics` | 误差指标，供验证表引用 |
| `figure_specs` | 声明式图描述（供 charts 渲染，也便于复现） |
| `warnings` | 数值层面的提示（截断、相位、归一化等） |
| `summary.bullets` | 结果概述（"发生了什么"的原始素材，非最终解释） |

## report.html 章节

1. **标题区**：实验编号、模型、生成时间、总状态色块（执行失败 / 部分验证未通过 / 通过）。
2. **研究问题与实验目的**：`question` 原文 + 一句话目的。
3. **模型公式与假设**：哈密顿量、基矢、ħ=1 约定、截断/微扰假设。
4. **参数、初态与单位**：参数表（标出"默认值"来源）、初态、时间范围。
5. **数值方法**：谱分解 `U(t)=V e^{-iΛt} V†`、采样点、容差。
6. **结果图像与关键数据**：每张图（图注 + 参数说明）+ 关键标量表。
7. **物理一致性验证**：检查表（状态色块、指标、容差、说明）+ overall + limits。
8. **结果解释与适用限制**：发生了什么 / 为什么 / 限制。
9. **复现配置**：可直接粘贴复用的请求 JSON、运行命令、文件清单。

要求：单文件、无外部依赖、`<img src="data:image/png;base64,...">` 内嵌、
打印友好（A4、避免图表被截断、状态色块打印后仍可辨）、中文排版。
报告只写本次真实数据，未做的实验不得出现。

## 终端摘要（`run` 的输出）

必须包含：实验 ID、输出目录、overall、失败检查 id 与误差/容差、关键标量（如最大跃迁概率、
E0、rel_err）、图文件名、报告路径。`--json` 时输出机器可读摘要（实验 ID、目录、
overall、checks、scalars、figures、report）。
