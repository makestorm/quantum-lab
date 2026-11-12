# QuantumLab · AI 辅助量子力学计算与验证

把自然语言描述的量子力学问题，变成**受约束的结构化请求** → 本地数值求解 → **物理一致性与数值误差检查** → 图像 / 数据 / 独立 HTML 报告。

QuantumLab 既是一个 **Agent Skill**（放进 `.dsh/skills/` 或同类技能目录，让 AI 助手按固定流程做量子实验），
也是一个**独立命令行工具**（只用 `numpy` + `Pillow`，不需要 scipy / matplotlib）。

| | |
|---|---|
| 语言 / 依赖 | Python ≥ 3.10，`numpy`，`Pillow`（标准库优先） |
| 模型 | 自旋 1/2 二能级、量子谐振子（截断 Fock）、非简并微扰 |
| 输出 | `png`（3000×1760 图表）、`csv`、`json`、单文件 `html` 报告（图内嵌，可打印 PDF） |
| 检查 | 14 项：输入合法性、有限性、厄米性、本征残差、正交归一、态归一化、概率守恒、能量实数/守恒、解析对比、能谱对比、截断诊断、求解器参照、微扰适用范围 |
| 测试 | 73 项单元测试 + 201 项独立交叉/对抗/端到端检查（见 [docs/VERIFICATION.md](docs/VERIFICATION.md)） |

![自旋布居数随时间的演化](examples/sample-experiment/ql-20261112-121251-fe8b/figures/population.png)

> 上图：`Ωx=2, Ωz=1, |↑⟩` 的拉比振荡。跃迁概率峰值 **0.8 = Ω⊥²/|Ω|²**，数值曲线与 Bloch 解析解重合
> （最大偏差 5.6e-16）。完整报告：[report.html](examples/sample-experiment/ql-20261112-121251-fe8b/report.html)

## 特性

- **自然语言 → 结构化请求**：白名单模型与参数、默认值全部标注来源，未知参数/单位/规模直接拒绝，
  不接受任意代码或表达式（无 `eval`/`exec`/`subprocess`）。
- **先确认再计算**：执行前打印哈密顿量、参数（哪些是默认值）、初态、单位、时间范围、求解方法与近似条件。
- **验证不走过场**：每项检查都给出指标值 + 容差 + 通过状态；不适用项标 `skipped` 并说明原因；
  区分「执行失败」「完成但部分验证未通过」「所执行检查通过」三态。
- **近似与误差分开报告**：微扰偏离精确解是物理预期（适用范围问题），不会写成"求解器算错了"；
  谐振子截断诊断也不宣称"已证明收敛"。
- **可复现**：每次计算生成实验编号与目录（request / result / validation / meta / 数据 / 图 / 报告），
  支持 `rerun`、改参数 `derive`、跨实验 `compare`，父实验永不覆盖。
- **零重依赖绘图**：Pillow 手写图表（自动刻度、对数轴、图例、注脚、中文字形兜底），
  在只有 numpy+Pillow 的环境里也能出图。

## 快速开始

### 1) 作为 Agent Skill 使用

```bash
git clone https://github.com/ZhouFeng/quantum-lab.git
# DSH：克隆到技能目录（或建立符号链接）
cp -r quantum-lab ~/.dsh/skills/quantum-lab          # Linux / macOS
xcopy /E /I quantum-lab %USERPROFILE%\.dsh\skills\quantum-lab   # Windows
```

技能入口是仓库根目录的 [SKILL.md](SKILL.md)，它会引导助手走完
「理解 → 澄清 → 确认 → 计算 → 验证 → 解释 → 保存」流程。

### 2) 作为命令行工具使用

```bash
pip install -r requirements.txt
python scripts/quantum_lab.py selfcheck          # 3 条物理基准，期望 10/10 PASS
```

```bash
# bash / zsh
python scripts/quantum_lab.py run --request examples/requests/spin_ox2_oz1.json --out-root ./my-lab

# PowerShell（会把内联参数里的双引号吃掉，所以用文件或管道）
python scripts/quantum_lab.py run --request examples\requests\spin_ox2_oz1.json --out-root .\my-lab
'{"model":"spin_half","params":{"Omega_x":2,"Omega_z":1}}' | python scripts/quantum_lab.py run --stdin
```

请求就是一段 JSON（字段细节见 [references/models.md](references/models.md)）：

```json
{
  "model": "spin_half",
  "params": {"Omega_x": 2.0, "Omega_z": 1.0},
  "initial_state": {"kind": "up"},
  "question": "计算 Ωx=2、Ωz=1 的自旋二能级系统，画出演化曲线，并解释为什么跃迁概率达不到 1。"
}
```

运行后会打印确认块与摘要，并落盘一个实验目录：

```
【实验已记录】
实验 ID：ql-20261112-121251-fe8b
总体验证：pass（通过） 检查计数：pass 10 / fail 0 / skipped 4
关键数值：Omega_eff=2.23606798、max_flip_prob_theory=0.8、pop_1_peak_numeric=0.8
图表（4 张）：analytic_compare, bloch_expectation, energy, population
报告：my-lab/ql-20261112-121251-fe8b/report.html
```

### 3) 常用命令

| 命令 | 用途 |
|---|---|
| `run --request req.json` | 计算、验证、绘图、落盘（`--json` 输出机器可读摘要） |
| `sweep --spec ... --param Omega_z --values 0,0.5,1` | 参数扫描叠加图 |
| `list [--model] [--text] --json` | 实验历史（倒序） |
| `show <id> --json` | 单个实验详情 |
| `rerun <id>` | 同参数重跑（新 ID，`relation=rerun`） |
| `derive <id> --set params.Omega_z=0` | 改参数另存（保留父实验，`relation=derived`） |
| `compare <id1> <id2> ...` | 参数/scalars 差异对比表 |
| `export <id> --format html\|json\|csv\|all` | 重新导出报告与数据 |
| `selfcheck` | 三条物理基准自检（|Ω|=√5、E_n=n+½、微扰闭式解） |

退出码：`0` 完成且所执行检查全部通过；`1` 执行失败**或** `overall=partial`；`2` 请求非法（单行消息，无 traceback）。

## 支持的模型

| 模型 | 哈密顿量（ħ=1） | 参数 | 主要输出 | 解析参照 |
|---|---|---|---|---|
| `spin_half` | `H = ½(Ωxσx + Ωyσy + Ωzσz)` | `Omega_x/y/z` | 本征值、`pop_0/1`、`⟨σx,y,z⟩`、能量 | Bloch 进动；`P↓ = (Ω⊥²/|Ω|²)sin²(|Ω|t/2)` |
| `harmonic_oscillator` | `H = ω(a†a + ½)`，截断 N 能级 | `omega`、`n_levels` | 能谱、能级占据、`⟨x⟩`、`⟨p⟩`、边界占据 | `E_n=ω(n+½)`；`⟨x⟩=√2 Re[A₀e^{-iωt}]` |
| `perturbation` | `H = (Δ/2)σz + (λ/2)σx` | `delta`、`lambda` | `E0` 精确/一阶/二阶、绝对与相对误差、λ 适用范围 | `E₀=-½√(Δ²+λ²)`；二阶 `-Δ/2-λ²/(4Δ)` |

初态支持 `up/down/plus_x/plus_y/custom`（自旋）与 `fock/coherent/superposition`（谐振子）；
时间范围与采样点可自定义，也可 `sweep` 扫描（≤60 点）。

## 验证与测试

```bash
# 单元测试（模型 / 图表 / 实验库）
python -m unittest discover -s tests -t tests -p "test_*.py"     # 73 tests, OK

# 独立交叉验证（Taylor 级数、RK4、闭式解；不依赖产品代码的谱分解）
python tests/test_integration.py                                  # 201 PASS / 0 FAIL
python tests/reference_physics.py                                 # 参照实现自检
```

独立验证报告（含参考实现、对抗性输入、端到端复现证据与已知风险）：
**[docs/VERIFICATION.md](docs/VERIFICATION.md)**

要点：产品数值与独立参照最大偏差 5.6e-16 ~ 1.2e-10；36+16+7 个非法/注入请求全部被拒；
`rerun` 与首次逐值一致（max diff 0.0）且父实验字节不变；PNG 全为 3000×1760；
报告 HTML 无外部引用、图 base64 内嵌；CSV 为 `utf-8-sig`。

## 目录结构

```
quantum-lab/
├── SKILL.md                 # 技能主文档：流程、模型速查、退出码、安全边界
├── scripts/
│   ├── quantum_lab.py       # CLI 入口
│   └── qllab/               # spec / models / validate / charts / store / report / cli / util
├── references/              # 模型公式与解析参照、验证规范、报告与 CSV 格式
├── docs/
│   ├── CONTRACT.md          # 模块接口契约（开发者）
│   └── VERIFICATION.md      # 独立验证报告
├── tests/
│   ├── test_models.py / test_charts_report.py / test_store_cli.py   # 73 项单元测试
│   ├── test_integration.py  # 201 项交叉/对抗/端到端套件
│   ├── reference_physics.py # 独立参照实现（Taylor / RK4 / 闭式解）
│   └── summarize_results.py # 打印最近一次验证汇总（CI 使用）
├── examples/
│   ├── requests/*.json      # 5 个可直接运行的示例请求
│   ├── run_examples.py      # 一键跑完示例 + 历史操作
│   └── sample-experiment/   # 一次完整实验的全部产物（含 report.html）
├── .github/workflows/tests.yml   # CI：单元测试、自检、示例冒烟、交叉验证套件
├── CHANGELOG.md / CONTRIBUTING.md / LICENSE / requirements.txt
└── README.md
```


## 设计原则

1. **白名单与上限**：模型/参数/单位/扫描点数全部受限，超限拒绝而不是静默降采样。
2. **不执行任意代码**：没有 `eval`/`exec`/`subprocess`，用户字符串只作为数据。
3. **不夸大结论**：`partial` 不会被说成"结果正确"；截断小不等于收敛；微扰偏离精确解属适用范围问题。
4. **可追溯**：请求、参数来源（给定/默认）、容差、求解器、耗时、父子关系全部入库。
5. **解释三段式**：发生了什么（数值）→ 为什么（机制）→ 限制（误差、近似、未验证部分）。

## 已知边界与路线图

- 已支持：三类模型、自定义初态与时间范围、参数扫描、实验历史（重跑/派生/对比）、JSON/CSV/PNG/HTML 导出。
- 未实现：ħ≠1 的单位制、显含时哈密顿量、简并/近简并微扰、开放量子系统、多体纠缠与大规模并行扫描
  （遇到时明确说明缺少的能力，不会替换成默认演示参数）。
- 规划中：文献证据整合（区分"文献结论 / 本地计算 / 综合推断"）、可选外部模型服务适配层
  （OpenAI / DeepSeek / GLM / Kimi / Claude；密钥不写入实验与日志）、PDF 排版导出。

## English overview

QuantumLab is an agent skill **and** a standalone CLI that turns natural-language quantum
problems into constrained, whitelisted requests, solves them locally with `numpy` (Hermitian
spectral decomposition, ħ=1), and then reports **physical-consistency and numerical checks**
with explicit metrics, tolerances and applicability limits before any explanation is written.

- Models: spin-1/2 two-level system, truncated quantum harmonic oscillator, non-degenerate perturbation theory.
- Outputs: 3000×1760 PNG figures, CSV/JSON data, and a single-file print-friendly HTML report (images embedded).
- Reproducibility: every run gets an experiment id and directory; `rerun`, `derive`, `compare` keep the parent intact.
- Safety: no `eval`/`exec`/`subprocess`, strict parameter/unit/scale validation, no silent substitution of unsupported conditions.
- Verification: 73 unit tests plus a 201-check independent red-team suite (Taylor/RK4/closed-form cross-checks) in [docs/VERIFICATION.md](docs/VERIFICATION.md).

```bash
pip install -r requirements.txt
python scripts/quantum_lab.py run --request examples/requests/spin_ox2_oz1.json --out-root ./my-lab
python scripts/quantum_lab.py selfcheck
```

## 许可证

MIT，见 [LICENSE](LICENSE)。欢迎提交 issue / PR，贡献指南见 [CONTRIBUTING.md](CONTRIBUTING.md)。

维护者：ZhouFeng（[@ZhouFeng](https://github.com/ZhouFeng)）
