# QuantumLab 独立验证报告（红队 / Red Team）

- 验证者：teammate `verifier`（独立于 physics-core / viz-report / store-cli）
- 日期：2026-11-12（Asia/Shanghai）
- 被验证代码：本仓库（`scripts/qllab/*.py`、`SKILL.md`、`references/*.md`、`docs/CONTRACT.md`）
- 运行环境：Python 3.12.14 + numpy 2.3.5 + Pillow 12.3.0（**无 scipy / matplotlib / pandas / pytest**，验证脚本只用标准库 + numpy + Pillow）
- 独立参照实现：[tests/reference_physics.py](tests/reference_physics.py)
- 交叉验证与对抗测试：[tests/test_integration.py](tests/test_integration.py)
- 原始机器可读结果：`tests/_artifacts/results-20261112-121802.json`（Lead 验收时复跑的最终全量，201 项；与 verifier 定稿时的 201 PASS 一致）。中间过程记录已在交付前清理，仅保留该最终结果。

## 0. 结论摘要

**全量 201 项检查 PASS / 0 FAIL / 0 BLOCKED**（覆盖 10 个领域：ref 6、static 4、spec 46、physics 42、validation 21、cli 24、artifacts 21、models_artifacts 16、e2e 17、regressions 4），其中物理数值与独立参照实现的最大偏差：

| 类别 | 最大偏差 | 判定阈值（任务书） | 结果 |
|---|---|---|---|
| 闭式解析对比（自旋 Bloch、谐振子 ⟨x⟩、微扰能级） | 1.2e-10（参照自检）；产品代码 5.6e-16 ~ 1.4e-15 | 1e-8 | PASS |
| Taylor 级数 / RK4 对拍（自旋、谐振子） | 6.7e-11 | 1e-6 | PASS |
| 产品态矢量 vs 独立 `expm`（不依赖谱分解） | 6.5e-14 | 1e-6 | PASS |
| 能谱 vs `ω(n+1/2)` | 7.1e-15 | 1e-8 | PASS |

对抗性输入（`spec` 节 36 个非法用例 + 1 个汇总判定 + 5 个正例 + 4 个规范化不变量；CLI 16 个非法请求 + 7 个 `--set` 注入）**全部被拒绝**：`SpecError` 或退出码 2，消息可读，无 traceback，无任意代码执行（用 canary 文件证明）。

验证期间共发现 **7 个实现缺陷（全部已修复并复测通过）+ 1 个需求文档笔误（已澄清更正）+ 2 个观察项（见第 9 节 O1/O2）**，详见第 9 节。**没有任何"意外通过"**（即不该被接受的输入被接受）——第 9 节的缺陷都是"该拒绝时崩了/该输出正确格式时输出错了"，不是"该拒绝却通过"。

> 结论限定：本报告只证明在所列参数、模型与检查项范围内数值一致、输入受限、交付物格式合规；不构成对未列出的物理情形（含时驱动、开放系统、简并微扰、多体）的验证。

## 1. 验证方法（独立性声明）

产品实现用厄米矩阵谱分解 `U(t)=V e^{-iΛt} V†`。独立参照实现刻意**不使用**该算法，也不 `import qllab.models`：

1. **Taylor 级数 + scaling & squaring**：`exp(A) ≈ Σ_{k=0}^{48} (A/2^s)^k/k!`，再平方 `s` 次（`||A/2^s||_∞ ≤ 0.5`），见 [tests/reference_physics.py:47](tests/reference_physics.py)。
2. **定步长 RK4**：直接积分 `i dψ/dt = Hψ`（每区间可细分），见 [tests/reference_physics.py:88](tests/reference_physics.py)。
3. **闭式解析解**：Bloch 进动 `n(t)=n∥Ω̂ + (n₀−n∥Ω̂)cos|Ω|t + (Ω̂×n₀)sin|Ω|t`；`P↓(t)=Ω⊥²/|Ω|²·sin²(|Ω|t/2)`；谐振子 `⟨x⟩(t)=√2 Re[A₀e^{-iωt}]`、`A₀=Σ c*ₙc_{n+1}√(n+1)`；微扰 `E₀=-½√(Δ²+λ²)`、`E₀⁽²⁾=-Δ/2-λ²/(4Δ)`。
4. **哈密顿量独立构造**：自旋用显式 Pauli 矩阵；谐振子用 `a[n-1,n]=√n` 独立搭 `a, a†`（不 import 产品代码）；微扰用 2×2 解析矩阵。

参照实现自身先做三方对拍（Taylor vs RK4 vs 闭式）：最差偏差 **1.225e-10**（RK4 的截断误差），确认参照实现可信后再用于检验产品代码。

## 2. 复现命令

```powershell
$PY = 'python'          # 任意 Python ≥3.10，已安装 numpy + Pillow
cd <仓库根>

# 参照实现自检（不需要产品代码）
& $PY tests\reference_physics.py

# 全量验证（约 1 分钟）
& $PY tests\test_integration.py

# 单节复跑
& $PY tests\test_integration.py --section physics,validation
& $PY tests\test_integration.py --section spec,cli          # 对抗输入
& $PY tests\test_integration.py --section artifacts,models_artifacts,e2e
& $PY tests\test_integration.py --section regressions       # η 回归用例
```

被验证修订（SHA-256 前 16 位，验证前后一致）：

| 文件 | hash | 最后修改 |
|---|---|---|
| spec.py | 7F209422383C339A | 11:44:14 |
| models.py | EDB6D2685A585D8F | 11:56:51 |
| validate.py | 77C058D7C8401625 | 11:22:30 |
| charts.py | 7284810D4C85BA3D | 11:33:35 |
| report.py | BEF6E177D3B47AF1 | 11:58:22 |
| store.py | 03D32225FF3466D7 | 11:34:51 |
| cli.py | 7B9AD44AEC34A7A2（定稿时）→ DE9DC4B34FB86757（Lead 收尾改动后，见第 10 节） | 12:17 |
| util.py | 8B1D772EFCAE8053 | 11:10:39 |

（`tests/_artifacts/results-20261112-121802.json` 记录最终全量 201 PASS 的逐项证据；除 cli.py 收尾改动外，
验证运行前后上述 hash 完全一致；cli.py 改动后已重跑全量并保持 201 PASS / 0 FAIL。）

## 3. 参照实现自检（`ref` 节，6 PASS）

| 指标 | 数值 |
|---|---|
| 自旋 Taylor vs Bloch（P↓） | 2.665e-15 |
| 自旋 Taylor vs Bloch（n_z） | 4.108e-15 |
| 自旋 RK4 vs Bloch（P↓ / n_z） | 6.074e-11 / 1.225e-10 |
| 自旋 Taylor vs RK4（态） | 8.910e-11 |
| 谐振子 Taylor vs 解析 ⟨x⟩ | 6.162e-14 |
| 谐振子 RK4 vs 解析 ⟨x⟩ | 5.107e-11 |
| 谐振子 Taylor vs RK4 | 6.686e-11 |
| 谐振子能谱 vs n+½ | 7.105e-15 |
| 相干态解析公式 vs √2cos t | 0.0 |
| `U(t)U(−t)−I` | 6.113e-18 |
| 态范数偏差（自旋 / 谐振子） | 4.441e-15 / 1.266e-13 |
| 微扰闭式解 vs 独立 `eigvalsh` | 0.0（特征向量残差 3.5e-18） |

## 4. 物理交叉校验（`physics` 节，42 PASS）

### 4.1 自旋 Ωx=2, Ωz=1, |↑⟩（周期 2π/√5 = 2.8099258924）

- H 矩阵与独立构造完全一致：`max|H−H_ref| = 0.0`；本征值 `±1.11803398875 = ±√5/2`，偏差 0.0。
- `Ω_eff = 2.23606797749979`，`max_flip_prob_theory = 0.7999999999999998`。
- 态矢量 vs 独立 Taylor `expm`：`max|Δψ| = 2.223e-15`。
- P↓ vs 解析 `(4/5)sin²(√5 t/2)`：`max = 5.551e-16`；峰值 `0.7999999999999996`，出现在 `t = T/2 = 1.4049629462`；`P↓(T) = 1.2e-32`。
- 概率和偏差 1.110e-15；`expect_z` 与闭式 Bloch 一致。
- **Ωy≠0**（Ωx=0.7, Ωy=−1.3, Ωz=0.4, |+x⟩）：H 偏差 0.0，`n_z` 偏差 1.110e-15，P↓ 偏差 9.992e-16，态偏差 vs 独立 expm 通过。
- **custom 初态**（(0.6, 0.8i)）：偏差 2.252e-15（expm）/ 1.388e-15（Bloch）。
- **Ωz 很大压峰**：Ωz=0.1 峰 0.990099009901；Ωz=20 峰 2.493765586035e-03，与 `1/(1+Ωz²/Ωx²)` 精确吻合。
- **时间反演**：`|U(t)U(−t)−I| = 4.441e-16`。

### 4.2 谐振子 ω=1, N=40

- H 与独立构造一致（偏差 0.0）；能谱 `max|Eₙ−(n+½)| = 7.105e-15`。
- 初态 |0⟩+|1⟩：态 vs 独立 `expm` 6.536e-14；`⟨x⟩` vs `cos(t)/√2` **4.996e-16**；`expect_x` 观察量与解析一致（5.551e-16）。
- **RK4 对拍**：产品态 vs 独立 RK4 6.686e-11。
- 相干态 α=1：`⟨x⟩` vs `√2 cos t` **6.800e-16**；参照实现截断泄漏 `1−Σ|cₙ|² = 0`（α=1,N=40，机器精度）。
- **截断诊断**：N=2、α=1 时 `truncation_diagnostic = fail`（边界占据 0.5，尾部概率 0.264），`overall = partial`，`failed=['truncation_diagnostic']`，detail 明确写"是模型截断限制，不等于数值发散"——符合 validation.md 第 12 项口径。

### 4.3 微扰 Δ=1, λ=0.1（**已按 Lead 更正后的口径**）

- H 与独立构造一致（偏差 0.0）。
- `E0_exact = −0.5024937810560445 = −½√(Δ²+λ²)`，与独立闭式解偏差 **0.0**。
- `E0_pt2 = −0.5025`，`E0_pt1 = −0.5`。
- `abs_err = 6.2189e-06`，`rel_err = 1.237616103905176e-05`，与独立计算偏差 0.0。
- **λ=5（超出适用范围）**：`perturbation_scope = skipped`（detail 写明"这是物理预期而非数值错误"，并给出 `lambda_valid_max = 0.5`），`solver_reference = pass`（`max|E_eigh−E_closed| = 4.44e-16`），`overall = pass`——**没有把物理近似偏离误判为求解器失败**。

## 5. 对抗性输入（`spec` 46 PASS + `cli` 24 PASS）

### 5.1 `spec.normalize_request` 直接调用（36 个非法用例 + 5 个正例 + 4 个规范化不变量）

全部非法用例抛 `SpecError`（可读中文消息），无一个"意外通过"。覆盖：

| 类别 | 用例 | 实际消息（节选） |
|---|---|---|
| 未知/重复参数 | `Omega_q`、`params` 与顶层重复 `Omega_x` | "未知参数 'Omega_q'…" / "参数 Omega_x 被重复指定" |
| 参数不属于模型 | spin 给 `omega` | "参数 'Omega_x' 不属于模型 harmonic_oscillator…" |
| 单位制 | `hbar=2`、`hbar="1"` | "暂不支持 units.hbar=2.0…" / "必须是实数" |
| 时间 | `n_steps=1`、`t_end=t_start` | "超出范围：要求 ≥ 2" / "必须大于 t_start" |
| 非有限/类型 | `nan`、`inf`、字符串 `"2"` | "必须是有限实数" / "必须是实数" |
| 退化参数 | Ω 全 0、ω=−1、ω=0、Δ=0、Δ<0 | "哈密顿量为零…" / "不能为 0" |
| 维数 | `n_levels=1`、`n_levels=401` | "超出范围：要求 ≥ 2 / ≤ 400" |
| 扫描 | 100 点、扫描不属于该模型的参数 | "超过上限 60" / "不属于模型…的可扫描参数" |
| 顶层字段/模型 | `banana`、`model="ising"`、`model=3` | "未知请求字段" / "不支持的模型" |
| 初态 | custom 零向量、`|α|=13`、`|α|=12.1`、kind=`cat`、Fock n=−1、Fock n≥N、叠加态 n≥N、全零振幅 | 全部拒绝 |
| 其它 | solver.method≠谱分解（修复后）、未知容差键、负容差、`formats=["pdf"]`、请求非 dict、缺 model | 全部拒绝 |

规范化不变量：custom 非归一矢量自动归一（`|v|=1.000000000000000`，`renormalized=True`）；默认 `t_end=2π/|Ω|`（2.8099258924162904）；`provenance` 标注默认/显式来源；canonical request 可 JSON 往返。

### 5.2 CLI 端（16 个非法请求 + 7 个 `--set` 注入）

- 16 个非法 `--spec`（含 `NaN`/`Infinity` 字面量、坏 JSON）**全部 exit 2**，stdout/stderr 无 `Traceback`、无 `File "` 帧，消息可读。
- `--set` 注入 7 种载荷（`__import__('os').system(...)`、`pathlib.write_text(...)`、`; import os`、`params.Nope=1`、非数值、裸 payload 路径、`1e999`）：**全部 exit 2**，canary 文件 `PWNED_CANARY.txt` **未生成**（证明没有被执行），无 traceback。
  - 注：第一版用例用"输出中是否含 PWNED"判断，因 CLI 会**回显**用户输入而误报；已改为 canary 文件判定（见 [tests/test_integration.py](tests/test_integration.py) `section_cli`）。
- 静态检查：`qllab/*.py` 中无 `scipy/matplotlib/pandas` 导入，无 `eval(`/`exec(`/`os.system(`/`subprocess.`/`__import__(`/`compile(`（排除 `re.compile`）出现（`cli.py` 不打印裸 traceback）。

## 6. 验证口径检查（`validation` 节，21 PASS）

随机抽 3 组参数（固定种子 20261112，模型/参数/时间/扫描各不相同，见 [tests/test_integration.py](tests/test_integration.py)）：

| 用例 | overall | counts | limits | skipped 原因 |
|---|---|---|---|---|
| 自旋 Ω=(3.5235, −1.6011, −0.9368), |↑⟩, t∈[0,3] | pass | 10/0/4 | 4 条，其中含模型/参数/时间边界与"未涵盖"清单 | 全部非空 |
| 谐振子 ω=1.9?, N=17, 相干态 | pass | 12/0/2 | 4 条，含 Fock 截断 N=17 说明 | 全部非空 |
| 微扰 Δ=2.1261, λ 采样 0.01…0.2 | pass | 8/0/6 | 4 条，含"只对当前 Δ 与采样点成立" | 全部非空 |

校验点：`overall ∈ {pass, partial, skipped}`；每个 `skipped` 都有非空 detail/原因文本；`limits` 非空且均为非空字符串；`counts` 之和 = `len(checks)`；`failed` 集合恰为 `status=="fail"` 的 id；`overall` 与检查项一致（有 fail→partial，无 fail 且有 pass→pass）。
`partial` 三态另由谐振子 N=2 用例覆盖（见 4.2）；`perturbation_scope` 超范围→`skipped` 另由 λ=5 用例覆盖（见 4.3）。

## 7. 报告与文件检查（`artifacts` 21 PASS + `models_artifacts` 16 PASS）

真实实验（自旋，CLI `run`）产物：

- 目录齐全：`request.json / result.json / validation.json / meta.json / data.csv / report.html / figures/`。
- **PNG**：4 张（`analytic_compare / bloch_expectation / energy / population`），PIL 可打开，全部 **3000×1760 PNG**。
- **report.html**：外部 `http(s)://` 引用 **0 个**；`data:image/png;base64,` 内嵌 4 段（= 图数）；`<h2>` 共 10 个且 9 类章节关键词（研究问题/模型/参数/数值方法/结果/验证/解释/复现/限制）齐全；含 `background-color` 状态色块 + 状态文案；声明 UTF-8。
- **data.csv**：UTF-8 **BOM** 存在；自旋表头 `t,pop_0,pop_1,expect_x,expect_y,expect_z,energy,pop_1_theory`（402 行）完整。
- `result.json / validation.json / meta.json` 关键字段齐全（含 `limits`、`parent_id`、`relation`、`duration_ms`）。

三个模型各跑一次真实 CLI 实验（`models_artifacts`）：

| 模型 | 图数 | 尺寸 | HTML 内嵌/外链 | data.csv 表头 |
|---|---|---|---|---|
| spin_half | 4 | 全部 3000×1760 | 4/0 | `t,pop_0,pop_1,expect_x,expect_y,expect_z,energy,pop_1_theory` |
| harmonic_oscillator | 8 | 全部 3000×1760 | 8/0 | `t,pop_0,pop_1,pop_2,pop_3,expect_x,expect_p,expect_n,boundary_pop,expect_x_theory,energy,expect_p_theory` |
| perturbation | 3 | 全部 3000×1760 | 3/0 | `lambda,E0_pt1,E0_pt2,E0_exact,abs_err,rel_err,within_threshold,E1_exact` |

`overall` 均为 `pass`（微扰 λ=0.2 在适用范围内）；CLI `sweep` 子命令 exit 0。

## 8. 端到端、复现与定向回归（`e2e` 17 + `regressions` 4 PASS）

`run → list → show → rerun → derive --set params.Omega_z=0 → compare → export`，全部 exit 0：

- `run --json` 摘要字段齐全（id/dir/overall/checks/scalars/figures/report/…），`overall=pass`。
- `list --json` 含新实验；`show --json` 可读。
- **rerun**：新 ID（≠父），`relation="rerun"`、`parent_id` 正确；`result.json` 的 `times/states/observables/scalars/metrics/eigen.values` 与首次**逐值完全一致（max diff = 0.0，≤1e-12）**。
- **derive**：新 ID，`params.Omega_z = 0.0`，`relation="derived"`、`parent_id` 正确。
- **父实验未被覆盖**：rerun 与 derive 前后父目录 `request.json/result.json/validation.json` 字节完全相同。
- `compare <id1> <id2> --json` 有 `ids/rows/diff`；`export --format html` 重新生成报告。
- **`selfcheck`：10/10 PASS，exit 0**。

### 8.1 定向回归用例（Lead 指定，`regressions` 节）

| 用例 | 判定口径 | 实测 |
|---|---|---|
| `warning_dedup` | 每条 request warning 在整次 `run` 的 stdout 中**恰好出现 1 次** | 3 条 warning 各出现 1 次（修复前 2 次/条；更早为 3 份合并） |
| `new_warnings_surfaced` | result 新增 warnings 仍需出现，且位于"补充提示"章节、各 1 次 | HO N=8 相干态：2 条新 warning（截断尾部 1.025e-05、边界占据 7.299e-05）各出现 1 次，`补充提示` 章节存在 |
| `omega_z0_energy_drift_consistent` | Ωz=0 时 `metrics`/`scalars`/`validation` 三方 `energy_drift` 一致 | 三方均为 **1.069e-16**，`energy_conservation = pass`（修复前 metrics 口径为 ~1e-4） |
| `partial_reported_as_partial` | 截断失败时 `overall=partial`、`failed=['truncation_diagnostic']` | 符合；注意 CLI 此时 exit 1（见 O2） |

## 9. 发现的问题与处理（按严重度排序）

> 我（verifier）未修改任何实现文件；以下均为"文件+行号+复现+期望/实际"，通过 `send_message` 报给 Lead，由对应负责人修复后我复测。

| # | 严重度 | 位置 | 问题 | 状态 |
|---|---|---|---|---|
| F1 | 高（需求侧） | 任务书 | 微扰期望值写成 `E0_exact=-0.50124988, rel_err≈0.0024937`，与 CONTRACT/models.md 的 `-½√(Δ²+λ²)` 及物理不符（正确：`-0.5024937810560445`，`rel_err=1.2376e-05`） | **已澄清**：Lead 确认是笔误，以 CONTRACT 为准，我方独立计算正确；相关实现与验收线已更正，已记入本报告 |
| F2 | 高 | `spec.py:458` | f-string 未转义 `e^{-iHt}` → 非法 `solver.method` 触发 `NameError`，CLI 打印完整 traceback 且 exit 1 | **已修复**，复测：exit 2、单行可读 SpecError、无 traceback |
| F3 | 高 | `models.py:733` `_ho_figures` | 访问不存在的 `obs["expect_p_theory"]` → 谐振子 `run_model` 对所有初态 `KeyError`，模型完全不可用 | **已修复**，复测：谐振子 8 项物理检查全 PASS |
| F4 | 高 | `cli.py:162` | 调用未定义的 `_dedupe` → 所有真实 `run` exit 1（NameError），artifacts/e2e 全线阻断 | **已修复**，复测：全量 197 PASS |
| F5 | 中 | `report.py:905` / `models.py` series | 微扰 `data.csv` 输出 `lambda,observables,scalars,value`，物理列缺失且数值被塞进 dict 字符串 | **已修复**，复测表头含全部必需列（见第 7 节） |
| F6 | 中 | `cli.py` `_load_raw` | `--request` 文件不容忍 UTF-8 BOM，报"请求不是合法 JSON"（Windows 常见） | **已修复**，复测：带 BOM 请求正常解析 |
| F7 | 中 | `spec.py:273` | 新增截断检查时 `def _canon_time(...)` 与下一行 `if` 挤在同一行 → 全模块 SyntaxError、CLI 全部命令 exit 1 | **已修复**，复测 `py_compile` 0 且全量通过 |
| F8 | 低 | `cli.py` `_execute_pipeline` / `_print_summary` | 请求自带 warnings 在"确认块"和"执行后摘要"各打印一次（同一进程共 2 次/条；`_dedupe` 已把原先 3 份合并为 1 份/章节） | **已修复（采纳方案 a）**：摘要改为只打印 result 新增的 `new_warnings`（标题"补充提示："），`warnings` 字段仍保留完整列表；复测每条 request warning 恰好出现 1 次，result 新提示仍出现 |
| O1 | 观察 | 本仓库（技能目录） | 技能目录残留非交付物：`tests/`（含 `probe_*`、`_probe_sub*`、`_tmp/`、`__pycache__`）与 `_chart_smoke.png`、`_chart_smoke2.png`；部分 `probe_*` 子目录当前拒绝访问（ACL 残留） | **Lead 已清理**（26 个早期测试残留目录 + 4 个 probe 目录 + `__pycache__` + 冒烟图；`tests/test_*.py` 作为交付物保留）。清理方式：`diagnose-windows-sandbox-acl` 判定 `NOT_THIS_CLASS`（当前用户已有 WRITE_DAC/WRITE_OWNER、无 AppContainer 包 ACE），故用一次经批准的完全权限删除，删除前逐条校验名称前缀与父目录 |
| O2 | 观察（口径） | `cli.py` 退出码 | `overall=partial`（计算完成但截断诊断 fail）时 CLI 返回 **1**，与 CONTRACT 中"1 = 执行失败"的注释无法区分；validation.md 明确 partial 属于"有完整结果" | **Lead 已在 `SKILL.md` 与 `CONTRACT.md` 写明**：`1 = 执行失败或 overall=partial`，并明确"判可交付性要看摘要 `overall`/`failed`，不能只看退出码"；本次已用 `partial_reported_as_partial` 用例固定该行为 |

**意外通过（不该被接受却被接受）**：无。
**未发现**：任何物理数值错误、任何"把物理近似偏离说成求解器失败"、任何外部网络引用、任何代码执行漏洞。

## 10. 残余风险与未覆盖项

1. **`overall="skipped"`（全部检查 skipped）未构造出真实用例**：14 项检查中最少也有 `hermiticity/finite_values` 等与模型无关的项可执行，产品代码按 `pass` 处理；本报告只验证了 `pass` 与 `partial` 两种 overall。
2. **同一修订的并发性**：验证期间队友仍在改文件（F2–F7 都是过程中发现并修复的）。本报告以 2 节列出的 hash 为准；若这些文件之后再有改动，需重跑全量。
3. **性能边界未压测**：`n_steps=20000`、`n_levels=400`、扫描 60 点等上限只验证了"参数被接受/拒绝"，未测最坏耗时与内存。
4. **图表视觉质量未做人工目检**：只验证了尺寸、可解码、无外链、内嵌与文字关键词；曲线是否被裁切、中文 glyph 是否缺字未逐张人工确认（依赖 `C:\Windows\Fonts\msyh.ttc`）。
5. **平台耦合**：字体路径为 Windows 专用；Linux/macOS 回退路径未验证。
6. **退出码语义（O2）**：`overall=partial` 与"执行失败"共用 exit 1；脚本调用方不能仅凭退出码区分。已用 `partial_reported_as_partial` 固定行为，但建议在 CONTRACT/SKILL.md 明确。
7. **技能目录残留（O1）**：`tests/`、`_chart_smoke*.png` 等非交付物尚未清理（部分 `probe_*` 目录 ACL 拒绝访问）；不影响计算正确性，但影响技能包整洁度。
8. **验证证据文件**：`tests/_artifacts/` 保留全部运行产物（多轮实验目录、结果 JSON），如发布技能包需排除。
9. **未覆盖的物理/能力边界**（与 SKILL.md 一致）：ħ≠1、含时哈密顿量、简并/近简并微扰、开放系统、多体纠缠、文献检索扩展等均未实现也未被本验证要求覆盖。

## 11. 最终判定

在上述模型、参数、检查项与修订范围内：

- **物理正确性**：通过（与独立 Taylor/RK4/闭式解一致，偏差见第 0/4 节，远优于约定阈值；微扰口径已按 Lead 更正后的 CONTRACT 验证）。
- **输入安全边界**：通过（36+16+7 个对抗用例全部被拒，无 traceback、无代码执行）。
- **验证口径与交付物**：通过（三态语义、skipped 原因、limits、9 章 HTML、内嵌图、3000×1760 PNG、utf-8-sig CSV 列齐全）。
- **端到端与复现**：通过（rerun 逐值一致 0.0、父实验不被覆盖、selfcheck 10/10）。
- **待办（不阻塞结论）**：O1 技能目录残留清理、O2 退出码语义写进文档 —— **两项均已由 Lead 完成，见第 12 节。**

## 12. 交付前收尾（Lead，2026-11-12 12:10–12:20）

在 verifier 定稿之后，Lead 做了以下收尾改动，并**重跑全量**确认结论不变：

| 变更 | 文件 | 说明 |
|---|---|---|
| 退出码语义文档化（O2） | `SKILL.md`、`docs/CONTRACT.md` | 写明 `1 = 执行失败或 overall=partial`，要求以摘要 `overall`/`failed` 判别 |
| 扫描表列契约 | `references/report-format.md` | 明确 `<param> + 标量列`、微扰固定 7 列、CSV 与 HTML 同源 |
| overall 口径澄清 | `references/validation.md` | `validate` 不返回"执行失败"，该状态属于 CLI/报告层 |
| 默认扫描点数 | `references/models.md` | 31 → 32 点（与 spec 实现一致） |
| figspec `ylog` | `docs/CONTRACT.md` | 补上 models 实际使用的最后一个扩展键 |
| CLI `--spec` 提示 | `scripts/qllab/cli.py` | PowerShell 会剥掉内联参数双引号；解析失败时给出改用 `--request`/`--stdin` 的可读建议 |
| 目录清理（O1） | 技能目录、示例目录 | 删除 26 个早期测试残留目录、4 个 probe 目录、`__pycache__`、临时冒烟图；`tests/_artifacts` 从 93.7 MB 精简为 1 个最终结果 JSON |

收尾后的复测（Lead 执行，同一修订）：

```powershell
cd <仓库根>
& $PY -m unittest discover -s "tests" -t "tests" -p "test_*.py"   # Ran 73 tests, OK
& $PY tests\test_integration.py                              # SUMMARY {'PASS': 201, 'FAIL': 0, 'BLOCKED': 0}
& $PY scripts/quantum_lab.py selfcheck   # 10/10 PASS, exit 0
```

结果文件：`tests/_artifacts/results-20261112-121802.json`（该目录被 `.gitignore` 排除）。
cli.py 收尾改动后的 hash：`DE9DC4B34FB86757`（其余模块 hash 与第 2 节一致）。
演示实验（5 组 + rerun/derive/compare/export，全部 exit 0）：`examples/`（成品见 `examples/sample-experiment/`）。

## 13. GitHub 仓库布局下的复跑（Lead，2026-11-12 12:54）

源码整理为仓库布局（仓库根即技能目录，`CONTRACT.md` 移入 `docs/`，集成套件改为自动定位技能根目录）后，
在原环境重新跑通全部检查，结论不变：

| 命令 | 结果 |
|---|---|
| `python -m unittest discover -s tests -t tests -p "test_*.py"` | Ran 73 tests，OK |
| `python tests/test_integration.py` | 201 PASS / 0 FAIL / 0 BLOCKED |
| `python scripts/quantum_lab.py selfcheck` | 10/10 PASS，exit 0 |
| `python tests/reference_physics.py` | 最差偏差 1.225e-10 |
| `python examples/run_examples.py` | 5 组 run + list/rerun/derive/compare/export 全部 exit 0 |

布局无关性：`tests/test_integration.py` 通过向上查找 `scripts/quantum_lab.py` 定位技能根目录
（也可用环境变量 `QUANTUMLAB_ROOT` 指定），因此既能在仓库根运行，也能在
`<workspace>/.dsh/skills/quantum-lab/tests/` 的 DSH 安装布局下运行。
