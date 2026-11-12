---
name: quantum-lab
description: 用受控本地数值实验做量子力学计算与验证：自旋 1/2 二能级系统的哈密顿量与时间演化、量子谐振子（截断 Fock 空间）能谱与叠加态演化、非简并微扰一阶/二阶能量与精确对角化对比；输出 png 图、csv、json 与独立 html 报告，并附物理一致性检查表。当用户要求"计算/验证量子力学模型、画跃迁概率或演化曲线、比较近似与精确解、扫描参数、复现或对比之前的量子实验"时使用。
---

# QuantumLab：量子力学计算与验证

本技能把自然语言问题变成**受限的结构化请求**，由本地求解器（numpy 谱分解）完成计算，
再用固定的检查表验证物理一致性与数值误差，最后生成图、数据与报告。

硬规则（先读）：

1. **不执行任意代码/表达式/命令**。只接受白名单模型与参数；用户说"运行这段 Python"
   一律拒绝并说明能力边界。
2. **先确认再计算**。执行前把模型、哈密顿量、参数（标注哪些是默认值）、初态、单位、
   时间范围、求解方法给用户看；参数有歧义先问（典型：只说"Ω=2"未说方向）。
3. **不把未通过验证的结果说成已验证**。状态三态：执行失败 / 完成但部分检查未通过 /
   所执行检查通过。结论必须限定在模型、参数与检查范围内。
4. **不编造数据**。解释只能引用本次结果里的数值；没跑的实验不能说跑了。
5. 用户明确指定的条件若不受支持，**报缺失能力**，不得静默替换成默认演示参数
   （例如 ħ≠1、时变哈密顿量、简并微扰、开放系统）。

## 运行方式

脚本在本 SKILL.md 同目录的 `scripts/quantum_lab.py`。Python 需要 numpy + Pillow；
若默认解释器缺依赖，用 `load_workspace_dependencies` 返回的 python 绝对路径。

```powershell
$QL = "<本技能目录>\scripts\quantum_lab.py"
$PY = "C:\Users\HUAWEI\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe"
& $PY $QL run --request req.json --out-root <输出目录>   # 推荐：请求写成 JSON 文件
'{"model":"spin_half","params":{"Omega_x":2,"Omega_z":1}}' | & $PY $QL run --stdin   # 或管道
& $PY $QL list --json                          # 历史实验
& $PY $QL derive <id> --set params.Omega_z=0   # 改参数另存新实验
& $PY $QL compare <id1> <id2> --json           # 跨实验对比
& $PY $QL export <id> --format html            # 重新导出报告
```

**PowerShell 注意**：Windows PowerShell 会把内联参数里的双引号吃掉，
所以 `--spec '{"model":...}'` 在 pwsh 下会变成非法 JSON（CLI 会给出提示）。
请用 `--request <文件>`，或把 JSON 先赋给变量再通过管道 `--stdin` 传入；
`--spec` 更适合 bash/zsh 或程序化调用。

请求也可以放文件：`run --request req.json`（兼容带 BOM 的 UTF-8 文件）。

**退出码语义**（脚本与 CI 都要依赖它）：

| 退出码 | 含义 |
|---|---|
| `0` | 执行完成，且所执行的检查全部通过（含全部 `skipped`） |
| `1` | 执行失败（异常/数值不可用），**或**执行完成但验证为 `partial`（有 `fail`） |
| `2` | 请求非法：字段/参数/单位/规模不在白名单，或缺依赖——stderr 只有一行可读消息，不打印 traceback |

因此脚本判"是否可交付"要看摘要里的 `overall`，不能只看退出码是 `0/1`：
`1` 时必须读 `failed` 与各检查的 `metric/tolerance` 区分"执行失败"与"部分验证未通过"。

## 工作流程

### 1. 理解与澄清
从用户话里提取：物理对象、模型参数、计算任务、观察量、验证要求、输出要求。
缺关键信息时只问真正影响计算的问题，例如：
- "Ω=2" 是 Ωx（横向驱动）还是 Ωz（纵向/失谐）？
- 自旋初态是 |↑⟩ 还是叠加态？（不问则按 |↑⟩ 并在确认块里写明默认值）
- 微扰强度 λ 与 Δ 的比值范围；谐振子截断维数 N。
一次最多问 2–3 个问题；其余采用默认值并在确认块中标注"默认值"。

### 2. 组装结构化请求
把理解结果写成下面的 JSON（字段细节见 `references/models.md`）：

```json
{
  "model": "spin_half",
  "params": {"Omega_x": 2.0, "Omega_z": 1.0},
  "initial_state": {"kind": "up"},
  "time": {"t_start": 0.0, "t_end": 6.283185307179586, "n_steps": 401},
  "sweep": null,
  "question": "用户原始问题（原样记录）"
}
```

- `model`：`spin_half` | `harmonic_oscillator` | `perturbation`。
- 参数名严格用 `Omega_x/Omega_y/Omega_z`、`omega`、`n_levels`、`delta`、`lambda`。
- 只写用户/你确认过的值；其余交给默认值（`--spec` 省略即可），默认值会在确认块显示。
- `sweep`：`{"param": "Omega_z", "values": [0, 1, 2]}`（≤60 点）。
- 用户原始问题放进 `question`，用于实验记录与复现。

### 3. 执行并读回状态
`run` 会打印确认块、验证表、实验 ID 与目录。必须检查摘要里的 overall：
- `执行失败`：无完整结果 → 报告原因，不说"完成"。
- `部分验证未通过`：列出失败项与误差/容差，判断是近似偏离、输入问题还是求解器问题。
- `通过`：说明"在本次所执行的检查范围内通过"。

### 4. 解释结论（三段式）
1. **发生了什么**：关键数值、曲线特征（峰值、周期、单调性、误差量级）。
2. **为什么**：用模型机制解释（拉比振荡、Ωz 限制最大跃迁概率、相对相位、
   二阶微扰按 λ²/Δ 缩放、截断导致边界占据）。
3. **限制**：误差、近似适用条件、截断、未验证的部分。

### 5. 交付
把图用 Markdown 引用（`![跃迁概率](绝对路径.png)`），并给出报告路径。
用户要报告时用 `export --format html`（单文件、内嵌图、可打印为 PDF）。

## 三类模型速查

| 模型 | 哈密顿量（ħ=1） | 关键输出 | 必做验证 |
|---|---|---|---|
| `spin_half` | H=½(Ωxσx+Ωyσy+Ωzσz) | 本征值、两个基态占据、⟨σx,y,z⟩、与 Bloch 解析解对比 | 厄米性、归一化、概率守恒、概率和=1、解析偏差 |
| `harmonic_oscillator` | H=ω(a†a+½)，N 能级截断 | 能谱、能级占据、⟨x⟩(t)、⟨p⟩(t)、边界占据 | 能谱 vs n+½、⟨x⟩ vs 解析、截断诊断 |
| `perturbation` | H=(Δ/2)σz+(λ/2)σx | E0 精确、一阶、二阶、绝对/相对误差、λ 适用范围 | 精确 vs -½√(Δ²+λ²)、近似偏离（非数值错误） |

常用定性结论（用于解释，仍需以本次数值为准）：
- 自旋：最大跃迁概率 = Ω⊥²/|Ω|²（Ω⊥²=Ωx²+Ωy²），Ωz≠0 时 <1；振荡角频率 = |Ω|。
- 谐振子：相干态 ⟨x⟩ 以 ω 振荡，振幅 √2|α|；叠加态的相对相位驱动 ⟨x⟩ 振荡；
  最高能级占据小**不等于**已证明收敛。
- 微扰：E0(二阶)=-Δ/2-λ²/(4Δ)，相对误差随 λ/Δ 增长；适用范围只对采样点成立。

详细公式、默认值与适用条件：`references/models.md`。

## 参数扫描与对比

- 单次请求内扫描：`sweep.param` 取该模型参数，最多 60 点。执行前说明规模与耗时。
- 跨请求对比：`derive` 生成新实验（保留父实验），再用 `compare` 出对比表/叠加图。
- 大规模扫描（>60 点）应拆分并说明成本；不静默截断用户请求。

## 历史与复现

`list` / `show` / `rerun` / `derive` / `compare` 覆盖：查找上次实验、同参数重跑、
改参数另存、最近两次对比。每个实验目录含 request/result/validation/meta/数据/图/报告，
`parent_id` 与 `relation` 记录派生关系。重跑永远生成新 ID，不覆盖原始结果。

## 验证与报告字段

检查项、指标、默认容差与状态语义见 `references/validation.md`；
目录结构、JSON 字段、CSV 列与 HTML 章节见 `references/report-format.md`。

## 可选扩展（默认不启用）

- 文献证据：接入检索服务后，报告中须分开标注"文献结论 / 本地计算 / 综合推断"；
  检索成功不代表计算通过，检索失败不阻塞本地计算。
- 外部模型服务（OpenAI/DeepSeek/GLM/Kimi/Claude/兼容接口）：只用于问题解析与解释；
  科学计算仍由本地求解器完成；密钥不写入实验文件、报告与普通日志；调用前说明可能计费。

## 自检与验证

```powershell
$SKILL = "<本技能目录>"
& $PY "$SKILL\scripts\quantum_lab.py" selfcheck     # 3 条物理基准，期望 10/10 PASS
# 单元测试（73 项，模型/图表/实验库）
& $PY -m unittest discover -s "$SKILL\tests" -t "$SKILL\tests" -p "test_*.py"
```

在 workspace 根还可跑独立验证套件 `python tests/test_integration.py`（201 项：独立参照对拍、
对抗性输入、交付物格式、端到端与回归），其参照实现 `tests/reference_physics.py` 用
Taylor 级数 / RK4 / 闭式解，**不依赖**本技能的谱分解；结论见 `docs/VERIFICATION.md`。
演示实验（三类模型 + 参数扫描 + rerun/derive/compare/export）见 `examples/`。

## 能力边界

不支持：任意 Python/命令执行、ħ≠1 的单位制、显含时哈密顿量、多体/多比特纠缠、
简并微扰与近简并情形、开放量子系统（退相干/Lindblad）、大规模并行扫描。
遇到这些：说明缺什么、给出可行替代（如改参数范围、先做单比特近似、改用解析推导）。
