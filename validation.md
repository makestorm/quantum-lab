# 验证规范：检查项、容差与结论口径

验证的目的不是"让结果看起来正确"，而是把**误差指标、容差、适用范围**摊开给用户看。
每次计算都必须输出完整检查表；不适用的项标 `skipped` 并写明原因，不能省略。

## 检查项

| # | id | 名称 | 指标 | 默认容差 | 适用模型 |
|---|---|---|---|---|---|
| 1 | `input_legality` | 输入合法性 | 参数类型/范围/维数是否通过规范化 | — | 全部 |
| 2 | `finite_values` | 数值有限性 | 所有输出中的 NaN/Inf 计数（应为 0） | 0 | 全部 |
| 3 | `hermiticity` | 厄米性 | `max|H - H†|` | `hermiticity_abs`=1e-10 | 全部 |
| 4 | `eigen_residual` | 本征方程残差 | `max|H v_k - E_k v_k|` | `eigen_residual_abs`=1e-8 | 全部 |
| 5 | `orthonormality` | 正交归一性 | `max|V†V - I|` | `orthonormality_abs`=1e-8 | 全部 |
| 6 | `state_norm` | 态归一化 | 演化全程 `max|||ψ(t)|-1|` | `norm_dev`=1e-9 | spin、HO |
| 7 | `probability` | 概率守恒 | `max|Σ占据 - 1|` | `probability_sum`=1e-9 | spin、HO |
| 8 | `energy_real` | 能量实数 | `max|Im E_k|` | `energy_imag`=1e-10 | 全部 |
| 9 | `energy_conservation` | 能量守恒 | `max|⟨H⟩(t) - ⟨H⟩(t₀)| / max(|⟨H⟩|, 1)` | `energy_drift`=1e-8 | spin、HO |
| 10 | `analytic_agreement` | 解析结果对比 | 数值与解析解最大绝对偏差（指标名随模型） | `analytic_abs`=1e-6 | spin、HO |
| 11 | `spectrum_agreement` | 能谱对比 | `max|E_n - ω(n+1/2)|` | `spectrum_abs`=1e-8 | HO |
| 12 | `truncation_diagnostic` | 截断诊断 | 边界能级最大占据 `boundary_pop`（+相干态展开泄漏） | `truncation_pop`=1e-6 | HO |
| 13 | `solver_reference` | 求解器参照 | 精确对角化 vs 闭式解 `-½√(Δ²+λ²)` | `analytic_abs`=1e-6 | perturbation |
| 14 | `perturbation_scope` | 微扰适用范围 | 当前 λ 下二阶近似相对误差 `rel_err` | `perturbation_rel`=1e-2 | perturbation |

用户可用 `tolerance` 覆盖任意项（值必须为正实数）。

## 状态语义

- `pass`：指标 ≤ 容差。
- `fail`：指标 > 容差，或出现 NaN/Inf、维度不符等硬错误。
- `skipped`：该检查不适用于本次任务（必须给出原因文本）。

**第 12、14 项的特殊口径**：
- `truncation_diagnostic`：指标在容差内 → `pass`（边界泄漏可忽略，但**只是本次条件下**）；
  超出 → `fail` 并提示增大 `n_levels`，同时说明这不等于"数值发散"，而是模型截断不足。
- `perturbation_scope`：当前 λ 在容差内 → `pass`（近似可用）；超出 → `skipped`，
  detail 写明"近似已超出适用范围，这是物理预期而非数值错误"，并给出 `lambda_valid_max`。
  **绝不能**把它记成求解器失败。

## 总状态（三态，报告与终端必须一致）

| overall | 含义 | 说法 |
|---|---|---|
| 执行失败 | 没有完整结果（异常、非法输入、数值发散到不可用）；由 CLI 层判定并给出退出码 | "执行失败：<原因>"，不得描述结果 |
| `partial` | 有完整结果，但至少一项检查 `fail` | "计算完成，但有 N 项验证未通过：…" |
| `pass` | 所有执行的检查通过（全部 skipped 除外） | "在本次执行的检查范围内通过" |

注：`validate.validate()` 只返回 `pass` / `partial` / `skipped`（全部 skipped 时）；
"执行失败"是 CLI/报告层的状态，`validate` 不返回它。

结论必须附带 `limits`（适用边界），例如："仅对 Ωx=2、Ωz=1、|↑⟩ 初态、
t∈[0,2π/√5] 成立；未验证更大 Ωz、未验证含时驱动、未验证开放系统。"

## 解释时的禁止事项

1. 不得把 `partial` 说成"结果正确"；不得只报图不报失败项。
2. 不得用"边界占据很小"证明收敛（只能说"在当前条件下边界泄漏可忽略"）。
3. 不得把微扰近似偏离精确解说成"程序算错了"。
4. 不得省略容差与误差数值；误差要给出量级（如 `3.2e-16`）。
5. 不得宣称验证了未执行的检查（例如未做解析对比就说"与解析解一致"）。
