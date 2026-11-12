# 更新日志

本项目遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 与 [语义化版本](https://semver.org/lang/zh-CN/)。

## [1.0.0] - 2026-11-12

首个完整版本：优先保证**单次实验的计算正确性、验证透明度与结果可复现性**。

### 新增

- **三类物理模型**：自旋 1/2 二能级（Bloch 进动）、量子谐振子（截断 Fock 空间）、非简并微扰
  （一阶/二阶能量 vs 精确对角化），统一采用 ħ=1 的无量纲约定。
- **结构化请求**（`qllab/spec.py`）：模型/参数/初态/时间/单位/求解器/扫描/容差/输出的规范化与白名单校验，
  参数别名（`Ωx`、`ω`、`Δ`、`λ`、`初态` 等）与默认值来源标注（`provenance`）。
- **验证内核**（`qllab/validate.py`）：14 项检查（输入合法性、有限性、厄米性、本征残差、正交归一、
  态归一化、概率守恒、能量实数/守恒、解析对比、能谱对比、截断诊断、求解器参照、微扰适用范围），
  每项含指标、容差、`pass/fail/skipped` 状态与适用边界 `limits`。
- **声明式图表**（`qllab/charts.py`）：纯 Pillow 渲染 3000×1760 PNG，自动 1/2/5×10ᵏ 刻度、对数轴、
  图例、注脚、参考线、中文/数学符号字形兜底。
- **报告导出**（`qllab/report.py`）：9 章中文报告，单文件 HTML（图 base64 内嵌、打印友好）、
  JSON、`utf-8-sig` CSV（时间序列与扫描表）。
- **实验库**（`qllab/store.py`）：实验编号与目录、原子写索引、按模型/参数/文本检索、
  跨实验对比、父子关系（`rerun` / `derived`）。
- **命令行**（`qllab/cli.py`）：`run / sweep / list / show / rerun / derive / compare / export / selfcheck`，
  `--json` 摘要，退出码 0/1/2，`derive --set` 仅接受白名单字段路径。
- **Agent Skill**：[SKILL.md](SKILL.md) 固化「理解 → 澄清 → 确认 → 计算 → 验证 → 解释 → 保存」流程，
  规定结论口径（不得把 `partial` 说成通过、不得用边界占据证明收敛、不得静默替换用户条件）。
- **测试与验证**：73 项单元测试；独立红队套件 201 项（Taylor 级数 / RK4 / 闭式解交叉对拍、
  对抗性输入、交付物格式、端到端与回归），报告见 [docs/VERIFICATION.md](docs/VERIFICATION.md)。

### 说明

- 绘图不依赖 matplotlib，报告中文字体优先使用 Windows 的 Microsoft YaHei，缺失时回退并替换字形。
- 未实现：ħ≠1 单位制、含时哈密顿量、简并微扰、开放系统、多体与大规模并行扫描、文献检索与外部模型服务
  （遇到不支持的条件会明确说明能力边界）。
