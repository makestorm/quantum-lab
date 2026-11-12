# 贡献指南

感谢愿意一起改进 QuantumLab。这个项目的第一原则是**诚实**：宁可少报结论，也不要让用户拿到
未被验证（或验证未通过）却看起来"成功"的结果。

## 开发环境

```bash
git clone https://github.com/ZhouFeng/quantum-lab.git
cd quantum-lab
python -m venv .venv && . .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt                 # numpy + Pillow，仅此两个
```

运行测试与自检（提交前请全部通过）：

```bash
python -m unittest discover -s tests -t tests -p "test_*.py"
python scripts/quantum_lab.py selfcheck
python tests/test_integration.py            # 独立交叉/对抗/端到端套件
python tests/reference_physics.py           # 参照实现自检
python examples/run_examples.py             # 示例冒烟（产物写在 examples/experiments/）
```

## 架构与接口

模块职责与数据结构（canonical request、result、validation report、figspec）定义在
[docs/CONTRACT.md](docs/CONTRACT.md)；模型公式、默认值与解析参照在
[references/models.md](references/models.md)；检查项与容差口径在
[references/validation.md](references/validation.md)。

改动数据结构的成本很高——如果确实需要，请同时更新契约、对应 `references/*.md`、测试与
[docs/VERIFICATION.md](docs/VERIFICATION.md) 里的相关结论。

## 代码要求

- 只使用 **numpy + Pillow + 标准库**；不要引入 scipy / matplotlib / pandas。
- **禁止** `eval` / `exec` / `os.system` / `subprocess` / `__import__`（`tests/test_integration.py`
  的 `static` 节会静态检查并失败）。
- 用户输入只能作为数据处理：新增参数必须进白名单并写明范围、类型、默认值与单位。
- 数值输出必须有限；新增指标要在 `validate` 中给出指标值、容差与适用条件。
- 面向用户的中文文案要可读、不含内部术语与堆栈；错误信息用 `SpecError` / `_InputError` 表达。
- 注释与 docstring 用中文，说明"为什么"而不是复述代码。

## 物理正确性

新增或修改模型必须提供**独立参照**：

1. 在 `tests/reference_physics.py` 里用不同算法（Taylor 级数、RK4、闭式解）实现一遍；
2. 在 `tests/test_integration.py` 的 `physics` / `ref` 节加交叉校验，阈值按现有档位
   （解析 1e-8、数值积分 1e-6、机器精度 1e-12）；
3. 在 `tests/test_models.py` 加带基准数值的单元测试（例如 `Ωx=2, Ωz=1` 的峰值必为 0.8）。

不要让产品代码与参照实现共用哈密顿量构造函数——独立才叫验证。

## 提交规范

- 提交信息用祈使句、中文或英文皆可，首行 ≤ 72 字符，例如
  `fix(models): 谐振子 boundary_pop 在 N=2 时越界`。
- 一个 PR 只做一件事；行为变化必须带测试，口径变化必须带文档。
- PR 描述里请写：改了什么、为什么、如何验证（命令 + 关键数值）、对既有结论的影响。

## 报告问题

请在 issue 中附上：请求 JSON（或复现命令）、退出码、摘要里的 `overall` 与失败检查、
预期值与实际值。**不要**粘贴任何密钥或凭据。
