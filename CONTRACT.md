# qllab 内部接口契约（v1）

本文件是拆并行开发的唯一接口约定。所有模块只依赖 `numpy` + `Pillow`（标准库可用），
**不得**引入 scipy / matplotlib / pandas 等未安装依赖。

数据流：

```
raw request (dict)
   └─ spec.normalize_request() -> canonical request (dict)      [Lead 维护]
        └─ models.run_model(req) -> result (dict, 可含 numpy)    [physics]
             ├─ validate.validate(req, result) -> report(dict)   [physics]
             ├─ charts.render_all(result) -> figures(list)       [viz]
             └─ store.create_experiment(...) / report.*          [store/cli, viz]
```

约定：**函数之间只传 dict / list / numpy.ndarray**，不定义额外 class 层级。
所有对外写出的 JSON 必须经 `util.to_jsonable()`。

## 1. canonical request

`spec.normalize_request(raw)` 返回：

```python
{
  "schema": "quantumlab/1",
  "model": "spin_half" | "harmonic_oscillator" | "perturbation",
  "params": {...},            # 已填默认值、已完成类型/范围检查
  "initial_state": {...},     # 各模型自己的规范形式
  "time": {"t_start": 0.0, "t_end": float, "n_steps": int},   # perturbation 无此项时为 None
  "units": {"hbar": 1.0, "note": "..."},
  "solver": {"method": "eigendecomposition", "note": "..."},
  "sweep": None | {"param": str, "values": [float, ...]},
  "tolerance": {name: float},  # 见 spec.defaults
  "output": {"figures": True, "formats": ["json", "csv", "png", "html"]},
  "question": str,             # 用户原始问题，可为 ""
  "warnings": [str],           # 归一化过程中产生的提示，如“Ω 方向未指明，按 Ωx 处理”
}
```

`params` 各模型：

| model | 参数 | 类型 | 默认 | 限制 |
|---|---|---|---|---|
| spin_half | `Omega_x` | float | 0.0 | \|Ω\|≤1e6 |
| | `Omega_y` | float | 0.0 | 同上 |
| | `Omega_z` | float | 0.0 | 同上 |
| | | | | 三者不能全 0 |
| harmonic_oscillator | `omega` | float | 1.0 | 0 < ω ≤ 1e5 |
| | `n_levels` | int | 40 | 2 ≤ N ≤ 400 |
| perturbation | `delta` | float | 1.0 | 0 < Δ ≤ 1e6（非简并要求） |
| | `lambda_` | float | 0.1 | 0 ≤ λ ≤ 1e6，参数名对外写作 `lambda` |

初始态：

```python
spin_half:  {"kind": "up"|"down"|"plus_x"|"plus_y"|"custom",
             "vector": None | [[re, im], [re, im]]}     # custom 时为 2 分量复向量
oscillator: {"kind": "fock"|"coherent"|"superposition",
             "n": int, "alpha": [re, im],
             "components": [{"n": int, "amplitude": [re, im]}, ...]}
perturbation: None
```

## 2. result（`models.run_model`）

```python
{
  "model": str,
  "params": dict,                 # 实际使用的参数
  "hamiltonian": np.ndarray,      # 复方阵
  "basis": {"labels": [str], "dim": int, "truncated": bool, "note": str},
  "eigen": {"values": np.ndarray, "vectors": np.ndarray},
  "times": np.ndarray | None,
  "states": np.ndarray | None,    # (n_steps, dim) 复振幅
  "observables": {str: np.ndarray},   # 随时间的量：pop_0, pop_1, expect_x, ...
  "scalars": {str: float|list|str},   # 不随时间的量：E0_exact, E0_pt2, ...
  "analytic": None | {"name": str, "formula": str, "values": dict, "note": str},
  "sweep": None | {"param": str, "values": [..], "series": [{..}, ..]},
  "metrics": {str: float},        # 误差类指标，供 validate 使用
  "figure_specs": [figspec, ...], # 见第 4 节
  "warnings": [str],
  "summary": {"title": str, "bullets": [str]},   # 结果概述，非最终解释
}
```

## 3. validation report（`validate.validate`）

```python
{
  "checks": [
    {"id": "hermiticity", "name": "厄米性", "status": "pass"|"fail"|"skipped",
     "metric": {"name": "max|H-H†|", "value": 0.0},
     "tolerance": 1e-10, "detail": "线性代数意义上的 H=H†", "scope": "..."},
  ],
  "overall": "pass" | "partial" | "skipped",   # 全部通过 / 有 fail / 全部 skipped
  "counts": {"pass": int, "fail": int, "skipped": int},
  "failed": ["hermiticity"],
  "limits": [str],       # 结论适用边界
}
```

## 4. figspec（声明式图描述）

```python
{
  "name": "population",          # 文件名 stem，仅 [a-z0-9_]
  "kind": "line"|"spectrum"|"error",   # spectrum/error 仍按 line 渲染，仅影响默认样式
  "title": str, "xlabel": str, "ylabel": str,
  "series": [{"label": str, "x": list[float], "y": list[float],
              "style": "line"|"dashed"|"dots"|"scatter"}],
  "hlines": [{"y": float, "label": str, "style": "dashed"}],   # 可选
  "vlines": [{"x": float, "label": str}],                      # 可选
  "ylim": None | [lo, hi], "xlim": None | [lo, hi],
  "ylog": False,                  # True 时纵轴取对数（下限 1e-16，图注需说明）
  "caption": str,                 # 图注，报告里使用
  "notes": [str],                 # 参数/单位说明，画在图底部
}
```

## 5. charts

```python
charts.render(figspec, width=1500, height=880, scale=2) -> bytes   # PNG bytes
charts.render_all(result) -> [{"name": str, "png": bytes, "caption": str}]
```

中文字体：`C:\Windows\Fonts\msyh.ttc`（缺失时回退 DejaVuSans，再回退默认）。图中必须出现
横纵坐标标签、标题、曲线图例、参数说明。

## 6. store

```python
store.ROOT_DEFAULT = "<cwd>/quantumlab-experiments"
store.create(root, req, result, validation, figures, meta) -> exp_id
store.load_index(root) -> [entry, ...]
store.find(root, query) -> [entry, ...]
store.get(root, exp_id) -> {"meta":..,"request":..,"result":..,"validation":..,"figures":{name:path}}
store.compare_entries(root, ids) -> {"ids": [...], "rows": [...], "diff": {...}}
```

每个实验目录：`request.json result.json validation.json data.csv figures/*.png report.html meta.json`。
`meta.json`：`{"id","created_at","model","params","parent_id","relation","duration_ms","solver","question"}`。
`relation` ∈ `null` | `"rerun"` | `"derived"` | `"sweep_child"`。

## 7. report

```python
report.bundle(req, result, validation, figures, meta) -> dict   # 报告数据
report.write_json(path, bundle_or_payload)
report.write_csv(path, result)          # 时间序列/扫描数据
report.write_html(path, bundle)         # 独立 HTML，base64 内嵌 PNG
```

## 8. cli

```python
cli.main(argv=None) -> int   # 0 成功；1 执行失败；2 参数/请求非法
```

子命令：`run / sweep / list / show / rerun / derive / compare / export / selfcheck`。
所有子命令支持 `--json` 输出机器可读摘要。安全边界：无 eval、无任意代码执行，
模型/参数白名单，规模上限（见 spec）。

退出码语义：`0` = 完成且所执行检查全部通过（含全 skipped）；`1` = 执行失败**或**
完成但 `overall=partial`（两者都靠摘要的 `overall`/`failed` 区分）；`2` = 请求/参数非法
（stderr 单行消息，无 traceback）。
