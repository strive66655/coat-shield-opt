# coat-shield-opt

涂层屏蔽方案代理建模与 NSGA-II 多目标优化项目。

项目使用已有仿真/实验数据训练代理模型，预测不同涂层组合在 `1Mev`、`5Mev`、`10Mev` 三个能量点上的屏蔽提升率，并用 NSGA-II 搜索候选方案。

## 项目结构

```text
coat-shield-opt/
  configs/
    config.yaml                         # 训练配置
  data/
    processed/e_clean.csv               # 清洗后的原始数据
  src/
    train/train.py                      # 训练入口
    optimize_nsga2.py                   # 原始 NSGA-II 优化入口
    test.py                             # 按 ZnO + 能量点分组优化入口
    models/                             # 模型定义
    utils/                              # 数据、指标、配置工具
  experiments/
    exp_baseline_e/                     # 已训练代理模型与评估结果
    exp_nsga2_resmlp/                   # 原始整体 NSGA-II 优化结果
    exp_nsga2_by_zno/                   # 固定 ZnO 分组优化结果
```

## 环境

当前项目建议使用 conda 环境 `ML`：

```powershell
conda activate ML
```

或者不激活环境，直接用：

```powershell
conda run -n ML python ...
```

需要的主要依赖：

```text
pandas
numpy
PyYAML
scikit-learn
matplotlib
joblib
torch
xgboost
pymoo
```

如果缺少 `pymoo`，安装：

```powershell
conda run -n ML pip install pymoo
```

## 训练代理模型

训练配置在 [configs/config.yaml](configs/config.yaml)。

默认训练 ResidualMLP：

```powershell
conda run -n ML python src\train\train.py --config configs\config.yaml
```

训练完成后，主要输出在：

```text
experiments/exp_baseline_e/
  models/ResidualMLP_improvement_surrogate.pth
  surrogate_metrics.csv
  used_dataset.csv
  plots/
```

当前训练数据中，输入特征为：

```text
Al_thickness_um
B4C_thickness_um
W_thickness_um
ZnO_thickness_um
```

预测目标为：

```text
improvement_1Mev
improvement_5Mev
improvement_10Mev
```

## 原始整体 NSGA-II 优化

脚本：

```text
src/optimize_nsga2.py
```

运行：

```powershell
conda run -n ML python src\optimize_nsga2.py
```

默认输出：

```text
experiments/exp_nsga2_resmlp/
  nsga2_pareto_candidates.csv
  nsga2_feasible_candidates.csv
```

该脚本把 `B4C`、`W`、`ZnO` 同时作为优化变量，目标为：

```text
1. 最大化三个能量点中的最小提升率
2. 最小化新增涂层厚度 B4C + W + ZnO
```

## 按 ZnO 和能量点分组优化

脚本：

```text
src/test.py
```

这个脚本会对每个固定 `ZnO` 分组，以及每个指定能量点，分别做 NSGA-II 多目标优化。

每个分组的目标为：

```text
1. 最大化当前能量点的预测提升率
2. 最小化新增涂层厚度 B4C + W + ZnO
```

默认使用训练数据范围限制搜索空间，避免代理模型在远离训练数据的区域严重外推。默认训练域文件为：

```text
experiments/exp_baseline_e/used_dataset.csv
```

当前训练域范围大致为：

```text
ZnO=50 um   -> B4C: 1-436 um, W: 1.7-58.3 um
ZnO=100 um  -> B4C: 1-320 um, W: 2.3-43.8 um
ZnO=150 um  -> B4C: 1-204 um, W: 2.9-29.3 um
ZnO=200 um  -> B4C: 1-88 um,  W: 3.5-14.8 um
```

正式运行：

```powershell
conda run -n ML python src\test.py --zno_values "50,100,150,200" --energy_points "1,5,10" --pop_size 100 --n_gen 200
```

默认输出：

```text
experiments/exp_nsga2_by_zno_energy/
  all_zno_energy_pareto_candidates.csv
  all_zno_energy_feasible_candidates.csv
  recommended_candidate_by_group.csv
  ZnO_50um/
    energy_1Mev/
      nsga2_pareto_candidates.csv
      nsga2_feasible_candidates.csv
    energy_5Mev/
    energy_10Mev/
  ZnO_100um/
  ZnO_150um/
  ZnO_200um/
```

其中：

```text
all_zno_energy_pareto_candidates.csv
```

保存所有 ZnO/能量点分组的 Pareto 候选解。

```text
all_zno_energy_feasible_candidates.csv
```

保存满足约束的可行解。

```text
recommended_candidate_by_group.csv
```

每个 `ZnO + energy_target` 分组各选出一个推荐方案，便于快速比较。

## 常用参数

```powershell
conda run -n ML python src\test.py `
  --zno_values "50,100,150,200" `
  --energy_points "1,5,10" `
  --pop_size 100 `
  --n_gen 200 `
  --min_required_improvement 0.5 `
  --max_coating_thickness_um 500 `
  --domain_margin_um 0
```

参数说明：

```text
--zno_values
  固定 ZnO 分组，例如 "50,100,150,200"。

--energy_points
  分别优化的能量点，例如 "1,5,10"。

--pop_size
  NSGA-II 种群大小。

--n_gen
  NSGA-II 迭代代数。

--min_required_improvement
  当前能量点的最低提升率约束。

--max_coating_thickness_um
  B4C + W + ZnO 的厚度上限。

--training_domain_path
  用于限制搜索范围的训练数据文件。

--domain_margin_um
  在训练数据范围外额外放宽的厚度边界，默认 0。
```

快速测试一组：

```powershell
conda run -n ML python src\test.py --zno_values "50" --energy_points "1" --pop_size 10 --n_gen 2 --output_dir experiments\_smoke_nsga2_by_zno_energy
```

## 结果字段说明

优化结果 CSV 中常见字段：

```text
energy_target
  当前优化的能量点，例如 1Mev、5Mev、10Mev。

Al_thickness_um
B4C_thickness_um
W_thickness_um
ZnO_thickness_um
  涂层厚度方案。

coating_thickness_um
  B4C + W + ZnO。

pred_improvement_1Mev
pred_improvement_5Mev
pred_improvement_10Mev
  代理模型预测的三个能量点提升率。

selected_energy_improvement
  当前 energy_target 对应的提升率。

pred_min_improvement
  三个能量点中的最小预测提升率。

pred_mean_improvement
  三个能量点预测提升率均值。

is_feasible
  是否满足厚度约束和当前能量点提升率约束。
```

## 注意事项

代理模型只能可靠地用于接近训练数据分布的区域。不要把明显超过训练数据范围的优化结果直接当作物理最优方案。

如果把 `B4C` 或 `W` 的搜索范围放到远超训练数据，例如 `W=400+ um`，模型可能给出很高但不可信的预测值。正式推荐方案应优先查看 `recommended_candidate_by_group.csv`，并用真实仿真或实验复核。
