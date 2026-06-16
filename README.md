# apple-watch-health-ai

This project is a personal wearable health analytics prototype.
It is not a medical diagnostic system.

一个面向个人健康趋势分析与健康算法原型的 Python 项目。项目计划处理从
Apple Health 导出的数据，探索 wearable health sensing、睡眠分析、心率、
HRV、活动与每日恢复状态之间的关系。

本项目不用于医学诊断，不进行疾病预测，也不能替代专业医疗建议。

## 当前阶段

**Apple Health parsing, daily features, and personal recovery trends**

当前仓库已包含安全的流式 XML 解析入口，用于统计 Record 类型并提取指定指标。
同时可以将指定指标聚合为 daily-level dataframe，并计算透明、固定权重的个人恢复
趋势分数。该分数不是医学或诊断模型，仓库不包含真实或模拟的个人健康记录。

## 项目模块

1. **Apple Health XML parsing**：`src/parse_apple_health.py` 流式解析本地
   `export.xml`，提取指定健康指标且不打印原始记录。
2. **Daily health feature construction**：`src/build_daily_features.py` 构建睡眠、
   HRV、静息心率、心率、步数、活动能量和运动分钟的 daily-level 特征。
3. **Rule-based recovery score**：`src/recovery_score.py` 用透明固定权重构建个人
   recovery trend score，不作为医学或临床评分。
4. **Recovery trend visualization**：`src/visualize.py` 和
   `notebooks/03_recovery_analysis.ipynb` 检查覆盖率、趋势和组件 association。
5. **Date-filtered analysis: Jun–Dec 2025**：固定观察 2025-06-01 至
   2025-12-31，并在该时间窗口内展示 recovery trend 和数据覆盖。
6. **ML extension: next-day recovery prediction**：`src/ml_features.py`、
   `src/train_recovery_ml.py`、`src/plot_ml_results.py` 和
   `notebooks/04_ml_recovery_prediction.ipynb` 使用时间切分比较 ML、persistence
   和 7-day rolling mean baseline。
7. **Limitations and future work**：明确样本量、缺失数据、rule-based target、
   单人时间序列和测试集模型选择限制，并优先改善标签和数据质量。

## 计划处理的数据

后续将从 Apple Health 的 `export.xml` 中提取：

- Heart Rate
- Resting Heart Rate
- Heart Rate Variability (HRV)
- Sleep Analysis
- Steps
- Active Energy
- Exercise Minutes
- Workouts

这些记录将被聚合为 daily-level dataframe，用于个人健康趋势分析。

## 运行 XML 解析

先安装依赖：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

将 Apple Health 导出的 `export.xml` 保存在本地 `data/raw/` 后运行：

```bash
python src/parse_apple_health.py \
  --input data/raw/export.xml \
  --output data/processed
```

解析器采用流式读取，不会在终端打印个人健康记录，只输出聚合计数。生成文件：

- `record_type_summary.csv`
- `heart_rate.csv`
- `resting_heart_rate.csv`
- `hrv.csv`
- `sleep_analysis.csv`
- `steps.csv`
- `active_energy.csv`
- `exercise_time.csv`

日期字段会转换为带 UTC 时区的 datetime 后写入 CSV。所有生成结果仍属于敏感个人
健康数据，不应提交或上传。

## 构建每日健康特征

完成 XML 解析后运行：

```bash
python src/build_daily_features.py \
  --input-dir data/processed \
  --output data/processed/daily_health_features.csv
```

默认按系统本地时区划分每日边界。也可以显式指定 IANA 时区：

```bash
python src/build_daily_features.py \
  --input-dir data/processed \
  --output data/processed/daily_health_features.csv \
  --timezone Asia/Shanghai
```

输出 `daily_health_features.csv`，包含：

- `heart_rate_mean`、`heart_rate_min`、`heart_rate_max`
- `resting_heart_rate_mean`
- `hrv_mean`、`hrv_median`
- `steps_total`
- `active_energy_total`
- `exercise_minutes_total`
- `sleep_hours`

输出覆盖所有输入指标的连续日期范围，没有观测的特征保留为空值。睡眠只统计
Apple Health 的 `Asleep*` 类别，重叠区间会合并，并按睡眠结束时的本地日期归属。
命令行只输出每日行数和各特征缺失率，不打印个人健康数据明细。

## 计算个人恢复趋势分数

完成每日特征构建后运行：

```bash
python src/recovery_score.py \
  --input data/processed/daily_health_features.csv \
  --output data/processed/daily_recovery_score.csv \
  --figure-dir reports/figures
```

脚本对每个指标基于个人历史计算 z-score，并用固定权重生成 0–100 分：

- 睡眠时长：30%，较高值正向贡献
- HRV：30%，较高值正向贡献
- 静息心率：20%，较高值负向贡献
- 前一日运动分钟：10%，高于个人平均的负荷产生负向贡献
- 前一日活动能量：10%，高于个人平均的负荷产生负向贡献

组件 z-score 截断在 `[-3, 3]` 后通过 sigmoid 映射为 0–100 分。缺失指标不会被
填充，而是按当日可用权重重新归一化。可用权重低于 40%，或睡眠、HRV、静息心率
全部缺失时，`recovery_score` 保留为空。CSV 同时保存各组件 z-score、组件分数、
组件数量和权重覆盖率。

趋势图由 `src/visualize.py` 中的 `plot_recovery_trend(...)` 生成。脚本一次导出两种
布局，每种布局同时提供 240 DPI PNG 和 SVG：

- `reports/figures/recovery_score_trend_clean.png`
- `reports/figures/recovery_score_trend_clean.svg`
- `reports/figures/recovery_score_trend_portfolio.png`
- `reports/figures/recovery_score_trend_portfolio.svg`

clean 版面向正式分析报告：保留弱化的每日分数、突出 14 日滚动均值、长期均值参考
线和完整 summary box，但减少点位注释。portfolio 版在此基础上增加滚动高低点标注，
并用极浅的垂直区域提示长数据空档。

两版都会显示整体 recovery 水平、最新 score、最新 rolling mean、长期均值、最佳
和最低 rolling period。超过 21 天没有有效 score 的区段不会被连线跨越。low、
moderate、strong 区间仅用于视觉引导，不是临床阈值。

`notebooks/03_recovery_analysis.ipynb` 直接复用同一个绘图函数，可用于进一步检查
覆盖率、趋势和组件关系。

这是回顾性的个人趋势探索指标，不是医学诊断、疾病预测或专业医疗建议。

## 构建 ML recovery 数据集

完成每日健康特征和 recovery score 构建后运行：

```bash
python src/ml_features.py \
  --features data/processed/daily_health_features.csv \
  --scores data/processed/daily_recovery_score.csv \
  --output data/processed/ml_recovery_dataset.csv
```

脚本按 `date` 合并两个 daily-level CSV，只使用聚合数据，不读取
`data/raw/export.xml`，也不会在终端输出逐日健康记录。输出
`ml_recovery_dataset.csv`，包含：

- 下一日回归目标 `target_recovery_next_day`
- 下一日低恢复二分类目标 `low_recovery_next_day`，阈值为 40
- recovery、睡眠、HRV、静息心率、运动分钟、活动能量和步数的 lag1 特征
- 上述字段基于当天及过去数据的 3 日、7 日 rolling mean 和 rolling std
- HRV、静息心率和 recovery score 的 7 日 trend
- 核心健康字段的 missing indicator
- `day_of_week`、`is_weekend` 和 `month`

最后一个日期因没有下一日 target 会被删除。若其他日期的下一日
`recovery_score` 本身为空，对应两个 target 也保留为空，不会错误编码成正常恢复。
rolling 统计设置 `min_periods=1`，标准差使用总体标准差 `ddof=0`；窗口只包含当前
行和之前的行，不使用未来数据。本步骤只构建 ML 数据集，不训练模型，也不生成图表。

## 训练 next-day recovery 模型

完成 ML 数据集构建后运行：

```bash
python src/train_recovery_ml.py \
  --input data/processed/ml_recovery_dataset.csv \
  --reports-dir reports
```

脚本分别训练 next-day recovery score 回归模型和 low recovery 二分类模型。若数据
完整覆盖 2025-06-01 至 2025-12-31，则使用 2025-06-01 至 2025-10-31 训练，
使用 2025-11-01 至 2025-12-31 测试；否则按时间顺序将最后 20% 的 target-valid
日期作为测试集。不会使用随机切分，target 为空的行不参与训练或评估。

回归评估包括 persistence 和 7-day rolling mean 两个 baseline，以及 Ridge、
Random Forest 和 HistGradientBoosting。分类评估包括 Logistic Regression、
Random Forest 和 HistGradientBoosting。线性模型 pipeline 使用 median imputer、
standard scaler 和模型；树模型 pipeline 使用 median imputer 和模型。所有模型
固定 `random_state=42`。

输出：

- `reports/ml_regression_metrics.csv`
- `reports/ml_classification_metrics.csv`
- `reports/ml_predictions.csv`

回归最佳模型按测试集 MAE 选择，MAE 相同时依次比较 RMSE 和模型名。分类最佳模型按
测试集 F1 选择，F1 相同时依次比较 ROC-AUC、Accuracy 和模型名。若分类测试集只有
一个类别，ROC-AUC 输出为空值而不会中断运行。prediction 文件同时包含训练期和
测试期的最佳模型预测，使用 `split` 字段区分；训练期预测是 in-sample 结果，不应
当作泛化性能。baseline 缺少当天 recovery 值时不进行填补，回归指标文件会记录每种
方法的实际样本数，并额外提供两个 baseline 都可用时的 `common_*` 指标用于公平比较。
该步骤只输出聚合指标和预测 CSV，不读取 `data/raw/export.xml`，也不会在终端打印
逐日健康记录。

## ML visualization and interpretation

完成模型训练后运行：

```bash
python src/plot_ml_results.py \
  --predictions reports/ml_predictions.csv \
  --regression-metrics reports/ml_regression_metrics.csv \
  --classification-metrics reports/ml_classification_metrics.csv \
  --ml-dataset data/processed/ml_recovery_dataset.csv \
  --output-dir reports/figures
```

命令会输出 test set 的 predicted-vs-actual、时间线、残差、low recovery confusion
matrix 和最佳回归模型的 top-15 feature importance，同时保存 PNG 和 SVG。
`notebooks/04_ml_recovery_prediction.ipynb` 将数据概览、baseline、回归、分类、特征
重要性和限制组织为一份可重复运行的分析。

Predicted-vs-actual 图中，点越接近 `y=x` 参考线，预测误差越小；MAE 表示典型绝对
误差，RMSE 会更强调少数大误差。不能只根据图形或模型名称判断改进，必须在同一批
test 样本上比较 ML 与 persistence、7-day rolling mean baseline 的 MAE/RMSE。
由于 baseline 可能缺值，优先查看回归指标中的 `common_*` 列。只有在共同样本和后续
独立时间窗口中都持续优于 baseline，才支持存在稳定的增量 predictive signal。

这些图描述个人数据中的 association 和 personal recovery trend，只用于探索
next-day recovery prediction。结果受样本量、缺失、时间漂移和 rule-based target
定义限制，**not a clinical diagnosis**，不能用于临床决策。

## 项目结论

在共同 test 样本上，最佳回归模型超过 persistence baseline，但没有超过 7-day
rolling mean baseline；回归 R2 仍为负，分类测试集也只有少量 low-recovery 正例。
因此当前特征只显示有限 association，没有证明存在稳定超过简单 rolling trend 的
增量 predictive signal。该结果本身是有效的 baseline evaluation 结论，而不是失败。

## 简历表述

> Built a personal Apple Watch health analytics pipeline to parse Apple Health
> XML data, construct daily sleep/HRV/resting heart rate/activity features,
> design a rule-based recovery score, and evaluate next-day recovery prediction
> using time-based validation. Compared ML models against persistence and 7-day
> rolling baselines; found limited incremental predictive signal beyond rolling
> trends, with key associations from resting heart rate, exercise load, and HRV
> variability.

### 在 VS Code 中运行

1. 用 VS Code 打开项目根目录。
2. 通过 **Python: Select Interpreter** 选择 `.venv/bin/python`。
3. 打开 VS Code 集成终端；确认提示符中已激活 `.venv`，或运行
   `source .venv/bin/activate`。
4. 在项目根目录执行上述解析命令。

打开 notebook 时，同样选择 `.venv` 对应的 Python kernel。

## 数据隐私

- Apple Health 导出数据属于敏感个人数据。
- `data/raw/` 和 `data/processed/` 已加入 `.gitignore`。
- 不要提交、上传或公开分享 `export.xml`。
- 不要在 notebook、日志、截图或报告中泄露可识别的个人健康信息。
- 在共享分析结果前，应检查并移除时间、位置、设备和其他可能用于识别个人的信息。

## 项目结构

```text
apple-watch-health-ai/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
├── notebooks/
│   ├── 01_explore_export_xml.ipynb
│   ├── 02_daily_health_features.ipynb
│   ├── 03_recovery_analysis.ipynb
│   └── 04_ml_recovery_prediction.ipynb
├── src/
│   ├── __init__.py
│   ├── parse_apple_health.py
│   ├── build_daily_features.py
│   ├── recovery_score.py
│   ├── ml_features.py
│   ├── train_recovery_ml.py
│   ├── plot_ml_results.py
│   ├── visualize.py
│   └── utils.py
└── reports/
    └── figures/
```

## Limitations and future work

下一步不应优先更换模型，而应先改善标签和数据质量：

1. 将固定 `<40` 的 low recovery label 改为个人分位数定义，例如个人 recovery
   score 的 lowest 25%，并检查类别稳定性。
2. 加入 data coverage score，区分完整数据日和缺失数据日，并在训练、评估和解释中
   显式报告覆盖质量。
3. 增加 component-level attribution，解释每天较低的 recovery score 主要与 HRV、
   静息心率、睡眠还是运动负荷组件有关。

后续验证还需要更长的 target-valid 时间序列、独立且未参与模型选择的未来测试窗口，
以及对设备来源、缺失机制和时间漂移的系统检查。
