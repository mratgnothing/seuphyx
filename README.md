# SEUPhyX

SEUPhyX 是东南大学物理实验中心的物理实验数据处理 Python 包。目前仓库中的主要应用是「密立根油滴实验」人工智能辅助数据处理系统，基于 Streamlit 提供网页交互界面，用于视觉/手动测量、机器学习 Q 聚类、符号回归和实验报告生成。

## Codex 快速阅读入口

后续 Codex 处理本仓库时，先读根目录的 `CODEX_CONTEXT.md`，再读当前 README。`CODEX_CONTEXT.md` 记录了当前代码里的数据处理、机器学习 Q 聚类、符号回归、旧版物理约束拟合以及报告生成的数据流。

当前主流程以 `src/seuphyx/core/oil/tabs/tab_classify.py` 中的 `charge_clustering()` 和 `src/seuphyx/core/oil/tabs/tab_regress.py` 调用的 `discovery_regression()` 为准。它不是先依赖传统分类标签再拟合，也不是先假设元电荷或整数倍关系，而是先由实验读数 `t,U` 计算连续电荷估计 `q`，再在 `q` 分布上做无监督聚类，聚类后用半峰宽筛选高置信点，最后对多个 `q` 簇的 `U-t` 曲线学习共享符号表达式。

传统机器学习分类和旧版整数 `n` 物理约束拟合路径仍保留在代码中用于兼容和阅读，但已经从当前 Streamlit 教学主流程中删除；当前 UI 只使用 AI 聚类和符号回归逻辑。

## 项目定位

本系统面向密立根油滴实验教学流程，目标是把实验数据从录入、可视化、分类、拟合到报告导出串成一个可运行的工作流。学生登录后，系统会按日期、学号和姓名创建独立工作目录，保存本次实验产生的数据、模型、图像和 PDF 报告。

## 当前实现的实验流程

1. 学生输入学号和姓名完成登记。
2. 在「数据记录」页录入下落时间和平衡电压数据。
3. 在「数据分类」页由 `U` 和 `t` 换算每个油滴的连续电荷估计 `q`，先显示未着色的 Q 分布，再点击执行 AI 聚类。
4. 使用 K-Means、Gaussian Mixture、DBSCAN 或 KDE 峰发现等无监督方法自动发现 `q` 的簇结构；半峰宽容差只在聚类完成后用于标记舍去点。
5. 在「机器学习—符号回归」页确认高可信点、选择三类符号模型策略，并在完整处理过程之后查看最终 `U_i(t)` 公式。
6. 在「打印报告」页生成实验报告。
7. 生成包含学生信息、q 簇发现、共享拟合结果的 PDF 实验报告。

## 当前聚类、筛选与拟合逻辑

当前教学目标是让学生在不知道电荷非连续性和元电荷数值的前提下，观察 AI 如何从复杂实验数据中发现规律。系统因此把主流程拆成两步：先无监督聚类，再符号回归。

### Q 计算

- 输入数据只有学生测得的下落时间 `t` 和平衡电压 `U`。
- `add_charge_estimates()` 根据实验参数计算油滴速度、半径和连续电荷估计。
- 前端统一用 `ChargeEstimate(1e-19C)` 作为横轴单位，即 `Q / 1e-19 C`。
- 这一步只做连续物理量换算，不使用元电荷数值，也不假设 `Q = n e`。

### 机器学习聚类

- 主要入口是 `charge_clustering(data, config)`。
- 默认方法为 K-Means，界面还提供 Gaussian Mixture、DBSCAN 和 KDE 峰发现。
- K-Means 与 Gaussian Mixture 可设置预期簇数；DBSCAN 可设置 `eps` 和 `min_samples`。
- 聚类结果会写出 `ChargeCluster`、`NearestChargeCluster`、`ChargeCenter(1e-19C)`、`ChargeDistance(1e-19C)`、`ChargeHalfWidth(1e-19C)`、`UseForFit` 和 `ClusterQuality`。

### 半峰宽过滤

- 半峰宽过滤发生在聚类之后，不在聚类前删点。
- 默认半峰宽容差是 `0.25 x 10^-19 C`。
- 若点满足 `abs(Q - cluster_center) <= half_width`，标记为 `UseForFit=True`。
- 超出半峰宽的点会保留在表格和图中，但标记为 `half_width_outlier`，不参与后续符号回归。
- 符号回归页还允许取消某些聚类结果，取消的簇会标记为 `cluster_disabled`。

### 符号回归

- 主入口是 `discovery_regression(data, config)`。
- 它先调用 `charge_clustering()`，再仅使用 `UseForFit=True` 的高置信点。
- 当前包含三类符号发现路径：
  - 全局候选式搜索：直接在所有高置信点上比较候选表达式。
  - 两阶段联合回归：先为多条 q 簇曲线共同搜索时间幂指数，再搜索曲线系数与簇中心 `Q_c` 的关系。
  - 神经网络 teacher 蒸馏：先用 MLP 集成学习平滑 `U=f(t,Q_c)` 曲面，再蒸馏为可解释符号表达式。
- 结果优先使用神经网络 teacher 蒸馏；若不可用，则回退到两阶段联合回归，再回退到全局候选式搜索。
- 当前符号回归结果版本为 `q-ai-clustering-symbolic-v8`。

### 后验物理解释

- 系统会把发现的 Q 簇中心间距作为后验结果，与公认元电荷做误差对照。
- 这个对照只用于解释发现结果，不参与聚类和拟合。
- 传统整数 `N` 反推与迭代物理拟合保留在 `physics_guided_regression()` 中，只作为对照检验方案。

## 已实现功能

### 1. Streamlit Web 应用

- 提供密立根油滴实验专用网页界面。
- 页面包含东南大学与物理实验中心标识。
- 使用宽屏布局和侧边栏提示当前学生、数据保存目录和联系信息。
- 通过 `seuphyx -n oil` 命令启动应用。
- 默认使用 Streamlit 端口 `8080`，并设置基础路径 `/seuphyx/oil`。

### 2. 学生登录与实验目录管理

- 启动后弹出学生信息登记对话框。
- 要求学号长度为 9 位。
- 要求姓名为中文或英文字符。
- 登录成功后在当前运行目录下创建：

```text
YYYY.MM.DD/oil_drop_<学号>_<姓名>/
```

- 本次实验数据、模型和报告均保存到该目录。

### 3. 实验数据记录

- 支持一次输入 5 组油滴数据。
- 每组数据包含：
  - 下落时间 `FallingTime(t/s)`
  - 平衡电压 `BalanceVoltage(U/V)`
- 自动过滤非正数输入。
- 自动检查重复点，避免重复写入。
- 首次保存时创建 `oil_drop.csv`。
- 后续数据追加写入 `oil_drop.csv`。
- 页面和侧边栏同步展示已保存数据。

### 4. 数据可视化

- 使用 Plotly 绘制散点图。
- 数据记录页只显示学生实测数据，红色高亮，横轴为 `Q / 1e-19 C`，不显示参考数据。
- 数据分类页将参考数据作为灰色背景点叠加，学生数据仍用红色高亮。
- 数据分类页先显示未经 AI 着色的原始 Q 分布，再显示聚类后的簇颜色、半峰宽区间和舍去点。
- 符号回归页展示多簇回归曲线、候选公式、共享符号表达式和传统物理拟合对照。

### 5. 参考数据与预训练模型

项目内置密立根油滴参考数据和预训练分类模型：

- `src/seuphyx/data/oil_drop_reference.csv`
- `src/seuphyx/data/points_svm_pipeline.joblib`

预训练模型用于：

- 给参考数据生成分类标签。
- 在「传统方法对照」页作为默认分类模型。
- 为「训练模型（选做）」页提供带标签训练数据。

### 6. 个性化机器学习模型训练（选做）

「训练模型（选做）」页支持学生基于参考数据训练自己的模型。

已支持的数据预处理：

- 标准化 `StandardScaler`
- 归一化 `MinMaxScaler`
- 鲁棒缩放 `RobustScaler`
- 可多选并按流水线组合

已支持的分类模型：

- 支持向量机 `SVC`
- 随机森林 `RandomForestClassifier`
- 梯度提升 `GradientBoostingClassifier`
- K 近邻 `KNeighborsClassifier`
- 朴素贝叶斯 `GaussianNB`
- 决策树 `DecisionTreeClassifier`

已支持的聚类模型：

- K-Means
- DBSCAN
- 层次聚类 `AgglomerativeClustering`

训练结果展示：

- 分类模型：训练集准确率、测试集准确率、交叉验证均值、混淆矩阵。
- 聚类模型：轮廓系数、Calinski-Harabasz 指数、Davies-Bouldin 指数。
- 分类或聚类结果散点图。
- 模型配置、预处理方法和参数。

模型保存：

- 训练完成后可保存为 `.joblib` 文件。
- 文件保存到学生本次实验工作目录。
- 传统方法对照页会自动扫描并加载该目录下的 `.joblib` 模型，供后续分类使用。

### 7. 传统分类（对照流程）

- 默认加载内置预训练 SVM 管线模型。
- 自动加载学生工作目录下保存的自定义 `.joblib` 模型。
- 对已录入实验点进行分类预测。
- 生成包含预测标签的 `data_pred` 数据表。
- 支持按类别分组展示实验数据。
- 可叠加参考数据的分类结果作为对照。
- 当前 UI 将传统分类放在「传统方法对照」页。它不再是「数据分类」或「机器学习—符号回归」的必要前置步骤。
- `Predicted` 是模型预测标签，不是人工真值；当前主拟合流程会重新从 `q` 分布发现 `ChargeCluster`。

### 8. AI 聚类与符号回归

- 将下落时间 `t` 和平衡电压 `U` 换算为电荷估计 `q`。
- 在 `q` 分布上进行无监督聚类，不预设元电荷大小或整数电荷数。
- 默认使用 K-Means，并提供 Gaussian Mixture、DBSCAN 和 KDE 峰发现选项。
- 聚类完成后使用半峰宽筛选高置信数据点，保留低置信点作为可视化对照。
- 对发现的多个 q 簇曲线进行共享符号表达式搜索，并优先用神经网络 teacher 学习平滑 `U=f(t,Q_c)` 曲面，再把 teacher 蒸馏为可解释符号表达式。
- 当前结果版本为 `q-ai-clustering-symbolic-v8`。若 Streamlit 会话里保存了旧版本结果，页面会要求重新执行拟合。
- 将发现的峰中心间距作为后验结果，再与公认元电荷进行误差对照。

### 9. 旧版物理约束聚类与拟合

旧版 `physics_guided_regression()` 仍保留在 `src/seuphyx/core/oil/tabs/regression.py` 中，作为兼容和对照逻辑。它与当前 UI 主流程不同：旧流程可以使用传统分类结果初始化参数，再围绕整数电荷数 `n` 做物理约束拟合。

当前实现包括两部分：

- 将机器学习分类结果作为可选初始化，估计全局参数 `A` 和 `b`。
- 根据理论关系把每个点转换为整数电荷数估计：

```text
n_float = A / ((U - b) * t^(3/2))
```

- 在 `n = 1, 2, 3, ...` 附近寻找峰值，只保留整数峰附近且残差合理的高置信数据点。
- 使用受约束统一结构进行拟合：

```text
U_n(t) = a_n * t^(-3/2) + b_n,  a_n ~= A / n
```

拟合实现细节：

- 默认处理 `n = 1` 到 `5`，页面中可调整最大 `n`。
- 使用整数峰半宽控制哪些点进入拟合。
- 使用 `scipy.optimize.least_squares` 和稳健损失进行全局拟合与逐峰校正拟合。
- 计算并保存 MSE、MAE、R2 等评估指标。
- 生成每一类的拟合曲线和拟合表达式。

该旧流程将聚类从普通 `(t, U)` 几何分类改为围绕整数 `n` 的物理聚类。未落在整数峰附近或残差过大的点会被标记为异常点，不参与最终拟合。当前「机器学习—符号回归」页仅把这个函数作为可展开的传统对照检验；阅读时不要把旧版 `PhysicsN` 路径误认为当前主路径。

### 10. 视觉自动测量预留

系统已新增「视觉测量」页和后端接口骨架，用于后续接入油滴视频自动测量。

当前已预留：

- 视频上传入口。
- 网格间距、帧率、电压、测量距离等标定参数。
- 点击跟踪、结束记录按钮位置。
- 后端 `VisionMeasurementConfig`、`TrackedOilDropMeasurement` 和 `OilDropVisionPipeline` 数据结构。

实际油滴检测、网格自动标定、点击跟踪和速度计算将在采集视频数据集后继续开发。

### 11. 实验报告生成

「打印报告」页可生成 PDF 报告。

报告内容包括：

- 报告标题。
- 学生姓名。
- 学生学号。
- 实验日期。
- 机器学习发现得到的共享表达式。
- 各 `q` 簇的拟合公式和拟合质量。
- 符号回归拟合图像。
- 核心数据规模、q 峰摘要和最终结论。

报告输出：

- PDF 文件保存到学生工作目录。
- 页面提供 PDF 下载按钮。
- 回归图像保存为 `regression_plot.png`。若 Plotly/Kaleido 图像导出不可用，
  报告会自动使用内置简化图，不向学生展示底层环境错误。

### 12. Python 包与命令行入口

项目采用 `pyproject.toml` 管理，包名为 `SEUPhyX`。

命令行入口：

```bash
seuphyx -n oil
```

该入口会定位 `seuphyx.core.oil.app.py` 并通过 Streamlit 启动密立根油滴实验页面。

## 安装与运行

建议使用 Python 3.10 或以上版本。

```bash
pip install -e .
```

启动密立根油滴实验系统：

```bash
seuphyx -n oil
```

启动后访问：

```text
http://localhost:8080/seuphyx/oil
```

## 公网部署

推荐使用 Streamlit Community Cloud 免费部署本系统。

部署参数：

```text
Repository: mratgnothing/seuphyx
Branch: main
Main file path: streamlit_app.py
```

仓库根目录的 `streamlit_app.py` 是云端入口文件，会自动加载
`src/seuphyx/core/oil/app.py`。云端依赖使用 `requirements.txt`，已避开当前
油滴实验主流程不需要的历史重依赖。

注意：免费云平台的本地文件通常不适合长期保存学生报告和模型文件。学生生成 PDF
后应立即下载；正式教学使用建议后续接入对象存储、数据库或校内服务器。

## 主要目录结构

```text
streamlit_app.py             # 云端部署入口
requirements.txt             # 云端轻量依赖
src/seuphyx/
  cli/
    main.py                 # 命令行入口
  core/
    oil/
      app.py                # 密立根油滴 Streamlit 主应用
      utils.py              # 绘图工具
      tabs/
        tab_record.py       # 数据记录
        tab_train.py        # 个性化模型训练
        tab_classify.py     # 机器学习 Q 聚类 + 传统分类对照函数
        tab_regress.py      # 机器学习—符号回归与传统物理拟合对照
        tab_vision.py       # 视觉测量预留页面
        tab_report.py       # PDF 报告生成
        regression.py       # charge_clustering + discovery_regression + physics_guided_regression
      vision.py             # 视觉测量后端接口骨架
  data/
    oil_drop_reference.csv  # 参考数据
    points_svm_pipeline.joblib
                            # 预训练 SVM 分类模型
  images/
    seu_logo.*              # 页面品牌图像资源
```

## 当前注意事项

- 当前仓库中实现的核心实验应用是 `oil`，即密立根油滴实验。
- 当前主拟合流程不再依赖无约束 PySR；`pyproject.toml` 中仍保留历史依赖，后续可按需要清理。
- PDF 报告优先使用 Plotly/Kaleido 导出回归图；若本机组件不可用，会退回内置简化图。
- 报告生成依赖已经完成机器学习—符号回归；报告页只展示核心结论、q 峰摘要、最终公式和图表，不输出逐点明细表。
- 训练页保存的聚类模型也会以 `.joblib` 形式出现在传统分类模型列表中，但传统分类函数会调用 `predict`，因此实际用于传统分类时更适合保存分类模型。
