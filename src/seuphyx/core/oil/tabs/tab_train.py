"""
Tab: 机器学习模型训练（个性化）
允许用户自定义预处理方式、选择模型、训练并保存模型
"""
# built-in
from pathlib import Path
from datetime import datetime

# third-party
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import joblib

# sklearn 预处理
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# sklearn 分类模型
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

# sklearn 聚类模型
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

# sklearn 评估
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                             silhouette_score, calinski_harabasz_score,
                             davies_bouldin_score)

# sklearn 流水线
from sklearn.pipeline import make_pipeline

# seuphyx
from seuphyx.core.oil.utils import plotly_plot
import seuphyx


@st.cache_resource(show_spinner=False)
def _load_joblib_model(path: str):
    return joblib.load(path)


@st.cache_data(show_spinner=False)
def _load_reference_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def render_tab_train():
    st.header("机器学习模型训练")
    work_dir = st.session_state.work_dir

    # 查看参考数据集
    data_dir = Path(seuphyx.__file__).parent / "data"
    model_file = data_dir / "points_svm_pipeline.joblib"
    model = _load_joblib_model(str(model_file))
    reference_file = data_dir / "oil_drop_reference.csv"
    data_ref = _load_reference_data(str(reference_file))
    xy_coords = data_ref.values
    y_pred_ref = model.predict(xy_coords)
    data_ref_pred = pd.concat(
        [data_ref, pd.DataFrame({"Predicted": y_pred_ref})], axis=1)

    st.info("📊 本页面允许你自定义预处理方式、选择模型类型，训练并保存你的个性化模型。")

    # 数据预览
    with st.expander("查看数据集", expanded=False):
        st.dataframe(data_ref_pred, )
        st.markdown("""
            <style>
            [data-testid="stElementToolbar"] {
                display: none;
            }
            </style>
            """,
                    unsafe_allow_html=True)

    # ========== 数据预处理选择 ==========
    st.subheader("1️⃣ 数据预处理")
    
    with st.expander("💡 什么是数据预处理?", expanded=False):
        st.markdown("""
        数据预处理是机器学习流程中的重要步骤,用于将原始数据转换为更适合模型训练的格式。
        不同的预处理方法适用于不同的场景:
        
        - **标准化 (StandardScaler)**: 将数据转换为均值为0、标准差为1的分布。适用于大多数机器学习算法。
          [📖 文档链接](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
        
        - **归一化 (MinMaxScaler)**: 将数据缩放到 [0, 1] 区间。适用于需要固定范围的场景。
          [📖 文档链接](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html)
        
        - **鲁棒缩放 (RobustScaler)**: 使用中位数和四分位数进行缩放,对异常值更鲁棒。
          [📖 文档链接](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html)
        """)

    preprocessing_options = st.multiselect("选择预处理方法（可多选）",
                                           options=[
                                               "标准化 (StandardScaler)",
                                               "归一化 (MinMaxScaler)",
                                               "鲁棒缩放 (RobustScaler)",
                                           ],
                                           placeholder="若无选择则不进行预处理",
                                           default=[],
                                           help="不同的预处理方法会影响模型性能")
    
    # 显示选中的预处理方法说明
    if "标准化 (StandardScaler)" in preprocessing_options:
        st.info("✅ **标准化**: 将数据转换为均值0、标准差1,公式: $z = \\frac{x - \\mu}{\\sigma}$")
    if "归一化 (MinMaxScaler)" in preprocessing_options:
        st.info("✅ **归一化**: 将数据缩放到[0,1],公式: $x' = \\frac{x - x_{min}}{x_{max} - x_{min}}$")
    if "鲁棒缩放 (RobustScaler)" in preprocessing_options:
        st.info("✅ **鲁棒缩放**: 使用中位数和四分位距缩放,对异常值不敏感")

    # 预处理参数配置
    preprocessing_params = {}

    # ========== 模型类型选择 ==========
    st.subheader("2️⃣ 模型选择")

    model_type = st.radio(
        "选择任务类型",
        options=["分类模型", "聚类模型"],
        horizontal=True,
    )

    selected_model = None
    model_params = {}
    if model_type == "分类模型":
        with st.expander("💡 什么是分类模型?", expanded=False):
            st.markdown("""
            **分类模型 (Classification)** 是监督学习的一种,用于将数据点分配到预定义的类别中。
            
            - 需要**标注数据**(带有类别标签)进行训练
            - 适用场景: 邮件分类(垃圾/正常)、疾病诊断(患病/健康)、图像识别等
            - 评估指标: 准确率、精确率、召回率、F1分数等
            
            [📖 sklearn分类模型总览](https://scikit-learn.org/stable/supervised_learning.html#supervised-learning)
            """)
        
        st.info("⚠️ 分类模型需要标注数据，如果数据没有标签，请先手动标注。")
        model_choice = st.selectbox("选择分类模型",
                                    options=[
                                        "支持向量机 (SVM)",
                                        "随机森林 (Random Forest)",
                                        "梯度提升 (Gradient Boosting)",
                                        "K近邻 (KNN)",
                                        "朴素贝叶斯 (Naive Bayes)",
                                        "决策树 (Decision Tree)",
                                    ])

        # 模型参数配置
        st.write("⚙️ 模型参数配置")
        
        if model_choice == "支持向量机 (SVM)":
            with st.expander("📘 SVM 模型说明", expanded=False):
                st.markdown("""
                **支持向量机 (Support Vector Machine)** 通过寻找最优超平面来分隔不同类别。
                
                - **优点**: 在高维空间有效,内存效率高,versatile(可使用不同核函数)
                - **适用场景**: 文本分类、图像识别、生物信息学
                - **关键参数**:
                  - `C`: 正则化参数,值越大越不容易过拟合
                  - `kernel`: 核函数类型(rbf适合非线性,linear适合线性可分)
                  - `gamma`: 核函数系数,影响单个样本的影响范围
                
                [📖 sklearn.svm.SVC 文档](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
                """)
            
            col1, col2 = st.columns(2)
            with col1:
                C = st.slider("C (正则化参数)", 0.01, 10.0, 1.0, 0.1)
                kernel = st.selectbox("核函数", ["rbf", "linear", "poly"])
            with col2:
                gamma = st.selectbox("gamma", ["scale", "auto"])
            model_params = {"C": C, "kernel": kernel, "gamma": gamma}
            selected_model = SVC(**model_params)

        elif model_choice == "随机森林 (Random Forest)":
            with st.expander("📘 随机森林模型说明", expanded=False):
                st.markdown("""
                **随机森林 (Random Forest)** 是集成学习方法,通过构建多棵决策树并投票得出结果。
                
                - **优点**: 准确率高,抗过拟合能力强,可以处理高维数据
                - **适用场景**: 特征重要性分析、分类和回归任务
                - **关键参数**:
                  - `n_estimators`: 森林中树的数量,越多越好但计算成本增加
                  - `max_depth`: 树的最大深度,控制模型复杂度
                  - `min_samples_split`: 节点分裂所需的最小样本数
                
                [📖 sklearn.ensemble.RandomForestClassifier 文档](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
                """)
            
            col1, col2 = st.columns(2)
            with col1:
                n_estimators = st.slider("树的数量", 10, 200, 100, 10)
                max_depth = st.slider("最大深度", 1, 20, 10)
            with col2:
                min_samples_split = st.slider("最小分裂样本数", 2, 20, 2)
            model_params = {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "min_samples_split": min_samples_split,
                "criterion": "gini",  # Default value for criterion
                "bootstrap": True,  # Default value for bootstrap
                "oob_score": False,  # Default value for oob_score
                "warm_start": False,  # Default value for warm_start
                "class_weight": None,  # Default value for class_weight
                "random_state": 42
            }
            selected_model = RandomForestClassifier(**model_params)

        elif model_choice == "梯度提升 (Gradient Boosting)":
            with st.expander("📘 梯度提升模型说明", expanded=False):
                st.markdown("""
                **梯度提升 (Gradient Boosting)** 通过顺序构建弱学习器,每个新树都纠正前面树的错误。
                
                - **优点**: 预测精度高,可以处理各种类型的数据
                - **适用场景**: Kaggle竞赛、需要高精度的分类任务
                - **关键参数**:
                  - `n_estimators`: 迭代次数(树的数量)
                  - `learning_rate`: 学习率,控制每棵树的贡献
                  - `max_depth`: 单棵树的最大深度
                
                [📖 sklearn.ensemble.GradientBoostingClassifier 文档](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)
                """)
            
            col1, col2 = st.columns(2)
            with col1:
                n_estimators = st.slider("迭代次数", 10, 200, 100, 10)
                learning_rate = st.slider("学习率", 0.01, 1.0, 0.1, 0.01)
            with col2:
                max_depth = st.slider("最大深度", 1, 10, 3)
            model_params = {
                "n_estimators": n_estimators,
                "learning_rate": learning_rate,
                "max_depth": max_depth,
                "random_state": 42
            }
            selected_model = GradientBoostingClassifier(**model_params)

        elif model_choice == "K近邻 (KNN)":
            with st.expander("📘 K近邻模型说明", expanded=False):
                st.markdown("""
                **K近邻 (K-Nearest Neighbors)** 通过寻找最近的K个邻居进行投票分类。
                
                - **优点**: 简单直观,无需训练过程,适合多分类问题
                - **适用场景**: 推荐系统、模式识别、异常检测
                - **关键参数**:
                  - `n_neighbors`: K值,邻居数量
                  - `weights`: 权重函数(uniform统一权重,distance距离加权)
                  - `metric`: 距离度量方式
                
                [📖 sklearn.neighbors.KNeighborsClassifier 文档](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
                """)
            
            col1, col2 = st.columns(2)
            with col1:
                n_neighbors = st.slider("邻居数量 (K)", 1, 20, 5)
                weights = st.selectbox("权重", ["uniform", "distance"])
            with col2:
                metric = st.selectbox("距离度量",
                                      ["euclidean", "manhattan", "minkowski"])
            model_params = {
                "n_neighbors": n_neighbors,
                "weights": weights,
                "metric": metric
            }
            selected_model = KNeighborsClassifier(**model_params)

        elif model_choice == "朴素贝叶斯 (Naive Bayes)":
            with st.expander("📘 朴素贝叶斯模型说明", expanded=False):
                st.markdown("""
                **朴素贝叶斯 (Naive Bayes)** 基于贝叶斯定理和特征独立假设的概率分类器。
                
                - **优点**: 训练速度快,对小规模数据表现好,适合多分类
                - **适用场景**: 文本分类、垃圾邮件过滤、情感分析
                - **特点**: 假设特征之间相互独立
                
                [📖 sklearn.naive_bayes.GaussianNB 文档](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html)
                """)
            st.info("✅ 朴素贝叶斯模型使用默认参数")
            selected_model = GaussianNB()

        elif model_choice == "决策树 (Decision Tree)":
            with st.expander("📘 决策树模型说明", expanded=False):
                st.markdown("""
                **决策树 (Decision Tree)** 通过树状结构进行决策,每个节点代表一个特征判断。
                
                - **优点**: 易于理解和解释,可视化直观,不需要数据归一化
                - **适用场景**: 规则提取、特征选择、可解释性要求高的场景
                - **关键参数**:
                  - `max_depth`: 树的最大深度,防止过拟合
                  - `min_samples_split`: 节点分裂所需的最小样本数
                  - `criterion`: 分裂标准(gini基尼系数,entropy信息熵)
                
                [📖 sklearn.tree.DecisionTreeClassifier 文档](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
                """)
            
            col1, col2 = st.columns(2)
            with col1:
                max_depth = st.slider("最大深度", 1, 20, 10)
                min_samples_split = st.slider("最小分裂样本数", 2, 20, 2)
            with col2:
                criterion = st.selectbox("分裂标准", ["gini", "entropy"])
            model_params = {
                "max_depth": max_depth,
                "min_samples_split": min_samples_split,
                "criterion": criterion,
                "random_state": 42
            }
            selected_model = DecisionTreeClassifier(**model_params)

    else:  # 聚类模型
        with st.expander("💡 什么是聚类模型?", expanded=False):
            st.markdown("""
            **聚类模型 (Clustering)** 是无监督学习的一种,用于发现数据中的自然分组。
            
            - 不需要**标注数据**,自动发现数据模式
            - 适用场景: 客户细分、图像分割、异常检测、数据探索
            - 评估指标: 轮廓系数、Calinski-Harabasz指数、Davies-Bouldin指数
            
            [📖 sklearn聚类模型总览](https://scikit-learn.org/stable/modules/clustering.html)
            """)
        
        st.info("✅ 聚类模型不需要标签,自动发现数据中的模式。")
        model_choice = st.selectbox("选择聚类模型",
                                    options=[
                                        "K-Means",
                                        "DBSCAN",
                                        "层次聚类 (Agglomerative)",
                                    ])

        st.write("⚙️ 模型参数配置")

        if model_choice == "K-Means":
            with st.expander("📘 K-Means 模型说明", expanded=False):
                st.markdown("""
                **K-Means** 将数据划分为K个簇,每个点属于距离最近的聚类中心。
                
                - **优点**: 简单快速,适合大规模数据,可扩展性好
                - **适用场景**: 客户细分、图像压缩、文档聚类
                - **关键参数**:
                  - `n_clusters`: 聚类数量K,需要预先指定
                  - `max_iter`: 最大迭代次数
                  - `init`: 初始化方法(k-means++更智能)
                
                [📖 sklearn.cluster.KMeans 文档](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
                """)
            
            col1, col2 = st.columns(2)
            with col1:
                n_clusters = st.slider("聚类数量", 2, 10, 3)
                max_iter = st.slider("最大迭代次数", 100, 1000, 300, 100)
            with col2:
                init = st.selectbox("初始化方法", ["k-means++", "random"])
            model_params = {
                "n_clusters": n_clusters,
                "max_iter": max_iter,
                "init": init,
                "random_state": 42
            }
            selected_model = KMeans(**model_params)

        elif model_choice == "DBSCAN":
            with st.expander("📘 DBSCAN 模型说明", expanded=False):
                st.markdown("""
                **DBSCAN (Density-Based Spatial Clustering)** 基于密度的聚类算法。
                
                - **优点**: 不需要预先指定聚类数,可以发现任意形状的簇,能识别噪声点
                - **适用场景**: 地理数据聚类、异常检测、不规则形状簇
                - **关键参数**:
                  - `eps`: 邻域半径,定义"邻近"的距离
                  - `min_samples`: 核心点所需的最小邻居数
                  - `metric`: 距离度量方式
                
                [📖 sklearn.cluster.DBSCAN 文档](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)
                """)
            
            col1, col2 = st.columns(2)
            with col1:
                eps = st.slider("邻域半径 (eps)", 0.1, 10.0, 0.5, 0.1)
                min_samples = st.slider("最小样本数", 1, 20, 5)
            with col2:
                metric = st.selectbox("距离度量",
                                      ["euclidean", "manhattan", "cosine"])
            model_params = {
                "eps": eps,
                "min_samples": min_samples,
                "metric": metric
            }
            selected_model = DBSCAN(**model_params)

        elif model_choice == "层次聚类 (Agglomerative)":
            with st.expander("📘 层次聚类模型说明", expanded=False):
                st.markdown("""
                **层次聚类 (Agglomerative Clustering)** 通过自底向上的方式构建聚类树。
                
                - **优点**: 不需要预先指定簇数,可以生成聚类树状图(dendrogram)
                - **适用场景**: 生物信息学、社交网络分析、文档层次分类
                - **关键参数**:
                  - `n_clusters`: 聚类数量
                  - `linkage`: 链接方式(ward最小化方差,complete最大距离,average平均距离)
                  - `metric`: 距离度量(ward只能用euclidean)
                
                [📖 sklearn.cluster.AgglomerativeClustering 文档](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html)
                """)
            
            col1, col2 = st.columns(2)
            with col1:
                n_clusters = st.slider("聚类数量", 2, 10, 3)
                linkage = st.selectbox(
                    "链接方法", ["ward", "complete", "average", "single"])
            with col2:
                metric = st.selectbox("距离度量",
                                      ["euclidean", "manhattan", "cosine"])
            model_params = {
                "n_clusters": n_clusters,
                "linkage": linkage,
            }
            if linkage != "ward":
                model_params["metric"] = metric
            selected_model = AgglomerativeClustering(**model_params)

    # ========== 开始训练 ==========
    st.subheader("3️⃣ 模型训练")

    if st.button("开始训练个人的模型", type="primary", use_container_width=True):
        with st.spinner("训练中，请稍候..."):
            try:
                # 准备数据
                data_ref = data_ref_pred[[
                    'FallingTime(t/s)', 'BalanceVoltage(U/V)'
                ]].values
                data_ref = data_ref.copy()

                # 数据预处理
                scalers = []

                if "标准化 (StandardScaler)" in preprocessing_options:
                    scalers.append(('StandardScaler', StandardScaler()))

                if "归一化 (MinMaxScaler)" in preprocessing_options:
                    scalers.append(('MinMaxScaler', MinMaxScaler()))

                if "鲁棒缩放 (RobustScaler)" in preprocessing_options:
                    scalers.append(('RobustScaler', RobustScaler()))

                # 训练模型
                if model_type == "分类模型":
                    labels = data_ref_pred['Predicted'].values

                    # 划分训练集和测试集
                    X_train, X_test, y_train, y_test = train_test_split(
                        data_ref, labels, test_size=0.2)

                    # 训练
                    clf = make_pipeline(*[scaler for _, scaler in scalers],
                                        selected_model)
                    clf.fit(X_train, y_train)

                    # 预测
                    y_pred_train = clf.predict(X_train)
                    y_pred_test = clf.predict(X_test)
                    y_pred_all = clf.predict(data_ref)

                    # 评估
                    train_score = clf.score(X_train, y_train)
                    test_score = clf.score(X_test, y_test)

                    # 交叉验证
                    cv_scores = cross_val_score(
                        clf,
                        data_ref,
                        labels,  # type: ignore
                        cv=5,
                    )

                    # 保存结果到 session_state
                    st.session_state.trained_model = {
                        'model':
                        selected_model,
                        'model_type':
                        model_type,
                        'model_name':
                        model_choice,
                        'scalers':
                        scalers,
                        'X_processed':
                        data_ref,
                        'y_true':
                        labels,
                        'y_pred':
                        y_pred_all,
                        'train_score':
                        train_score,
                        'test_score':
                        test_score,
                        'cv_scores':
                        cv_scores,
                        'classification_report':
                        classification_report(y_test, y_pred_test),
                        'confusion_matrix':
                        confusion_matrix(y_test, y_pred_test),
                        'preprocessing_options':
                        preprocessing_options,
                        'model_params':
                        model_params,
                    }

                    st.success("✅ 分类模型训练完成！")

                else:  # 聚类模型
                    # 训练
                    clf = make_pipeline(*[scaler for _, scaler in scalers],
                                        selected_model)
                    y_pred = clf.fit_predict(data_ref)

                    # 评估
                    silhouette = silhouette_score(data_ref, y_pred)
                    calinski = calinski_harabasz_score(data_ref, y_pred)
                    davies = davies_bouldin_score(data_ref, y_pred)

                    # 保存结果
                    st.session_state.trained_model = {
                        'model': selected_model,
                        'model_type': model_type,
                        'model_name': model_choice,
                        'scalers': scalers,
                        'X_processed': data_ref,
                        'y_pred': y_pred,
                        'silhouette_score': silhouette,
                        'calinski_score': calinski,
                        'davies_bouldin_score': davies,
                        'preprocessing_options': preprocessing_options,
                        'model_params': model_params,
                    }

                    st.success("✅ 聚类模型训练完成！")

                st.session_state.clf = clf

            except Exception as e:
                st.error(f"❌ 训练失败: {e}")
                st.exception(e)

    # ========== 显示训练结果 ==========
    if 'trained_model' in st.session_state:
        st.subheader("4️⃣ 训练结果")

        result = st.session_state.trained_model

        # 显示配置信息
        with st.expander("📝 模型配置信息", expanded=False):
            st.write(f"**模型类型**: {result['model_type']}")
            st.write(f"**模型名称**: {result['model_name']}")
            ppopt = result['preprocessing_options']
            ppopt = ', '.join(ppopt) if ppopt else '无'
            st.write(f"**预处理方法**: {ppopt}")
            st.write(f"**模型参数**: {result['model_params']}")

        # 性能指标
        st.write("### 📊 性能指标")

        if result['model_type'] == "分类模型":
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("训练集准确率", f"{result['train_score']*100:.1f}%")
            with col2:
                st.metric("测试集准确率", f"{result['test_score']*100:.1f}%")
            with col3:
                st.metric("交叉验证均值", f"{result['cv_scores'].mean()*100:.1f}%")

            # 混淆矩阵
            st.subheader("混淆矩阵")
            fig_cm = go.Figure(
                data=go.Heatmap(z=result['confusion_matrix'],
                                colorscale='Blues',
                                text=result['confusion_matrix'],
                                texttemplate="%{text}",
                                textfont={"size": 18}),
                layout=go.Layout(
                    title="混淆矩阵",
                    xaxis=dict(title="预测类别"),
                    yaxis=dict(title="真实类别"),
                    font=dict(family='DejaVu Serif', size=16),
                    margin=dict(l=60, r=30, t=30, b=60),
                    colorway=px.colors.qualitative.D3,
                    height=400,
                ),
            )
            st.plotly_chart(fig_cm, use_container_width=True)

        else:  # 聚类模型
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("轮廓系数",
                          f"{result['silhouette_score']:.4f}",
                          help="范围[-1,1]，越接近1越好")
            with col2:
                st.metric("Calinski-Harabasz指数",
                          f"{result['calinski_score']:.2f}",
                          help="值越大越好")
            with col3:
                st.metric("Davies-Bouldin指数",
                          f"{result['davies_bouldin_score']:.4f}",
                          help="值越小越好")

        # 可视化结果
        st.write("### 📈 结果可视化")

        # 准备绘图数据
        X_plot = result['X_processed']
        y_plot = result['y_pred']

        # 如果维度大于2，只取前两维
        if X_plot.shape[1] > 2:
            X_plot = X_plot[:, :2]
            st.info("注意：数据维度>2，仅显示前两个维度")

        # 创建散点图
        unique_labels = np.unique(y_plot)
        fig = go.Figure()

        for label in unique_labels:
            mask = y_plot == label
            fig.add_trace(
                go.Scatter(x=X_plot[mask, 0],
                           y=X_plot[mask, 1],
                           mode='markers',
                           name=f'类别 {label}',
                           marker=dict(size=10,
                                       line=dict(width=1, color='white'))))

        fig.update_layout(
            title=f"{result['model_name']} - 聚类/分类结果",
            xaxis_title="特征1"
            if "PCA降维" in result['preprocessing_options'] else "下落时间 (t/s)",
            yaxis_title="特征2"
            if "PCA降维" in result['preprocessing_options'] else "平衡电压 (U/V)",
            font=dict(size=14),
            height=500,
            margin=dict(l=60, r=30, t=60, b=60),
            colorway=px.colors.qualitative.D3,
        )

        st.plotly_chart(fig, use_container_width=True)

        # ========== 保存模型 ==========
        st.subheader("5️⃣ 保存模型")

        # 获取已有模型数量
        existing_models = list(
            work_dir.glob(f"{result['model_name'].replace(' ', '_')}*.joblib"))
        next_number = len(existing_models) + 1

        default_filename = f"{result['model_name'].replace(' ', '_').replace('(', '').replace(')', '')}_{next_number:03d}.joblib"

        col1, col2 = st.columns([5, 1])
        with col1:
            model_filename = st.text_input("模型文件名",
                                           value=default_filename,
                                           help="模型将保存为 .joblib 格式")

        with col2:
            st.write("")
            st.write("")
            if st.button("保存模型", type="primary"):
                try:
                    joblib.dump(st.session_state.clf,
                                work_dir / model_filename)
                except Exception as e:
                    st.error(f"保存失败: {e}")

        # 显示已保存的模型
        with st.expander("📂 查看已保存的模型"):
            saved_models = sorted(work_dir.glob("*.joblib"))
            if saved_models:
                st.write(f"共 {len(saved_models)} 个已保存模型：")
                for model_path in saved_models:
                    st.write(f"- {model_path.name}")
            else:
                st.write("暂无已保存的模型")
