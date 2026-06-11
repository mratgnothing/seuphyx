# SEUPhyX Codex Context

This file is the fast entry point for future Codex runs in `D:\GitHub\seuphyx`.
It summarizes the current Millikan oil-drop data processing pipeline as
implemented in code, especially AI charge clustering, symbolic regression,
legacy physics-guided fitting, session-state data flow, and report generation.

## Current Main Flow

The active app is the Millikan oil-drop Streamlit workflow.

- Cloud/local entrypoint: `streamlit_app.py`
- App shell: `src/seuphyx/core/oil/app.py`
- Data recording tab: `src/seuphyx/core/oil/tabs/tab_record.py`
- AI charge clustering tab: `src/seuphyx/core/oil/tabs/tab_classify.py`
- Symbolic regression tab: `src/seuphyx/core/oil/tabs/tab_regress.py`
- Algorithm implementation: `src/seuphyx/core/oil/tabs/regression.py`
- Report generation: `src/seuphyx/core/oil/tabs/tab_report.py`

The teaching-first logic is:

1. Record measured `FallingTime(t/s)` and `BalanceVoltage(U/V)`.
2. Convert each oil-drop reading to a continuous charge estimate `Q`, displayed
   as `ChargeEstimate(1e-19C)`.
3. Use unsupervised AI clustering on the one-dimensional `Q` distribution.
4. Apply the half-width tolerance after clustering to mark high-confidence
   points and excluded/outlier points.
5. Run multi-cluster symbolic regression on the selected high-confidence
   clusters to discover an interpretable shared law.
6. Keep traditional integer-`N` physics-guided fitting only as a comparison
   path.

Do not describe the active teaching flow as "known elementary charge first" or
"integer multiple first". The app is designed to let students observe how AI can
discover charge grouping and a common curve law from complex experimental data.

## UI Flow

Current tabs in `app.py`:

1. `数据记录`
   - Students enter measured `t,U` values.
   - The page shows only student-owned red Q-U scatter points.
   - Reference data is intentionally hidden on this page so students do not see
     the expected distribution too early.

2. `数据分类`
   - This is now the main "机器学习—聚类分析" page.
   - It overlays reference data as dim gray background points and highlights
     student data in red.
   - It first shows the raw, uncolored Q distribution.
   - After the student clicks the AI clustering button, it colors discovered
     clusters and marks half-width outliers.
   - It writes:
     - `st.session_state.charge_clustering_result`
     - `st.session_state.data_discovery_clustered`

3. `机器学习—符号回归`
   - This is the main fitting page.
   - It calls `discovery_regression()` and writes:
     - `st.session_state.regression_results`
     - `st.session_state.data_discovery_clustered`
   - `RESULT_VERSION` in `tab_regress.py` must match
     `result["result_version"]` from `discovery_regression()`.
   - Current symbolic regression result version:
     `q-ai-clustering-symbolic-v8`

4. `传统方法对照`
   - Keeps the old supervised/semi-supervised `U-t` classifier and model
     training workflow.
   - Its `Predicted` labels are comparison outputs, not ground truth.

5. `视觉测量`

6. `打印报告`

## Session State Data Flow

1. `app.py` initializes `st.session_state.data` with:
   - `FallingTime(t/s)`
   - `BalanceVoltage(U/V)`
2. `tab_record.py` appends user-entered measurements into `oil_drop.csv` in the
   student's work directory.
3. `tab_classify.py` runs AI Q clustering and can populate:
   - `charge_clustering_result`
   - `data_discovery_clustered`
4. `tab_regress.py` builds analysis data from student data plus optional
   `oil_drop_reference.csv`, then calls `discovery_regression()`.
5. `tab_report.py` requires `regression_results`; if old `data_pred` is missing,
   it uses the discovered clustered data from `regression_results["clusters"]`.

Important fields:

- `ChargeEstimate(1e-19C)`: continuous charge estimate in units of `1e-19 C`.
- `ChargeCluster`: accepted cluster label after half-width filtering.
- `NearestChargeCluster`: nearest discovered cluster before half-width exclusion.
- `ChargeCenter(1e-19C)`: center of the nearest discovered Q cluster.
- `ChargeDistance(1e-19C)`: signed offset from the nearest cluster center.
- `ChargeHalfWidth(1e-19C)`: tolerance used after clustering.
- `UseForFit`: whether the point participates in symbolic regression.
- `ClusterQuality`: `fit`, `half_width_outlier`, or `cluster_disabled`.
- `Predicted`: old classifier pseudo-label from the traditional comparison path.
- `PhysicsN`: old integer-`N` label from `physics_guided_regression()`.

## AI Charge Clustering Logic

The reusable clustering entry point is:

```python
charge_clustering(data, config)
```

in `src/seuphyx/core/oil/tabs/regression.py`.

Execution order:

1. `_clean_data()`
   - Requires `FallingTime(t/s)` and `BalanceVoltage(U/V)`.
   - Coerces both to numeric.
   - Drops missing values and non-positive falling times.

2. `add_charge_estimates()`
   - Converts raw `t,U` readings into:
     - `Velocity(mm/s)`
     - `Radius(um)`
     - `ChargeEstimate(C)`
     - `ChargeEstimate(1e-19C)`
   - This step estimates continuous charge and does not assume quantization or a
     known elementary charge.

3. `_ml_charge_peaks()`
   - Selects an unsupervised method from `DiscoveryRegressionConfig`:
     - `KMeans`
     - `GaussianMixture`
     - `DBSCAN`
     - `KDE`
   - The UI defaults to `K-Means`.
   - K-Means and Gaussian Mixture use `requested_clusters` when provided.
   - DBSCAN uses `dbscan_eps` and `dbscan_min_samples`.
   - KDE uses density peak finding and remains available as an alternate
     discovery method.

4. `_fallback_charge_peaks()`
   - If the selected method cannot produce reliable clusters, the function falls
     back to a robust path.
   - For varied data, it can use one-dimensional KMeans and silhouette score.
   - For too few or degenerate points, it uses one robust median cluster.

5. `_assign_charge_clusters()`
   - Assigns each point to the nearest discovered Q center.
   - A point enters later fitting only when:

```text
abs(Q - nearest_center) <= half_width
```

   - The default half-width tolerance is `0.25` in units of `1e-19 C`.
   - This is a post-clustering filter, not a pre-clustering manual deletion.

6. `_apply_selected_clusters()`
   - Symbolic regression can optionally disable some discovered clusters.
   - Disabled clusters are marked `cluster_disabled` and excluded from fitting.

7. `_estimate_peak_stability()`
   - Bootstraps the Q distribution and estimates whether each peak is
     rediscovered.

8. `_estimate_common_spacing()`
   - Treats discovered cluster centers as posterior results.
   - Fits equal spacing between centers and compares the spacing with accepted
     elementary charge only as a post-analysis check.
   - The accepted elementary charge is not used to create clusters.

`charge_clustering()` returns `result_version = "q-ai-clustering-v8"` and
`mode = "charge_clustering"`.

## Symbolic Regression Logic

The main fitting entry point is:

```python
discovery_regression(data, config)
```

It calls `charge_clustering()` first, then fits only the high-confidence points
where `UseForFit == True`.

Symbolic search currently includes:

1. `_fit_shared_symbolic_model()`
   - Searches global candidate expressions directly on all high-confidence
     points.

2. `_fit_two_stage_symbolic_model()`
   - Stage 1: search a common time exponent across multiple Q clusters.
   - Stage 2: search how the fitted cluster coefficient depends on the cluster
     center `Q_c`.
   - This is the explicit multi-cluster / joint symbolic-regression path.

3. `_fit_neural_teacher_distillation()`
   - Trains an MLP ensemble teacher on high-confidence points.
   - Samples smooth teacher curves per discovered Q cluster.
   - Distills the teacher surface into a compact symbolic expression.

Final result selection:

1. Prefer neural teacher distillation when available.
2. Else use two-stage symbolic discovery.
3. Else use global candidate search.

The result includes:

- `mode = "discovery"`
- `result_version = "q-ai-clustering-symbolic-v8"`
- `global_params`
- `symbolic_expression`
- `candidate_models`
- `two_stage`
- `neural_teacher`
- `density`
- `spacing`
- `peaks`
- `clusters`
- `peak_summary`
- `evaluation`
- `data`

## Configuration Defaults To Know

`DiscoveryRegressionConfig` defaults:

- `fall_distance_mm = 1.45`
- `plate_distance_mm = 5.0`
- `charge_unit_c = 1e-19`
- `clustering_method = "KMeans"`
- `requested_clusters = 5`
- `dbscan_eps = 0.25`
- `dbscan_min_samples = 4`
- `max_clusters = 8`
- `min_points_per_cluster = 4`
- `half_width_1e19c = 0.25`
- `selected_clusters = None`
- `kde_bandwidth = 0.08`
- `peak_prominence = 0.02`
- `analysis_lower_percentile = 0.5`
- `analysis_upper_percentile = 85.0`
- `stability_bootstrap_samples = 16`
- `min_symbolic_points = 8`
- `symbolic_pareto_tolerance_percent = 1.5`
- `neural_teacher_models = 5`
- `time_power_min = -3.0`
- `time_power_max = -0.25`
- `time_power_step = 0.05`

The UI exposes clustering method, requested cluster count, half-width tolerance,
selected clusters, DBSCAN parameters, KDE parameters, minimum points per cluster,
analysis upper percentile, and symbolic time-power search step.

## Traditional Classification Path

Traditional ML classification is retained for comparison and teaching.

Relevant files:

- `src/seuphyx/core/oil/tabs/tab_train.py`
- `src/seuphyx/core/oil/tabs/tab_classify.py`
- `src/seuphyx/data/points_svm_pipeline.joblib`
- `src/seuphyx/data/oil_drop_reference.csv`

`render_traditional_classification()` loads the built-in SVM or saved `.joblib`
models and predicts labels for current `U-t` points. It writes:

- `st.session_state.data_pred`
- `st.session_state.model`

Do not treat `Predicted` as ground truth. It is a model output from the
comparison workflow.

## Legacy Physics-Guided Regression

`physics_guided_regression()` remains in `regression.py` as a comparison path.

Its logic is different from the active AI path:

1. Optionally use `Predicted` labels to initialize global `A,b`.
2. Compute:
   `PhysicsNFloat = A / ((U - b) * t^(3/2))`
3. Round to integer `PhysicsN`.
4. Keep points close to integer peaks using `peak_width`.
5. Iteratively refit global `A,b`.
6. Remove residual outliers.
7. Fit each integer `n` curve:
   `U_n(t) = a_n * t^(-3/2) + b_n`, with `a_n ~= A / n`.

This path produces `PhysicsN`, `PhysicsNDistance`, and `PhysicsResidual(V)`.
Those fields are not the same as `ChargeCluster` and
`ChargeEstimate(1e-19C)`.

## Report Logic

`tab_report.py` checks for:

- saved experimental data
- `regression_results`
- existing report PDF

It can generate a PDF after AI symbolic regression is complete. The report
includes student info, total analyzed points, high-confidence points, discovered
common charge spacing when available, shared symbolic expression, Q-cluster
formulas, regression plot, and a saved point table with Q estimates and cluster
labels.

If Plotly/Kaleido image export fails, the PDF still keeps text and table
content.

## Common Pitfalls

- Do not say the active flow assumes charge quantization or known elementary
  charge before clustering.
- Do not describe old `Predicted` labels as true labels.
- Do not confuse `ChargeCluster` with `PhysicsN`.
- Do not apply half-width filtering before clustering; it is explicitly
  post-clustering filtering.
- Do not assume accepted elementary charge is used to create clusters. It is
  only used after discovery as an interpretive comparison.
- If Streamlit UI or result schema changes, bump `RESULT_VERSION` and returned
  `result_version` together, restart the active localhost Streamlit process on
  port `8501`, and verify `/_stcore/health`.
