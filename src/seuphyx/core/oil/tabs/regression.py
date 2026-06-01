"""
Physics-guided fitting for the Millikan oil-drop experiment.

The main regression path is intentionally constrained by the experiment model:

    U_n(t) = a_n * t^(-3/2) + b,    a_n ~= A / n,    n = 1, 2, 3, ...

This avoids fitting visually plausible curves that extrapolate incorrectly
outside the measured voltage/time window.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import sympy as sp
from scipy.optimize import least_squares
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


TIME_COL = "FallingTime(t/s)"
VOLTAGE_COL = "BalanceVoltage(U/V)"
PREDICTED_COL = "Predicted"


@dataclass
class PhysicsRegressionConfig:
    """Configuration for integer-n clustering and constrained fitting."""

    max_n: int = 5
    peak_width: float = 0.25
    min_points_per_peak: int = 3
    initial_a: float = 54402.3027
    initial_b: float = -7.5
    use_predicted_labels_for_init: bool = True
    discard_predicted_label: int | None = 6
    max_iterations: int = 8
    residual_sigma_factor: float = 3.0


def _clean_data(data: pd.DataFrame) -> pd.DataFrame:
    required = [TIME_COL, VOLTAGE_COL]
    missing = [col for col in required if col not in data.columns]
    if missing:
        raise ValueError(f"缺少必要数据列: {', '.join(missing)}")

    cleaned = data.copy()
    cleaned[TIME_COL] = pd.to_numeric(cleaned[TIME_COL], errors="coerce")
    cleaned[VOLTAGE_COL] = pd.to_numeric(cleaned[VOLTAGE_COL], errors="coerce")
    cleaned = cleaned.dropna(subset=required)
    cleaned = cleaned[cleaned[TIME_COL] > 0].reset_index(drop=True)
    if cleaned.empty:
        raise ValueError("没有可用于拟合的正下落时间数据点。")
    return cleaned


def _x_from_t(t_values: np.ndarray) -> np.ndarray:
    return np.asarray(t_values, dtype=float) ** (-1.5)


def _robust_line_fit(x_values: np.ndarray,
                     y_values: np.ndarray,
                     initial: tuple[float, float]) -> tuple[float, float]:
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)

    def residual(theta):
        a_value, b_value = theta
        return a_value * x_values + b_value - y_values

    result = least_squares(
        residual,
        x0=np.asarray(initial, dtype=float),
        loss="soft_l1",
        f_scale=5.0,
        max_nfev=10000,
    )
    return float(result.x[0]), float(result.x[1])


def _numeric_label(value) -> int | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    rounded = round(numeric)
    if abs(numeric - rounded) > 1e-6:
        return None
    return int(rounded)


def _estimate_initial_parameters(data: pd.DataFrame,
                                 config: PhysicsRegressionConfig
                                 ) -> tuple[float, float, list[dict]]:
    if not config.use_predicted_labels_for_init or PREDICTED_COL not in data:
        return config.initial_a, config.initial_b, []

    x_values = _x_from_t(data[TIME_COL].to_numpy(float))
    grouped_estimates = []

    for label in sorted(data[PREDICTED_COL].dropna().unique()):
        n_value = _numeric_label(label)
        if n_value is None or n_value < 1 or n_value > config.max_n:
            continue
        if (config.discard_predicted_label is not None
                and n_value == config.discard_predicted_label):
            continue

        mask = data[PREDICTED_COL].map(_numeric_label) == n_value
        sub = data.loc[mask]
        if len(sub) < 2:
            continue

        x_sub = x_values[mask.to_numpy()]
        y_sub = sub[VOLTAGE_COL].to_numpy(float)
        a_init = config.initial_a / n_value
        a_value, b_value = _robust_line_fit(x_sub, y_sub,
                                            (a_init, config.initial_b))
        if a_value > 0 and np.isfinite(a_value) and np.isfinite(b_value):
            grouped_estimates.append({
                "n": n_value,
                "a": a_value,
                "b": b_value,
                "A_from_n": a_value * n_value,
                "points": int(len(sub)),
            })

    if not grouped_estimates:
        return config.initial_a, config.initial_b, []

    a_global = float(np.median([item["A_from_n"] for item in grouped_estimates]))
    b_global = float(np.median([item["b"] for item in grouped_estimates]))
    return a_global, b_global, grouped_estimates


def _assign_integer_peaks(data: pd.DataFrame, a_global: float, b_global: float,
                          max_n: int) -> pd.DataFrame:
    result = data.copy()
    t_values = result[TIME_COL].to_numpy(float)
    u_values = result[VOLTAGE_COL].to_numpy(float)

    a_point = (u_values - b_global) * np.power(t_values, 1.5)
    n_float = np.divide(
        a_global,
        a_point,
        out=np.full_like(a_point, np.nan, dtype=float),
        where=a_point > 0,
    )
    n_nearest = np.rint(n_float)
    valid = np.isfinite(n_float) & (n_nearest >= 1) & (n_nearest <= max_n)

    result["PhysicsNFloat"] = n_float
    result["PhysicsN"] = np.where(valid, n_nearest, np.nan)
    result["PhysicsNDistance"] = np.where(valid, np.abs(n_float - n_nearest),
                                          np.nan)
    return result


def _fit_global_physics(data: pd.DataFrame,
                        assigned_n: np.ndarray,
                        use_mask: np.ndarray,
                        initial: tuple[float, float]) -> tuple[float, float]:
    x_values = _x_from_t(data.loc[use_mask, TIME_COL].to_numpy(float))
    y_values = data.loc[use_mask, VOLTAGE_COL].to_numpy(float)
    n_values = assigned_n[use_mask].astype(float)

    def residual(theta):
        a_global, b_global = theta
        return a_global / n_values * x_values + b_global - y_values

    result = least_squares(
        residual,
        x0=np.asarray(initial, dtype=float),
        bounds=([1e-9, -200.0], [np.inf, 200.0]),
        loss="soft_l1",
        f_scale=5.0,
        max_nfev=20000,
    )
    return float(result.x[0]), float(result.x[1])


def _robust_residual_limit(residuals: np.ndarray,
                           sigma_factor: float) -> tuple[float, float]:
    residuals = np.asarray(residuals, dtype=float)
    finite = residuals[np.isfinite(residuals)]
    if finite.size == 0:
        return 15.0, 0.0
    median = np.median(finite)
    mad = np.median(np.abs(finite - median))
    sigma = 1.4826 * mad
    return max(15.0, sigma_factor * sigma), float(sigma)


def _fit_peak_curve(sub: pd.DataFrame,
                    n_value: int,
                    a_global: float,
                    b_global: float) -> tuple[float, float]:
    x_values = _x_from_t(sub[TIME_COL].to_numpy(float))
    y_values = sub[VOLTAGE_COL].to_numpy(float)
    a_prior = a_global / n_value

    def residual(theta):
        a_value, b_value = theta
        data_residual = a_value * x_values + b_value - y_values
        prior_residual = np.array([
            (a_value - a_prior) / max(abs(a_prior) * 0.08, 1.0),
            (b_value - b_global) / 8.0,
        ])
        return np.concatenate([data_residual, prior_residual])

    result = least_squares(
        residual,
        x0=np.asarray([a_prior, b_global], dtype=float),
        bounds=([1e-9, -200.0], [np.inf, 200.0]),
        loss="soft_l1",
        f_scale=5.0,
        max_nfev=10000,
    )
    return float(result.x[0]), float(result.x[1])


def physics_guided_regression(
        data: pd.DataFrame,
        config: PhysicsRegressionConfig | None = None) -> dict:
    """Cluster data by integer charge number and fit physical oil-drop curves."""
    config = config or PhysicsRegressionConfig()
    cleaned = _clean_data(data)

    a_global, b_global, label_initial_estimates = _estimate_initial_parameters(
        cleaned, config)

    clustered = _assign_integer_peaks(cleaned, a_global, b_global,
                                      config.max_n)
    for _ in range(config.max_iterations):
        n_values = clustered["PhysicsN"].to_numpy(float)
        distance = clustered["PhysicsNDistance"].to_numpy(float)
        use_mask = np.isfinite(n_values) & (distance <= config.peak_width)

        if use_mask.sum() < max(2, config.min_points_per_peak):
            break

        next_a, next_b = _fit_global_physics(clustered, n_values, use_mask,
                                            (a_global, b_global))
        if np.isclose(next_a, a_global, rtol=1e-5, atol=1e-5) and np.isclose(
                next_b, b_global, rtol=1e-5, atol=1e-5):
            a_global, b_global = next_a, next_b
            break

        a_global, b_global = next_a, next_b
        clustered = _assign_integer_peaks(cleaned, a_global, b_global,
                                          config.max_n)

    n_values = clustered["PhysicsN"].to_numpy(float)
    x_values = _x_from_t(clustered[TIME_COL].to_numpy(float))
    y_values = clustered[VOLTAGE_COL].to_numpy(float)
    global_prediction = np.divide(
        a_global,
        n_values,
        out=np.full_like(n_values, np.nan, dtype=float),
        where=np.isfinite(n_values),
    ) * x_values + b_global
    residuals = global_prediction - y_values
    distance = clustered["PhysicsNDistance"].to_numpy(float)
    preliminary_mask = np.isfinite(n_values) & (distance <= config.peak_width)
    residual_limit, robust_sigma = _robust_residual_limit(
        residuals[preliminary_mask], config.residual_sigma_factor)
    use_mask = preliminary_mask & (np.abs(residuals) <= residual_limit)

    clustered["PhysicsResidual(V)"] = residuals
    clustered["UseForFit"] = use_mask
    clustered["ClusterQuality"] = np.where(
        use_mask, "fit",
        np.where(preliminary_mask, "residual_outlier", "integer_outlier"))

    t_symbol = sp.Symbol("t", real=True, positive=True)
    data_by_n = {}
    evaluation = {}
    peak_summary = []

    for n_value in range(1, config.max_n + 1):
        mask = (clustered["PhysicsN"] == n_value) & clustered["UseForFit"]
        sub = clustered.loc[mask].sort_values(TIME_COL)
        if len(sub) < config.min_points_per_peak:
            continue

        a_value, b_value = _fit_peak_curve(sub, n_value, a_global, b_global)
        tt = sub[TIME_COL].to_numpy(float)
        yy = sub[VOLTAGE_COL].to_numpy(float)
        y_pred = a_value * _x_from_t(tt) + b_value

        mse = mean_squared_error(yy, y_pred)
        mae = mean_absolute_error(yy, y_pred)
        r2 = r2_score(yy, y_pred) if len(sub) >= 2 else np.nan

        t_min = float(tt.min())
        t_max = float(tt.max())
        if np.isclose(t_min, t_max):
            t_min = max(t_min * 0.95, 1e-9)
            t_max = t_max * 1.05
        t_line = np.linspace(t_min, t_max, 200)
        y_line = a_value * _x_from_t(t_line) + b_value

        a_rounded = round(a_value, 4)
        b_rounded = round(b_value, 4)
        fitted_expr = a_rounded * t_symbol**(-sp.Rational(3, 2)) + b_rounded

        data_by_n[n_value] = [t_line, y_line, fitted_expr]
        evaluation[n_value] = {
            "params": {
                "a": a_rounded,
                "b": b_rounded,
                "A_over_n": round(a_global / n_value, 4),
            },
            "a_times_n": round(a_value * n_value, 4),
            "mse": float(mse),
            "mae": float(mae),
            "r2": float(r2) if np.isfinite(r2) else np.nan,
            "points": int(len(sub)),
        }
        peak_summary.append({
            "n": n_value,
            "points": int(len(sub)),
            "a": a_rounded,
            "b": b_rounded,
            "a*n": round(a_value * n_value, 4),
            "mse": round(float(mse), 4),
            "mae": round(float(mae), 4),
            "r2": round(float(r2), 4) if np.isfinite(r2) else np.nan,
        })

    return {
        "regression_form": "U_n(t) = a_n * t^(-3/2) + b_n, a_n ~= A / n",
        "global_params": {
            "A": round(a_global, 4),
            "b": round(b_global, 4),
            "max_n": int(config.max_n),
            "peak_width": float(config.peak_width),
            "residual_limit": round(float(residual_limit), 4),
            "robust_sigma": round(float(robust_sigma), 4),
        },
        "label_initial_estimates": label_initial_estimates,
        "clusters": clustered,
        "peak_summary": pd.DataFrame(peak_summary),
        "evaluation": evaluation,
        "data": data_by_n,
    }


def symbolic_regression_model(data_pred):
    """Backward-compatible entry point for older callers."""
    return physics_guided_regression(data_pred)
