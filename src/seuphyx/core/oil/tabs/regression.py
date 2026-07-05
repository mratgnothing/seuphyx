"""
Discovery and fitting tools for the Millikan oil-drop experiment.

The teaching-first path does not assume charge quantization or a known
elementary charge. It converts measured U and t into per-drop charge estimates,
discovers natural peaks in the charge distribution, and then searches for a
shared symbolic expression across the discovered curves.

The older physics-guided integer-n fitter is kept below as a comparison path.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import sympy as sp
from scipy.optimize import least_squares
from scipy.signal import find_peaks, peak_widths
from scipy.stats import gaussian_kde
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import silhouette_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


TIME_COL = "FallingTime(t/s)"
VOLTAGE_COL = "BalanceVoltage(U/V)"
PREDICTED_COL = "Predicted"
SOURCE_COL = "Source"
VELOCITY_COL = "Velocity(mm/s)"
RADIUS_COL = "Radius(um)"
CHARGE_COL = "ChargeEstimate(C)"
CHARGE_UNIT_COL = "ChargeEstimate(1e-19C)"
CHARGE_CLUSTER_COL = "ChargeCluster"
NEAREST_CHARGE_CLUSTER_COL = "NearestChargeCluster"
CHARGE_CENTER_COL = "ChargeCenter(1e-19C)"
CHARGE_DISTANCE_COL = "ChargeDistance(1e-19C)"
CHARGE_HALF_WIDTH_COL = "ChargeHalfWidth(1e-19C)"
DISCOVERY_QUALITY_COL = "ClusterQuality"
USE_FOR_FIT_COL = "UseForFit"
ELEMENTARY_CHARGE_C = 1.602176634e-19


@dataclass
class DiscoveryRegressionConfig:
    """Configuration for prior-free charge discovery and symbolic fitting."""

    fall_distance_mm: float = 1.45
    plate_distance_mm: float = 5.0
    oil_density_kg_m3: float = 981.0
    air_density_kg_m3: float = 1.29
    viscosity_pa_s: float = 1.83e-5
    gravity_m_s2: float = 9.80665
    charge_unit_c: float = 1e-19
    kde_bandwidth: float = 0.08
    peak_prominence: float = 0.02
    clustering_method: str = "KMeans"
    requested_clusters: int | None = 5
    dbscan_eps: float = 0.25
    dbscan_min_samples: int = 4
    max_clusters: int = 8
    min_points_per_cluster: int = 4
    half_width_1e19c: float | None = 0.25
    half_width_scale: float = 1.0
    selected_clusters: tuple[int, ...] | None = None
    analysis_lower_percentile: float = 0.5
    analysis_upper_percentile: float = 85.0
    density_grid_size: int = 2500
    stability_bootstrap_samples: int = 16
    min_symbolic_points: int = 8
    symbolic_pareto_tolerance_percent: float = 1.5
    symbolic_model_strategy: str = "auto"
    neural_teacher_models: int = 5
    time_power_min: float = -3.0
    time_power_max: float = -0.25
    time_power_step: float = 0.05


def _charge_density_delta(config: DiscoveryRegressionConfig) -> float:
    return max(config.oil_density_kg_m3 - config.air_density_kg_m3, 1e-9)


def add_charge_estimates(
        data: pd.DataFrame,
        config: DiscoveryRegressionConfig | None = None) -> pd.DataFrame:
    """Convert measured U and t into charge estimates without quantization."""
    config = config or DiscoveryRegressionConfig()
    result = _clean_data(data)

    fall_distance_m = config.fall_distance_mm * 1e-3
    plate_distance_m = config.plate_distance_mm * 1e-3
    density_delta = _charge_density_delta(config)

    t_values = result[TIME_COL].to_numpy(float)
    u_values = result[VOLTAGE_COL].to_numpy(float)
    velocity_m_s = fall_distance_m / t_values
    radius_m = np.sqrt(
        9.0 * config.viscosity_pa_s * velocity_m_s /
        (2.0 * density_delta * config.gravity_m_s2))
    effective_weight_n = (
        4.0 * np.pi / 3.0 * radius_m**3 * density_delta *
        config.gravity_m_s2)
    charge_c = effective_weight_n * plate_distance_m / u_values

    result[VELOCITY_COL] = velocity_m_s * 1000.0
    result[RADIUS_COL] = radius_m * 1e6
    result[CHARGE_COL] = charge_c
    result[CHARGE_UNIT_COL] = charge_c / config.charge_unit_c
    return result


def _finite_charge_values(data: pd.DataFrame) -> np.ndarray:
    q_values = data[CHARGE_UNIT_COL].to_numpy(float)
    return q_values[np.isfinite(q_values) & (q_values > 0)]


def _interpolate_grid(grid: np.ndarray, positions: np.ndarray) -> np.ndarray:
    return np.interp(positions, np.arange(len(grid)), grid)


def _peak_half_widths(centers: np.ndarray, fwhm: np.ndarray,
                      config: DiscoveryRegressionConfig) -> np.ndarray:
    if config.half_width_1e19c is not None and config.half_width_1e19c > 0:
        return np.full(len(centers), float(config.half_width_1e19c))

    half_widths = []
    for idx, center in enumerate(centers):
        gaps = []
        if idx > 0:
            gaps.append(center - centers[idx - 1])
        if idx < len(centers) - 1:
            gaps.append(centers[idx + 1] - center)
        gap_cap = 0.45 * min(gaps) if gaps else fwhm[idx] / 2.0
        half_width = min(fwhm[idx] / 2.0, gap_cap)
        if not np.isfinite(half_width) or half_width <= 0:
            half_width = gap_cap if gap_cap > 0 else max(center * 0.05, 0.05)
        half_widths.append(half_width * config.half_width_scale)
    return np.asarray(half_widths, dtype=float)


def _kde_charge_peaks(q_values: np.ndarray,
                      config: DiscoveryRegressionConfig) -> tuple[list[dict],
                                                                  dict] | None:
    if len(q_values) < max(5, config.min_points_per_cluster * 2):
        return None
    if len(np.unique(np.round(q_values, 8))) < 2:
        return None

    lower = min(max(config.analysis_lower_percentile, 0.0), 40.0)
    upper = min(max(config.analysis_upper_percentile, lower + 5.0), 99.5)
    q_min, q_max = np.percentile(q_values, [lower, upper])
    if not np.isfinite(q_min) or not np.isfinite(q_max) or q_max <= q_min:
        q_min, q_max = float(np.min(q_values)), float(np.max(q_values))
    span = q_max - q_min
    if span <= 0:
        return None

    pad = span * 0.08
    grid = np.linspace(max(0.0, q_min - pad), q_max + pad,
                       int(config.density_grid_size))
    kde_values = q_values[(q_values >= q_min) & (q_values <= q_max)]
    if len(kde_values) < max(5, config.min_points_per_cluster * 2):
        kde_values = q_values
    try:
        density = gaussian_kde(kde_values,
                               bw_method=config.kde_bandwidth)(grid)
    except Exception:
        return None

    prominence = max(float(np.max(density)) * config.peak_prominence, 1e-12)
    peaks, props = find_peaks(
        density,
        prominence=prominence,
        distance=max(3, len(grid) // 80),
    )
    if len(peaks) == 0:
        return None

    if len(peaks) > config.max_clusters:
        keep = np.argsort(props["prominences"])[::-1][:config.max_clusters]
        peaks = peaks[keep]
    peaks = np.sort(peaks)

    widths = peak_widths(density, peaks, rel_height=0.5)
    left_positions = _interpolate_grid(grid, widths[2])
    right_positions = _interpolate_grid(grid, widths[3])
    centers = grid[peaks]
    fwhm = right_positions - left_positions
    half_widths = _peak_half_widths(centers, fwhm, config)

    peak_rows = []
    for idx, peak_idx in enumerate(peaks, start=1):
        peak_rows.append({
            "cluster": idx,
            "center": float(centers[idx - 1]),
            "density": float(density[peak_idx]),
            "left_half": float(left_positions[idx - 1]),
            "right_half": float(right_positions[idx - 1]),
            "fwhm": float(fwhm[idx - 1]),
            "half_width": float(half_widths[idx - 1]),
            "method": "KDE",
        })

    density_info = {
        "grid": grid,
        "density": density,
        "peaks": np.asarray([row["center"] for row in peak_rows]),
        "method": "KDE",
        "analysis_min": float(q_min),
        "analysis_max": float(q_max),
        "analysis_points": int(len(kde_values)),
        "total_points": int(len(q_values)),
    }
    return peak_rows, density_info


def _analysis_charge_window(q_values: np.ndarray,
                            config: DiscoveryRegressionConfig
                            ) -> tuple[float, float]:
    lower = min(max(float(config.analysis_lower_percentile), 0.0), 40.0)
    upper = min(max(float(config.analysis_upper_percentile), lower + 5.0),
                99.5)
    q_min, q_max = np.percentile(q_values, [lower, upper])
    if not np.isfinite(q_min) or not np.isfinite(q_max) or q_max <= q_min:
        q_min, q_max = float(np.min(q_values)), float(np.max(q_values))
    return float(q_min), float(q_max)


def _analysis_charge_values(q_values: np.ndarray,
                            config: DiscoveryRegressionConfig) -> np.ndarray:
    q_min, q_max = _analysis_charge_window(q_values, config)
    core = q_values[(q_values >= q_min) & (q_values <= q_max)]
    if len(core) < max(5, config.min_points_per_cluster * 2):
        return q_values
    return core


def _fallback_charge_peaks(q_values: np.ndarray,
                           config: DiscoveryRegressionConfig) -> tuple[list[dict],
                                                                       dict]:
    max_k = min(config.max_clusters,
                max(1, len(q_values) // max(config.min_points_per_cluster, 1)))
    if max_k < 2 or len(np.unique(np.round(q_values, 8))) < 2:
        center = float(np.median(q_values))
        mad = np.median(np.abs(q_values - center))
        sigma = 1.4826 * mad
        if config.half_width_1e19c is not None and config.half_width_1e19c > 0:
            half_width = float(config.half_width_1e19c)
        else:
            half_width = max(1.1774 * sigma, center * 0.05, 0.05)
        return ([{
            "cluster": 1,
            "center": center,
            "density": np.nan,
            "left_half": center - half_width,
            "right_half": center + half_width,
            "fwhm": 2 * half_width,
            "half_width": half_width,
            "method": "robust-median",
        }], {
            "grid": np.array([]),
            "density": np.array([]),
            "peaks": np.array([center]),
            "method": "robust-median",
        })

    best_labels = None
    best_score = -np.inf
    for k_value in range(2, max_k + 1):
        model = KMeans(n_clusters=k_value, n_init=20, random_state=42)
        labels = model.fit_predict(q_values.reshape(-1, 1))
        if len(np.unique(labels)) < 2:
            continue
        try:
            score = silhouette_score(q_values.reshape(-1, 1), labels)
        except Exception:
            score = -np.inf
        if score > best_score:
            best_score = score
            best_labels = labels

    if best_labels is None:
        return _fallback_charge_peaks(q_values[:1], config)

    centers = []
    raw_half_widths = []
    for label in np.unique(best_labels):
        sub = q_values[best_labels == label]
        center = float(np.median(sub))
        mad = float(np.median(np.abs(sub - center)))
        sigma = 1.4826 * mad
        centers.append(center)
        raw_half_widths.append(max(1.1774 * sigma, 0.05))

    order = np.argsort(centers)
    centers = np.asarray(centers, dtype=float)[order]
    raw_half_widths = np.asarray(raw_half_widths, dtype=float)[order]
    half_widths = _peak_half_widths(centers, 2 * raw_half_widths, config)

    peak_rows = []
    for idx, center in enumerate(centers, start=1):
        half_width = float(half_widths[idx - 1])
        peak_rows.append({
            "cluster": idx,
            "center": float(center),
            "density": np.nan,
            "left_half": float(center - half_width),
            "right_half": float(center + half_width),
            "fwhm": float(2 * half_width),
            "half_width": half_width,
            "method": "KMeans",
        })
    return peak_rows, {
        "grid": np.array([]),
        "density": np.array([]),
        "peaks": centers,
        "method": "KMeans",
    }


def _cluster_count_from_config(q_values: np.ndarray,
                               config: DiscoveryRegressionConfig) -> int:
    max_k = min(
        int(config.max_clusters),
        max(1, len(q_values) // max(int(config.min_points_per_cluster), 1)),
    )
    if config.requested_clusters is not None:
        return int(np.clip(config.requested_clusters, 1, max_k))
    return max_k


def _best_kmeans_labels(q_values: np.ndarray,
                        max_k: int) -> tuple[np.ndarray | None, str]:
    if max_k < 2 or len(np.unique(np.round(q_values, 8))) < 2:
        return None, "KMeans"
    best_labels = None
    best_score = -np.inf
    for k_value in range(2, max_k + 1):
        model = KMeans(n_clusters=k_value, n_init=20, random_state=42)
        labels = model.fit_predict(q_values.reshape(-1, 1))
        if len(np.unique(labels)) < 2:
            continue
        try:
            score = silhouette_score(q_values.reshape(-1, 1), labels)
        except Exception:
            score = -np.inf
        if score > best_score:
            best_score = score
            best_labels = labels
    return best_labels, "KMeans-auto"


def _charge_peaks_from_labels(q_values: np.ndarray, labels: np.ndarray,
                              method: str,
                              config: DiscoveryRegressionConfig) -> tuple[list[dict],
                                                                          dict] | None:
    valid_labels = [label for label in np.unique(labels) if label != -1]
    if not valid_labels:
        return None

    centers = []
    raw_half_widths = []
    for label in valid_labels:
        sub = q_values[labels == label]
        if len(sub) < max(1, config.min_points_per_cluster):
            continue
        center = float(np.median(sub))
        mad = float(np.median(np.abs(sub - center)))
        sigma = 1.4826 * mad
        if not np.isfinite(sigma) or sigma <= 0:
            sigma = float(np.std(sub)) if len(sub) > 1 else 0.0
        centers.append(center)
        raw_half_widths.append(max(1.1774 * sigma, 0.05))

    if not centers:
        return None

    order = np.argsort(centers)
    centers = np.asarray(centers, dtype=float)[order]
    raw_half_widths = np.asarray(raw_half_widths, dtype=float)[order]
    half_widths = _peak_half_widths(centers, 2 * raw_half_widths, config)

    peak_rows = []
    for idx, center in enumerate(centers, start=1):
        half_width = float(half_widths[idx - 1])
        peak_rows.append({
            "cluster": idx,
            "center": float(center),
            "density": np.nan,
            "left_half": float(center - half_width),
            "right_half": float(center + half_width),
            "fwhm": float(2 * half_width),
            "half_width": half_width,
            "method": method,
        })
    return peak_rows, {
        "grid": np.array([]),
        "density": np.array([]),
        "peaks": centers,
        "method": method,
    }


def _ml_charge_peaks(q_values: np.ndarray,
                     config: DiscoveryRegressionConfig) -> tuple[list[dict],
                                                                 dict] | None:
    method = (config.clustering_method or "KMeans").strip().lower()
    if method in {"kde", "density", "峰发现kde"}:
        return _kde_charge_peaks(q_values, config)

    fit_values = _analysis_charge_values(q_values, config)
    if len(fit_values) < max(2, config.min_points_per_cluster):
        return None
    if len(np.unique(np.round(fit_values, 8))) < 2:
        return None

    q_matrix = fit_values.reshape(-1, 1)
    if method in {"k-means", "kmeans", "k means"}:
        k_value = _cluster_count_from_config(fit_values, config)
        if config.requested_clusters is None:
            labels, label_method = _best_kmeans_labels(fit_values, k_value)
        else:
            if k_value < 2:
                return None
            labels = KMeans(
                n_clusters=k_value,
                n_init=20,
                random_state=42,
            ).fit_predict(q_matrix)
            label_method = "KMeans"
        if labels is None:
            return None
        return _charge_peaks_from_labels(fit_values, labels, label_method,
                                         config)

    if method in {"gmm", "gaussian mixture", "gaussianmixture", "高斯混合"}:
        k_value = _cluster_count_from_config(fit_values, config)
        if k_value < 2:
            return None
        labels = GaussianMixture(
            n_components=k_value,
            random_state=42,
            covariance_type="full",
            n_init=5,
        ).fit_predict(q_matrix)
        return _charge_peaks_from_labels(fit_values, labels, "GaussianMixture",
                                         config)

    if method in {"dbscan", "density clustering"}:
        labels = DBSCAN(
            eps=max(float(config.dbscan_eps), 1e-6),
            min_samples=max(int(config.dbscan_min_samples), 1),
        ).fit_predict(q_matrix)
        return _charge_peaks_from_labels(fit_values, labels, "DBSCAN", config)

    return _kde_charge_peaks(q_values, config)


def _apply_selected_clusters(clustered: pd.DataFrame,
                             selected_clusters: tuple[int, ...] | None
                             ) -> pd.DataFrame:
    if not selected_clusters:
        return clustered
    selected = {int(value) for value in selected_clusters}
    result = clustered.copy()
    has_cluster = result[CHARGE_CLUSTER_COL].notna()
    disabled = pd.Series(False, index=result.index)
    disabled.loc[has_cluster] = ~(
        result.loc[has_cluster, CHARGE_CLUSTER_COL].astype(int).isin(selected)
    )
    result.loc[disabled, USE_FOR_FIT_COL] = False
    result.loc[disabled, DISCOVERY_QUALITY_COL] = "cluster_disabled"
    return result


def charge_clustering(
        data: pd.DataFrame,
        config: DiscoveryRegressionConfig | None = None) -> dict:
    """Run prior-free charge clustering without fitting a symbolic formula."""
    config = config or DiscoveryRegressionConfig()
    charged = add_charge_estimates(data, config)
    q_values = _finite_charge_values(charged)
    if len(q_values) < max(2, config.min_points_per_cluster):
        raise ValueError("可用于聚类的正电荷估计点太少。")

    peak_result = _ml_charge_peaks(q_values, config)
    if peak_result is None:
        peak_result = _fallback_charge_peaks(q_values, config)
    peaks, density_info = peak_result
    clustered, peaks = _assign_charge_clusters(charged, peaks)
    clustered = _apply_selected_clusters(clustered, config.selected_clusters)
    peaks = _estimate_peak_stability(q_values, peaks, config)
    spacing = _estimate_common_spacing(peaks)

    peak_summary = []
    for peak in peaks:
        cluster_id = int(peak["cluster"])
        sub = clustered[
            (clustered[CHARGE_CLUSTER_COL] == cluster_id) &
            clustered[USE_FOR_FIT_COL]]
        peak_summary.append({
            "cluster": cluster_id,
            "Q_center(1e-19C)": round(float(peak["center"]), 4),
            "half_width(1e-19C)": round(float(peak["half_width"]), 4),
            "stability(%)": round(float(peak.get("stability_percent")), 1)
            if np.isfinite(peak.get("stability_percent", np.nan)) else np.nan,
            "fit_points": int(len(sub)),
            "nearest_points": int(peak.get("all_nearest_points", 0)),
            "method": peak.get("method", ""),
        })

    return {
        "result_version": "q-ai-clustering-v9",
        "mode": "charge_clustering",
        "density": density_info,
        "spacing": spacing,
        "peaks": peaks,
        "clusters": clustered,
        "peak_summary": pd.DataFrame(peak_summary),
        "config": {
            "clustering_method": config.clustering_method,
            "requested_clusters": config.requested_clusters,
            "half_width_1e19c": config.half_width_1e19c,
            "dbscan_eps": config.dbscan_eps,
            "dbscan_min_samples": config.dbscan_min_samples,
        },
    }


def _assign_charge_clusters(data: pd.DataFrame,
                            peaks: list[dict]) -> tuple[pd.DataFrame,
                                                        list[dict]]:
    result = data.copy()
    centers = np.asarray([peak["center"] for peak in peaks], dtype=float)
    half_widths = np.asarray([peak["half_width"] for peak in peaks],
                             dtype=float)
    q_values = result[CHARGE_UNIT_COL].to_numpy(float)

    nearest_index = np.argmin(np.abs(q_values[:, None] - centers[None, :]),
                              axis=1)
    nearest_cluster = nearest_index + 1
    nearest_center = centers[nearest_index]
    nearest_half_width = half_widths[nearest_index]
    distance = q_values - nearest_center
    use_for_fit = (
        np.isfinite(q_values) & np.isfinite(nearest_center) &
        (np.abs(distance) <= nearest_half_width))

    result[NEAREST_CHARGE_CLUSTER_COL] = nearest_cluster
    result[CHARGE_CLUSTER_COL] = np.where(use_for_fit, nearest_cluster, np.nan)
    result[CHARGE_CENTER_COL] = nearest_center
    result[CHARGE_DISTANCE_COL] = distance
    result[CHARGE_HALF_WIDTH_COL] = nearest_half_width
    result[USE_FOR_FIT_COL] = use_for_fit
    result[DISCOVERY_QUALITY_COL] = np.where(use_for_fit, "fit",
                                             "half_width_outlier")

    for idx, peak in enumerate(peaks):
        mask = (nearest_index == idx) & use_for_fit
        peak["points"] = int(np.sum(mask))
        peak["all_nearest_points"] = int(np.sum(nearest_index == idx))
    return result, peaks


def _estimate_common_spacing(peaks: list[dict]) -> dict:
    usable = [peak for peak in peaks if peak.get("points", 0) > 0]
    centers = np.asarray([peak["center"] for peak in usable], dtype=float)
    weights = np.asarray([max(peak.get("points", 1), 1) for peak in usable],
                         dtype=float)
    if len(centers) < 2:
        return {
            "cluster_count": int(len(centers)),
            "spacing_1e19C": np.nan,
            "spacing_c": np.nan,
            "spacing_error_percent": np.nan,
            "gap_cv": np.nan,
            "equal_spacing_r2": np.nan,
        }

    ranks = np.arange(len(centers), dtype=float)
    slope, intercept = np.polyfit(ranks, centers, deg=1, w=np.sqrt(weights))
    fitted = intercept + slope * ranks
    gaps = np.diff(centers)
    gap_cv = float(np.std(gaps) / np.mean(gaps)) if np.mean(gaps) else np.nan
    ss_res = float(np.sum((centers - fitted)**2))
    ss_tot = float(np.sum((centers - np.mean(centers))**2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    spacing_c = slope * 1e-19
    spacing_error = (
        abs(spacing_c - ELEMENTARY_CHARGE_C) / ELEMENTARY_CHARGE_C * 100.0)
    return {
        "cluster_count": int(len(centers)),
        "spacing_1e19C": float(slope),
        "spacing_c": float(spacing_c),
        "spacing_error_percent": float(spacing_error),
        "gap_cv": gap_cv,
        "equal_spacing_r2": float(r2) if np.isfinite(r2) else np.nan,
        "intercept_1e19C": float(intercept),
        "centers_1e19C": centers,
        "gaps_1e19C": gaps,
        "equal_spacing_fit_1e19C": fitted,
    }


def _estimate_peak_stability(q_values: np.ndarray, peaks: list[dict],
                             config: DiscoveryRegressionConfig) -> list[dict]:
    """Bootstrap the q distribution to estimate how repeatable each peak is."""
    if not peaks or len(q_values) < max(10, config.min_points_per_cluster * 3):
        for peak in peaks:
            peak["stability_percent"] = np.nan
        return peaks

    centers = np.asarray([peak["center"] for peak in peaks], dtype=float)
    if len(centers) >= 2:
        nearest_gap = np.empty(len(centers), dtype=float)
        for idx, center in enumerate(centers):
            other = np.delete(centers, idx)
            nearest_gap[idx] = float(np.min(np.abs(other - center)))
    else:
        nearest_gap = np.full(len(centers), np.nan)

    tolerances = []
    for idx, peak in enumerate(peaks):
        half_width = float(peak.get("half_width", 0.0))
        gap_cap = (
            nearest_gap[idx] * 0.42 if np.isfinite(nearest_gap[idx])
            else max(half_width * 2.0, 0.25))
        tolerances.append(max(0.10, min(max(half_width * 1.6, 0.18), gap_cap)))
    tolerances = np.asarray(tolerances, dtype=float)

    counts = np.zeros(len(peaks), dtype=int)
    attempts = 0
    rng = np.random.default_rng(42)
    sample_size = min(len(q_values), 1500)
    for _ in range(max(0, int(config.stability_bootstrap_samples))):
        sample = rng.choice(q_values, size=sample_size, replace=True)
        peak_result = _kde_charge_peaks(sample, config)
        if peak_result is None:
            continue
        sampled_peaks, _ = peak_result
        sampled_centers = np.asarray(
            [peak["center"] for peak in sampled_peaks], dtype=float)
        if len(sampled_centers) == 0:
            continue
        attempts += 1
        for idx, center in enumerate(centers):
            if np.min(np.abs(sampled_centers - center)) <= tolerances[idx]:
                counts[idx] += 1

    for idx, peak in enumerate(peaks):
        stability = counts[idx] / attempts * 100.0 if attempts else np.nan
        peak["stability_percent"] = float(stability)
        peak["stability_attempts"] = int(attempts)
    return peaks


def _power_grid(config: DiscoveryRegressionConfig) -> np.ndarray:
    return np.round(
        np.arange(config.time_power_min, config.time_power_max + 1e-12,
                  config.time_power_step), 10)


def _rational_power(value: float) -> sp.Rational:
    return sp.Rational(int(round(value * 20)), 20)


def _power_complexity(value: float) -> int:
    rational = _rational_power(value)
    return int(abs(rational.p) + rational.q)


def _rounded_coeff(value: float) -> float:
    return round(float(value), 4)


def _power_term(base, power: float):
    return base**_rational_power(power)


def _candidate_features(family: str, q_values: np.ndarray,
                        t_values: np.ndarray, charge_power: float,
                        time_power: float) -> np.ndarray:
    q_term = np.power(q_values, charge_power)
    t_term = np.power(t_values, time_power)
    mixed = q_term * t_term
    if family == "single_power":
        return mixed.reshape(-1, 1)
    if family == "additive_power":
        return np.column_stack([q_term, t_term])
    if family == "time_correction":
        return np.column_stack([mixed, t_term])
    if family == "charge_correction":
        return np.column_stack([mixed, q_term])
    raise ValueError(f"Unknown symbolic candidate family: {family}")


def _candidate_expression(row: dict, fixed_charge: float | None = None):
    t_symbol = sp.Symbol("t", real=True, positive=True)
    q_symbol = sp.Symbol("Q_c", real=True, positive=True)
    q_base = q_symbol if fixed_charge is None else float(fixed_charge)
    q_term = _power_term(q_base, row["charge_power"])
    t_term = _power_term(t_symbol, row["time_power"])
    mixed = q_term * t_term

    coeffs = row["coefficients"]
    expr = _rounded_coeff(coeffs[0])
    if row["family"] == "single_power":
        expr += _rounded_coeff(coeffs[1]) * mixed
    elif row["family"] == "additive_power":
        expr += _rounded_coeff(coeffs[1]) * q_term
        expr += _rounded_coeff(coeffs[2]) * t_term
    elif row["family"] == "time_correction":
        expr += _rounded_coeff(coeffs[1]) * mixed
        expr += _rounded_coeff(coeffs[2]) * t_term
    elif row["family"] == "charge_correction":
        expr += _rounded_coeff(coeffs[1]) * mixed
        expr += _rounded_coeff(coeffs[2]) * q_term
    return sp.simplify(expr)


def _predict_symbolic_candidate(row: dict, q_values: np.ndarray,
                                t_values: np.ndarray) -> np.ndarray:
    features = _candidate_features(row["family"], q_values, t_values,
                                   row["charge_power"], row["time_power"])
    coeffs = np.asarray(row["coefficients"], dtype=float)
    return coeffs[0] + features @ coeffs[1:]


def _fit_candidate(family: str, q_centers: np.ndarray, t_values: np.ndarray,
                   u_values: np.ndarray, charge_power: float,
                   time_power: float) -> dict | None:
    features = _candidate_features(family, q_centers, t_values, charge_power,
                                   time_power)
    if not np.all(np.isfinite(features)):
        return None
    x_matrix = np.column_stack([np.ones(len(t_values)), features])
    if np.linalg.matrix_rank(x_matrix) < x_matrix.shape[1]:
        return None

    try:
        coeffs, *_ = np.linalg.lstsq(x_matrix, u_values, rcond=None)
    except np.linalg.LinAlgError:
        return None

    prediction = x_matrix @ coeffs
    residuals = prediction - u_values
    rss = float(np.sum(residuals**2))
    n_points = len(u_values)
    n_params = len(coeffs)
    bic = n_points * np.log(max(rss / n_points, 1e-12)
                            ) + n_params * np.log(n_points)
    mae = mean_absolute_error(u_values, prediction)
    rmse = float(np.sqrt(mean_squared_error(u_values, prediction)))
    r2 = r2_score(u_values, prediction)
    row = {
        "family": family,
        "charge_power": float(charge_power),
        "time_power": float(time_power),
        "coefficients": [float(value) for value in coeffs],
        "b": float(coeffs[0]),
        "coef1": float(coeffs[1]) if len(coeffs) > 1 else np.nan,
        "coef2": float(coeffs[2]) if len(coeffs) > 2 else np.nan,
        "bic": float(bic),
        "mae": float(mae),
        "rmse": rmse,
        "r2": float(r2),
    }
    return row


def _fit_shared_symbolic_model(clustered: pd.DataFrame,
                               config: DiscoveryRegressionConfig) -> dict:
    fit_data = clustered[
        clustered[USE_FOR_FIT_COL] &
        clustered[CHARGE_CLUSTER_COL].notna()].copy()
    if len(fit_data) < config.min_symbolic_points:
        raise ValueError("半峰宽筛选后可用于拟合的数据点不足。")
    if fit_data[CHARGE_CLUSTER_COL].nunique() < 2:
        raise ValueError("至少需要两个电荷峰才能归纳多曲线共享公式。")

    t_values = fit_data[TIME_COL].to_numpy(float)
    u_values = fit_data[VOLTAGE_COL].to_numpy(float)
    q_centers = fit_data[CHARGE_CENTER_COL].to_numpy(float)
    charge_powers = np.asarray(
        [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0],
        dtype=float,
    )

    candidate_families = [
        "single_power",
        "additive_power",
        "time_correction",
        "charge_correction",
    ]
    candidates = []
    best = None
    for charge_power in charge_powers:
        for time_power in _power_grid(config):
            for family in candidate_families:
                row = _fit_candidate(family, q_centers, t_values, u_values,
                                     charge_power, time_power)
                if row is None:
                    continue
                candidates.append(row)
                if best is None or row["bic"] < best["bic"]:
                    best = row

    if best is None:
        raise ValueError("没有找到可用的共享符号表达式。")

    candidates_df = pd.DataFrame(candidates).sort_values(
        "bic", ascending=True).head(20).reset_index(drop=True)
    if not candidates_df.empty:
        formulas = []
        latex_formulas = []
        for _, candidate in candidates_df.iterrows():
            candidate_dict = candidate.to_dict()
            expr = _candidate_expression(candidate_dict)
            formulas.append(str(expr))
            latex_formulas.append(sp.latex(expr))
        candidates_df["formula"] = formulas
        candidates_df["latex"] = latex_formulas
    expression = _candidate_expression(best)

    return {
        "best": best,
        "expression": expression,
        "candidate_models": candidates_df,
    }


def _fit_time_stage(fit_data: pd.DataFrame,
                    config: DiscoveryRegressionConfig) -> pd.DataFrame:
    """Find a common time exponent while allowing each q peak its own slope."""
    cluster_ids = sorted(fit_data[CHARGE_CLUSTER_COL].dropna().unique())
    t_values = fit_data[TIME_COL].to_numpy(float)
    u_values = fit_data[VOLTAGE_COL].to_numpy(float)
    labels = fit_data[CHARGE_CLUSTER_COL].to_numpy(float)
    rows = []

    for time_power in _power_grid(config):
        t_term = np.power(t_values, time_power)
        x_columns = [np.ones(len(fit_data))]
        for cluster_id in cluster_ids:
            x_columns.append(np.where(labels == cluster_id, t_term, 0.0))
        x_matrix = np.column_stack(x_columns)
        if not np.all(np.isfinite(x_matrix)):
            continue
        if np.linalg.matrix_rank(x_matrix) < x_matrix.shape[1]:
            continue
        try:
            coeffs, *_ = np.linalg.lstsq(x_matrix, u_values, rcond=None)
        except np.linalg.LinAlgError:
            continue

        prediction = x_matrix @ coeffs
        residuals = prediction - u_values
        rss = float(np.sum(residuals**2))
        n_points = len(u_values)
        n_params = len(coeffs)
        bic = n_points * np.log(max(rss / n_points, 1e-12)
                                ) + n_params * np.log(n_points)
        rows.append({
            "time_power": float(time_power),
            "power_complexity": _power_complexity(float(time_power)),
            "intercept": float(coeffs[0]),
            "cluster_coefficients": {
                int(cluster_id): float(coeffs[idx + 1])
                for idx, cluster_id in enumerate(cluster_ids)
            },
            "rmse": float(np.sqrt(mean_squared_error(u_values, prediction))),
            "mae": float(mean_absolute_error(u_values, prediction)),
            "r2": float(r2_score(u_values, prediction)),
            "bic": float(bic),
        })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("bic",
                                          ascending=True).reset_index(drop=True)


def _fit_coefficient_stage(coefficients: pd.DataFrame) -> pd.DataFrame:
    """Discover how per-peak time coefficients depend on the q peak center."""
    if coefficients.empty or len(coefficients) < 2:
        return pd.DataFrame()

    q_values = coefficients["Q_center(1e-19C)"].to_numpy(float)
    a_values = coefficients["time_coefficient"].to_numpy(float)
    charge_powers = np.asarray(
        [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0],
        dtype=float,
    )
    rows = []
    for charge_power in charge_powers:
        x_values = np.power(q_values, charge_power)
        if not np.all(np.isfinite(x_values)):
            continue
        denom = float(np.dot(x_values, x_values))
        if denom <= 0:
            continue
        amplitude = float(np.dot(x_values, a_values) / denom)
        prediction = amplitude * x_values
        residuals = prediction - a_values
        rss = float(np.sum(residuals**2))
        n_points = len(a_values)
        bic = n_points * np.log(max(rss / n_points, 1e-12)
                                ) + np.log(n_points)
        ss_tot = float(np.sum((a_values - np.mean(a_values))**2))
        r2 = 1.0 - rss / ss_tot if ss_tot > 0 else np.nan
        rows.append({
            "charge_power": float(charge_power),
            "power_complexity": _power_complexity(float(charge_power)),
            "amplitude": amplitude,
            "rmse": float(np.sqrt(mean_squared_error(a_values, prediction))),
            "mae": float(mean_absolute_error(a_values, prediction)),
            "r2": float(r2) if np.isfinite(r2) else np.nan,
            "bic": float(bic),
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("bic",
                                          ascending=True).reset_index(drop=True)


def _fit_two_stage_symbolic_model(clustered: pd.DataFrame,
                                  peaks: list[dict],
                                  config: DiscoveryRegressionConfig) -> dict:
    """Discover U(t, Q_c) as time-law first, coefficient-law second."""
    fit_data = clustered[
        clustered[USE_FOR_FIT_COL] &
        clustered[CHARGE_CLUSTER_COL].notna()].copy()
    if len(fit_data) < config.min_symbolic_points:
        return {}
    if fit_data[CHARGE_CLUSTER_COL].nunique() < 2:
        return {}

    time_stage = _fit_time_stage(fit_data, config)
    if time_stage.empty:
        return {}

    rmse_min = float(time_stage["rmse"].min())
    tolerance = 1.0 + float(config.symbolic_pareto_tolerance_percent) / 100.0
    time_near_best = time_stage[time_stage["rmse"] <= rmse_min * tolerance]
    time_best_idx = time_near_best.sort_values(
        ["power_complexity", "bic"], ascending=[True, True]).index[0]
    time_stage["selected"] = False
    time_stage.loc[time_best_idx, "selected"] = True
    time_best = time_stage.loc[time_best_idx].to_dict()
    coefficient_map = time_best["cluster_coefficients"]
    peak_map = {int(peak["cluster"]): peak for peak in peaks}
    coefficient_rows = []
    for cluster_id, coefficient in coefficient_map.items():
        peak = peak_map.get(int(cluster_id))
        if peak is None:
            continue
        coefficient_rows.append({
            "cluster": int(cluster_id),
            "Q_center(1e-19C)": float(peak["center"]),
            "time_coefficient": float(coefficient),
            "points": int(peak.get("points", 0)),
        })
    coefficient_df = pd.DataFrame(coefficient_rows).sort_values(
        "Q_center(1e-19C)").reset_index(drop=True)
    coefficient_stage = _fit_coefficient_stage(coefficient_df)
    if coefficient_stage.empty:
        return {
            "time_stage_candidates": time_stage,
            "cluster_coefficients": coefficient_df,
        }

    coeff_rmse_min = float(coefficient_stage["rmse"].min())
    coeff_near_best = coefficient_stage[
        coefficient_stage["rmse"] <= coeff_rmse_min * tolerance]
    coeff_best_idx = coeff_near_best.sort_values(
        ["power_complexity", "bic"], ascending=[True, True]).index[0]
    coefficient_stage["selected"] = False
    coefficient_stage.loc[coeff_best_idx, "selected"] = True
    coeff_best = coefficient_stage.loc[coeff_best_idx].to_dict()
    candidate = {
        "family": "single_power",
        "charge_power": float(coeff_best["charge_power"]),
        "time_power": float(time_best["time_power"]),
        "coefficients": [
            float(time_best["intercept"]),
            float(coeff_best["amplitude"]),
        ],
        "b": float(time_best["intercept"]),
        "coef1": float(coeff_best["amplitude"]),
        "coef2": np.nan,
    }

    t_values = fit_data[TIME_COL].to_numpy(float)
    u_values = fit_data[VOLTAGE_COL].to_numpy(float)
    q_centers = fit_data[CHARGE_CENTER_COL].to_numpy(float)
    prediction = _predict_symbolic_candidate(candidate, q_centers, t_values)
    residuals = prediction - u_values
    rss = float(np.sum(residuals**2))
    n_points = len(u_values)
    n_params = 2
    bic = n_points * np.log(max(rss / n_points, 1e-12)
                            ) + n_params * np.log(n_points)
    candidate.update({
        "bic": float(bic),
        "mae": float(mean_absolute_error(u_values, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(u_values, prediction))),
        "r2": float(r2_score(u_values, prediction)),
    })
    expression = _candidate_expression(candidate)

    return {
        "best": candidate,
        "expression": expression,
        "time_stage_candidates": time_stage,
        "coefficient_stage_candidates": coefficient_stage,
        "cluster_coefficients": coefficient_df,
    }


def _teacher_features(q_values: np.ndarray, t_values: np.ndarray) -> np.ndarray:
    return np.column_stack([
        np.log(np.asarray(t_values, dtype=float)),
        np.log(np.asarray(q_values, dtype=float)),
    ])


def _fit_teacher_candidate(family: str, q_centers: np.ndarray,
                           t_values: np.ndarray, u_values: np.ndarray,
                           charge_power: float, time_power: float) -> dict | None:
    row = _fit_candidate(family, q_centers, t_values, u_values, charge_power,
                         time_power)
    if row is None:
        return None
    row["power_complexity"] = (
        _power_complexity(charge_power) + _power_complexity(time_power))
    row["distilled_from"] = "neural_teacher"
    return row


def _select_distilled_candidate(candidates: pd.DataFrame,
                                config: DiscoveryRegressionConfig) -> dict | None:
    if candidates.empty:
        return None
    candidates["selected"] = False
    candidates["role"] = ""
    empirical_idx = candidates.sort_values(
        ["bic", "rmse"], ascending=[True, True]).index[0]
    candidates.loc[empirical_idx, "role"] = "lowest_error"

    simple_pool = candidates[candidates["family"] == "single_power"]
    if not simple_pool.empty:
        r2_floor = float(simple_pool["r2"].max()) - 0.001
        near_simple_best = simple_pool[simple_pool["r2"] >= r2_floor]
        selected_idx = near_simple_best.sort_values(
            ["power_complexity", "bic"], ascending=[True, True]).index[0]
        candidates.loc[selected_idx, "role"] = "simple_main_law"
    else:
        rmse_min = float(candidates["rmse"].min())
        tolerance = 1.0 + float(
            config.symbolic_pareto_tolerance_percent) / 100.0
        near_best = candidates[candidates["rmse"] <= rmse_min * tolerance]
        selected_idx = near_best.sort_values(
            ["power_complexity", "bic"], ascending=[True, True]).index[0]
        candidates.loc[selected_idx, "role"] = "pareto_simple"

    candidates["selected"] = False
    candidates.loc[selected_idx, "selected"] = True
    return candidates.loc[selected_idx].to_dict()


def _fit_neural_teacher_distillation(clustered: pd.DataFrame,
                                     peaks: list[dict],
                                     config: DiscoveryRegressionConfig) -> dict:
    """Use a neural teacher to denoise the discovered curves, then distill it."""
    if int(config.neural_teacher_models) <= 0:
        return {}
    fit_data = clustered[
        clustered[USE_FOR_FIT_COL] &
        clustered[CHARGE_CLUSTER_COL].notna()].copy()
    if len(fit_data) < max(40, config.min_symbolic_points * 3):
        return {}
    if fit_data[CHARGE_CLUSTER_COL].nunique() < 2:
        return {}

    q_values = fit_data[CHARGE_CENTER_COL].to_numpy(float)
    t_values = fit_data[TIME_COL].to_numpy(float)
    u_values = fit_data[VOLTAGE_COL].to_numpy(float)
    x_values = _teacher_features(q_values, t_values)
    y_mean = float(np.mean(u_values))
    y_std = float(np.std(u_values))
    if y_std <= 0 or not np.isfinite(y_std):
        return {}
    y_scaled = (u_values - y_mean) / y_std

    rng = np.random.default_rng(2026)
    order = rng.permutation(len(fit_data))
    test_count = max(1, int(len(order) * 0.2))
    test_idx = order[:test_count]
    train_idx = order[test_count:]
    if len(train_idx) < 20:
        return {}

    models = []
    seeds = [11, 23, 37, 51, 73, 97, 131]
    for seed in seeds[:max(1, int(config.neural_teacher_models))]:
        model = make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(48, 24),
                activation="tanh",
                alpha=1e-3,
                learning_rate_init=0.01,
                max_iter=900,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=25,
                random_state=seed,
            ),
        )
        try:
            model.fit(x_values[train_idx], y_scaled[train_idx])
        except Exception:
            continue
        models.append(model)

    if not models:
        return {}

    def predict_teacher(q_array: np.ndarray, t_array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x_array = _teacher_features(q_array, t_array)
        raw = np.column_stack([model.predict(x_array) for model in models])
        pred = raw.mean(axis=1) * y_std + y_mean
        uncertainty = raw.std(axis=1) * y_std
        return pred, uncertainty

    train_pred, _ = predict_teacher(q_values[train_idx], t_values[train_idx])
    test_pred, test_uncertainty = predict_teacher(q_values[test_idx],
                                                  t_values[test_idx])
    train_y = u_values[train_idx]
    test_y = u_values[test_idx]
    teacher_metrics = {
        "model": "MLP ensemble",
        "model_count": int(len(models)),
        "train_rmse": float(np.sqrt(mean_squared_error(train_y, train_pred))),
        "test_rmse": float(np.sqrt(mean_squared_error(test_y, test_pred))),
        "test_mae": float(mean_absolute_error(test_y, test_pred)),
        "test_r2": float(r2_score(test_y, test_pred)),
        "mean_uncertainty_v": float(np.mean(test_uncertainty)),
        "train_points": int(len(train_idx)),
        "test_points": int(len(test_idx)),
    }

    teacher_q = []
    teacher_t = []
    teacher_u = []
    teacher_cluster = []
    teacher_uncertainty = []
    for peak in peaks:
        cluster_id = int(peak["cluster"])
        sub = fit_data[fit_data[CHARGE_CLUSTER_COL] == cluster_id]
        if len(sub) < config.min_points_per_cluster:
            continue
        q_center = float(peak["center"])
        t_min, t_max = np.percentile(sub[TIME_COL].to_numpy(float), [2, 98])
        if not np.isfinite(t_min) or not np.isfinite(t_max) or t_max <= t_min:
            continue
        t_line = np.linspace(float(t_min), float(t_max), 90)
        q_line = np.full_like(t_line, q_center, dtype=float)
        u_line, uncertainty_line = predict_teacher(q_line, t_line)
        teacher_q.append(q_line)
        teacher_t.append(t_line)
        teacher_u.append(u_line)
        teacher_cluster.extend([cluster_id] * len(t_line))
        teacher_uncertainty.append(uncertainty_line)

    if not teacher_q:
        return {}
    teacher_q_values = np.concatenate(teacher_q)
    teacher_t_values = np.concatenate(teacher_t)
    teacher_u_values = np.concatenate(teacher_u)
    teacher_uncertainty_values = np.concatenate(teacher_uncertainty)

    charge_powers = np.asarray(
        [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0],
        dtype=float,
    )
    candidate_families = [
        "single_power",
        "additive_power",
        "time_correction",
        "charge_correction",
    ]
    candidates = []
    for charge_power in charge_powers:
        for time_power in _power_grid(config):
            for family in candidate_families:
                row = _fit_teacher_candidate(
                    family, teacher_q_values, teacher_t_values,
                    teacher_u_values, charge_power, time_power)
                if row is not None:
                    candidates.append(row)
    if not candidates:
        return {}

    candidates_df = pd.DataFrame(candidates).sort_values(
        "bic", ascending=True).reset_index(drop=True)
    best = _select_distilled_candidate(candidates_df, config)
    if best is None:
        return {}

    selected_indices = candidates_df[
        candidates_df.get("selected", False)].index.tolist()
    role_indices = candidates_df[
        candidates_df.get("role", "") != ""].index.tolist()
    keep_indices = list(dict.fromkeys(list(range(20)) + selected_indices +
                                      role_indices))
    display_candidates = candidates_df.loc[keep_indices].copy()
    formulas = []
    latex_formulas = []
    for _, candidate in display_candidates.iterrows():
        candidate_dict = candidate.to_dict()
        expr = _candidate_expression(candidate_dict)
        formulas.append(str(expr))
        latex_formulas.append(sp.latex(expr))
    display_candidates["formula"] = formulas
    display_candidates["latex"] = latex_formulas
    expression = _candidate_expression(best)

    teacher_frame = pd.DataFrame({
        "cluster": np.asarray(teacher_cluster, dtype=int),
        "Q_center(1e-19C)": teacher_q_values,
        TIME_COL: teacher_t_values,
        "TeacherVoltage(U/V)": teacher_u_values,
        "TeacherUncertainty(V)": teacher_uncertainty_values,
    })

    return {
        "best": best,
        "expression": expression,
        "candidate_models": display_candidates,
        "teacher_metrics": teacher_metrics,
        "teacher_points": teacher_frame,
    }


def discovery_regression(
        data: pd.DataFrame,
        config: DiscoveryRegressionConfig | None = None) -> dict:
    """Discover charge peaks first, then fit shared symbolic U(t, Q_c)."""
    config = config or DiscoveryRegressionConfig()
    clustering_result = charge_clustering(data, config)
    peaks = clustering_result["peaks"]
    density_info = clustering_result["density"]
    clustered = clustering_result["clusters"]
    spacing = clustering_result["spacing"]
    symbolic = _fit_shared_symbolic_model(clustered, config)
    two_stage = _fit_two_stage_symbolic_model(clustered, peaks, config)
    neural = _fit_neural_teacher_distillation(clustered, peaks, config)

    strategy = getattr(config, "symbolic_model_strategy", "auto")
    selected_method = "global_candidate_search"
    best = symbolic["best"]
    symbolic_expression = symbolic["expression"]
    if strategy == "two_stage":
        if two_stage.get("best"):
            best = two_stage["best"]
            symbolic_expression = two_stage["expression"]
            selected_method = "two_stage"
    elif strategy == "neural_teacher_distillation":
        if neural.get("best"):
            best = neural["best"]
            symbolic_expression = neural["expression"]
            selected_method = "neural_teacher_distillation"
        elif two_stage.get("best"):
            best = two_stage["best"]
            symbolic_expression = two_stage["expression"]
            selected_method = "two_stage"
    elif strategy == "global_candidate_search":
        selected_method = "global_candidate_search"
    else:
        if neural.get("best"):
            best = neural["best"]
            symbolic_expression = neural["expression"]
            selected_method = "neural_teacher_distillation"
        elif two_stage.get("best"):
            best = two_stage["best"]
            symbolic_expression = two_stage["expression"]
            selected_method = "two_stage"
    fit_data = clustered[clustered[USE_FOR_FIT_COL]].copy()
    prediction = _predict_symbolic_candidate(
        best,
        fit_data[CHARGE_CENTER_COL].to_numpy(float),
        fit_data[TIME_COL].to_numpy(float),
    )
    clustered["SymbolicResidual(V)"] = np.nan
    clustered.loc[fit_data.index, "SymbolicResidual(V)"] = (
        prediction - fit_data[VOLTAGE_COL].to_numpy(float))

    data_by_cluster = {}
    evaluation = {}
    peak_summary = []
    for peak in peaks:
        cluster_id = int(peak["cluster"])
        sub = clustered[
            (clustered[CHARGE_CLUSTER_COL] == cluster_id) &
            clustered[USE_FOR_FIT_COL]].sort_values(TIME_COL)
        if len(sub) < config.min_points_per_cluster:
            continue

        q_center = float(peak["center"])
        t_values = sub[TIME_COL].to_numpy(float)
        u_values = sub[VOLTAGE_COL].to_numpy(float)
        t_min, t_max = float(t_values.min()), float(t_values.max())
        if np.isclose(t_min, t_max):
            t_min = max(t_min * 0.95, 1e-9)
            t_max = t_max * 1.05
        t_line = np.linspace(t_min, t_max, 200)
        y_line = _predict_symbolic_candidate(
            best,
            np.full_like(t_line, q_center, dtype=float),
            t_line,
        )
        y_pred = _predict_symbolic_candidate(
            best,
            np.full_like(t_values, q_center, dtype=float),
            t_values,
        )

        mse = mean_squared_error(u_values, y_pred)
        mae = mean_absolute_error(u_values, y_pred)
        r2 = r2_score(u_values, y_pred) if len(sub) >= 2 else np.nan
        cluster_expr = _candidate_expression(best, fixed_charge=q_center)

        data_by_cluster[cluster_id] = [t_line, y_line, cluster_expr]
        evaluation[cluster_id] = {
            "params": {
                "q_center_1e19C": round(q_center, 4),
                "half_width_1e19C": round(float(peak["half_width"]), 4),
            },
            "mse": float(mse),
            "mae": float(mae),
            "r2": float(r2) if np.isfinite(r2) else np.nan,
            "points": int(len(sub)),
        }
        peak_summary.append({
            "cluster": cluster_id,
            "Q_center(1e-19C)": round(q_center, 4),
            "half_width(1e-19C)": round(float(peak["half_width"]), 4),
            "stability(%)": round(float(peak.get("stability_percent")), 1)
            if np.isfinite(peak.get("stability_percent", np.nan)) else np.nan,
            "points": int(len(sub)),
            "nearest_points": int(peak.get("all_nearest_points", 0)),
            "method": peak.get("method", ""),
            "mse": round(float(mse), 4),
            "mae": round(float(mae), 4),
            "r2": round(float(r2), 4) if np.isfinite(r2) else np.nan,
        })

    return {
        "result_version": "q-ai-clustering-symbolic-v10",
        "mode": "discovery",
        "regression_form": (
            "Charge clusters are discovered from q with unsupervised learning; "
            "post-cluster half-width filtering selects high-confidence points, "
            "then symbolic regression searches a shared interpretable law."),
        "global_params": {
            "family": best["family"],
            "discovery_method": selected_method,
            "requested_symbolic_model": strategy,
            "coef1": round(float(best["coef1"]), 4),
            "coef2": round(float(best["coef2"]), 4)
            if np.isfinite(best["coef2"]) else np.nan,
            "b": round(float(best["b"]), 4),
            "charge_power": round(float(best["charge_power"]), 4),
            "time_power": round(float(best["time_power"]), 4),
            "formula_rmse": round(float(best["rmse"]), 4),
            "formula_mae": round(float(best["mae"]), 4),
            "formula_r2": round(float(best["r2"]), 4),
            "fit_points": int(clustered[USE_FOR_FIT_COL].sum()),
            "cluster_count": int(spacing["cluster_count"]),
            "spacing_1e19C": round(float(spacing["spacing_1e19C"]), 4)
            if np.isfinite(spacing["spacing_1e19C"]) else np.nan,
            "spacing_c": float(spacing["spacing_c"])
            if np.isfinite(spacing["spacing_c"]) else np.nan,
            "spacing_error_percent": round(
                float(spacing["spacing_error_percent"]), 2)
            if np.isfinite(spacing["spacing_error_percent"]) else np.nan,
            "gap_cv": round(float(spacing["gap_cv"]), 4)
            if np.isfinite(spacing["gap_cv"]) else np.nan,
            "equal_spacing_r2": round(float(spacing["equal_spacing_r2"]), 4)
            if np.isfinite(spacing["equal_spacing_r2"]) else np.nan,
            "fall_distance_mm": float(config.fall_distance_mm),
            "plate_distance_mm": float(config.plate_distance_mm),
            "clustering_method": str(config.clustering_method),
            "requested_clusters": config.requested_clusters,
            "half_width_1e19c": config.half_width_1e19c,
            "kde_bandwidth": float(config.kde_bandwidth),
            "peak_prominence": float(config.peak_prominence),
            "analysis_lower_percentile": float(config.analysis_lower_percentile),
            "analysis_upper_percentile": float(config.analysis_upper_percentile),
            "symbolic_pareto_tolerance_percent": float(
                config.symbolic_pareto_tolerance_percent),
        },
        "symbolic_expression": symbolic_expression,
        "candidate_models": symbolic["candidate_models"],
        "two_stage": two_stage,
        "neural_teacher": neural,
        "density": density_info,
        "spacing": spacing,
        "peaks": peaks,
        "clusters": clustered,
        "peak_summary": pd.DataFrame(peak_summary),
        "evaluation": evaluation,
        "data": data_by_cluster,
    }


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
    return discovery_regression(data_pred)
