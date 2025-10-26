"""HAC/Newey–West t-test for QuantConnect backtest JSONs."""

import json
import math
import os
from glob import glob  # noqa: F401 (public surface parity with original snippet)
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.stattools import durbin_watson

SCIPY_SURVIVAL_FUNCTION = getattr(norm, "sf")


def _daily_last_unique(source_series: pd.Series) -> pd.Series:
    """Return daily last observation with stable timezone handling."""
    parsed_index = pd.to_datetime(
        source_series.index, utc=True, errors="coerce"
    )
    numeric_series = pd.Series(
        pd.to_numeric(source_series.to_numpy(copy=False), errors="coerce"),
        index=parsed_index,
    )
    cleaned = numeric_series[~numeric_series.index.isna()]
    cleaned = cleaned.dropna()
    if cleaned.empty:
        return pd.Series(dtype=float)
    cleaned.index = pd.DatetimeIndex(cleaned.index).tz_convert(None)
    cleaned = cleaned.sort_index()
    cleaned = cleaned[~cleaned.index.duplicated(keep="last")]
    cleaned = cleaned.resample("D").last().dropna()
    return cleaned.astype(float)


def _first_non_empty_series(charts: Dict) -> Optional[pd.Series]:
    """Return the first usable equity series from QC chart blobs."""
    if charts is None:
        return None

    def _extract_values(series_object) -> Optional[pd.Series]:
        series_values = None
        if isinstance(series_object, dict):
            series_values = series_object.get("values") or series_object.get(
                "Values"
            )
        if not series_values or not isinstance(series_values, list):
            return None
        parsed_rows = []
        for point in series_values:
            if not isinstance(point, dict):
                continue
            raw_timestamp = (
                point.get("x")
                or point.get("time")
                or point.get("t")
                or point.get("date")
            )
            raw_value = point.get("y") or point.get("value") or point.get("v")
            if raw_value is None or raw_timestamp is None:
                continue
            try:
                numeric_value = float(raw_value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(numeric_value):
                continue
            timestamp = pd.to_datetime(
                raw_timestamp,
                utc=True,
                errors="coerce",
            )
            if pd.isna(timestamp):
                continue
            parsed_rows.append((timestamp, numeric_value))
        if not parsed_rows:
            return None
        timestamp_index, values = zip(*parsed_rows)
        extracted_series = pd.Series(
            values, index=pd.DatetimeIndex(timestamp_index)
        )
        extracted_series = _daily_last_unique(extracted_series)
        if extracted_series.empty:
            return None
        return extracted_series

    preferred_chart_names = [
        "TWR NAV",
        "Strategy Equity",
        "Equity",
        "Portfolio Equity",
    ]
    for preferred_name in preferred_chart_names:
        chart = (
            charts.get(preferred_name) if isinstance(charts, dict) else None
        )
        if not isinstance(chart, dict):
            continue
        series_dict = chart.get("series") or chart.get("Series") or {}
        if isinstance(series_dict, dict):
            for series_name in ["NAV", "Equity", "Strategy Equity"]:
                series_object = series_dict.get(series_name)
                extracted_series = (
                    _extract_values(series_object) if series_object else None
                )
                if extracted_series is not None and len(extracted_series) > 1:
                    return extracted_series
            for series_object in series_dict.values():
                extracted_series = _extract_values(series_object)
                if extracted_series is not None and len(extracted_series) > 1:
                    return extracted_series

    if isinstance(charts, dict):
        for chart in charts.values():
            if not isinstance(chart, dict):
                continue
            series_dict = chart.get("series") or chart.get("Series") or {}
            if isinstance(series_dict, dict):
                for series_object in series_dict.values():
                    extracted_series = _extract_values(series_object)
                    if (
                        extracted_series is not None
                        and len(extracted_series) > 1
                    ):
                        return extracted_series
    return None


def load_qc_equity_series(path: str):
    """Return series, label, and shift extracted from a QC backtest JSON."""
    with open(path, "r", encoding="utf-8") as file_handle:
        json_blob = json.load(file_handle)
    charts = json_blob.get("charts") or json_blob.get("Charts")
    equity_series = _first_non_empty_series(charts)
    if equity_series is not None:
        equity_series = _daily_last_unique(equity_series)
        if equity_series.empty:
            equity_series = None
    if equity_series is None:
        rolling_window = json_blob.get("rollingWindow") or {}
        rows = []
        for window_key, window_value in rolling_window.items():
            portfolio_stats = (window_value or {}).get(
                "portfolioStatistics"
            ) or {}
            end_equity = portfolio_stats.get("endEquity")
            date_token = None
            if "_" in window_key:
                date_token = window_key.split("_", 1)[1]
            if end_equity is not None and date_token is not None:
                try:
                    rows.append(
                        (
                            pd.to_datetime(
                                date_token,
                                utc=True,
                                errors="coerce",
                            ),
                            float(end_equity),
                        )
                    )
                except Exception:
                    pass
        if rows:
            valid_rows = [
                (timestamp, value)
                for timestamp, value in rows
                if pd.notna(timestamp)
            ]
            if valid_rows:
                index_values, equity_values = zip(*valid_rows)
                equity_series = pd.Series(
                    equity_values, index=pd.DatetimeIndex(index_values)
                )
                equity_series = _daily_last_unique(equity_series)
            else:
                equity_series = None
        if equity_series is None or equity_series.empty:
            raise ValueError(f"Could not extract a usable series from {path}.")
    label = os.path.basename(path)
    parameter_config = json_blob.get("parameterSet") or {}
    shift_raw = (
        parameter_config.get("BacktestShiftMonths")
        or parameter_config.get("backtestShiftMonths")
        or "0"
    )
    try:
        shift_months = int(float(shift_raw))
    except Exception:
        shift_months = 0
    return equity_series.astype(float), label, shift_months


def equity_to_log_returns(equity_series: pd.Series) -> pd.Series:
    """Convert equity curve to daily log returns."""
    daily_series = _daily_last_unique(equity_series)
    if daily_series.empty:
        return pd.Series(dtype=float)
    positive_values = daily_series[daily_series > 0]
    log_returns = np.log(positive_values / positive_values.shift(1))
    log_returns = log_returns.replace([np.inf, -np.inf], np.nan).dropna()
    return log_returns


def add_multiple_testing(
    results_dataframe: pd.DataFrame,
    p_column: str = "pvalue",
    alpha: float = 0.05,
):
    """
    Adds adjusted p-values and reject flags for:
      - Bonferroni (FWER)
      - Holm (FWER)
      - Benjamini–Hochberg (FDR)
    """
    p_values = np.asarray(results_dataframe[p_column].values, dtype=float)

    adjustment_methods = [
        ("bonferroni", "bonferroni"),
        ("holm", "holm"),
        ("fdr_bh", "fdr_bh"),
    ]
    for adjustment_label, adjustment_method in adjustment_methods:
        test_result = multipletests(
            p_values, alpha=alpha, method=adjustment_method
        )
        reject_flags, adjusted_p_values = test_result[:2]
        results_dataframe[f"p_adj_{adjustment_label}"] = adjusted_p_values
        results_dataframe[f"reject_{adjustment_label}"] = reject_flags

    return results_dataframe


def _auto_bandwidth_L(sample_size: int) -> int:
    """Andrews (1991)-style rule of thumb used for 'auto'."""
    if sample_size <= 1:
        return 0
    bandwidth = int(math.floor(4.0 * (sample_size / 100.0) ** (2.0 / 9.0)))
    return max(0, bandwidth)


def _nw_mean_test_with_statsmodels(
    diff_log_returns: np.ndarray,
    bandwidth_lags: Optional[int],
    alpha: float,
):
    """
    Regress diff on a constant and use HAC (Newey–West) covariance to
    get the t-stat and CI for the mean. Returns tuple:
      (mean_hat, t_stat, p_two, p_gt, (ci_low, ci_up), L_used)
    """
    returns_array = np.asarray(diff_log_returns, dtype=float)
    returns_array = returns_array[np.isfinite(returns_array)]
    sample_size = returns_array.shape[0]
    if sample_size < 2:
        return (np.nan, np.nan, np.nan, np.nan, (np.nan, np.nan), 0)

    if bandwidth_lags is None or (
        isinstance(bandwidth_lags, str)
        and str(bandwidth_lags).lower() == "auto"
    ):
        bandwidth_lags = _auto_bandwidth_L(sample_size)
    max_lags = int(max(0, min(int(bandwidth_lags), max(0, sample_size - 1))))

    design_matrix = np.ones((sample_size, 1), dtype=float)
    model = sm.OLS(returns_array, design_matrix)
    fitted_model = model.fit(
        cov_type="HAC",
        cov_kwds={"maxlags": max_lags},
    )

    mean_estimate = float(fitted_model.params[0])
    t_statistic = float(fitted_model.tvalues[0])
    two_sided_p_value = float(fitted_model.pvalues[0])
    greater_p_value = (
        float(SCIPY_SURVIVAL_FUNCTION(t_statistic))
        if np.isfinite(t_statistic)
        else float("nan")
    )
    ci_low, ci_up = [
        float(confidence)
        for confidence in fitted_model.conf_int(alpha=alpha)[0]
    ]

    return (
        mean_estimate,
        t_statistic,
        two_sided_p_value,
        greater_p_value,
        (ci_low, ci_up),
        max_lags,
    )


def compare_files(
    model_path: str,
    benchmark_path: str,
    alpha: float = 0.05,
    bandwidth_lags: Optional[int] = None,
) -> dict:
    """Compare one model file to one benchmark file. Returns a dict result."""
    model_series, model_label, model_shift = load_qc_equity_series(model_path)
    benchmark_series, benchmark_label, benchmark_shift = load_qc_equity_series(
        benchmark_path
    )

    model_returns = equity_to_log_returns(model_series)
    benchmark_returns = equity_to_log_returns(benchmark_series)
    joined_returns = pd.concat(
        [model_returns, benchmark_returns], axis=1, join="inner"
    )
    joined_returns.columns = ["model", "bench"]
    joined_returns = joined_returns.dropna()
    if joined_returns.shape[0] < 3:
        raise ValueError(
            "Insufficient overlapping daily points to run the test."
        )

    return_diff = (joined_returns["model"] - joined_returns["bench"]).values
    (
        mean_estimate,
        t_statistic,
        two_sided_p_value,
        greater_p_value,
        (ci_lower, ci_upper),
        lags_used,
    ) = _nw_mean_test_with_statsmodels(return_diff, bandwidth_lags, alpha)

    model_is_better = (
        np.isfinite(greater_p_value)
        and greater_p_value < alpha
        and mean_estimate > 0.0
    )

    start_date = joined_returns.index.min()
    end_date = joined_returns.index.max()
    result = {
        "model_label": model_label,
        "benchmark_label": benchmark_label,
        "start_date": str(pd.to_datetime(start_date).date()),
        "end_date": str(pd.to_datetime(end_date).date()),
        "n_days": int(joined_returns.shape[0]),
        "L": int(lags_used),
        "mean_diff_logret": float(mean_estimate),
        "t_stat": float(t_statistic),
        "p_value_two_sided": float(two_sided_p_value),
        "p_value_one_sided_gt": float(greater_p_value),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "alpha": float(alpha),
        "better_than_benchmark": bool(model_is_better),
        "shift_months_model": int(model_shift),
        "shift_months_benchmark": int(benchmark_shift),
    }
    return result


def _pair_by_shift(
    models: List[str],
    benches: List[str],
) -> List[Tuple[str, str, int]]:
    """Pair files by BacktestShiftMonths (from JSON content)."""

    def _shift(path: str):
        try:
            return load_qc_equity_series(path)[2]
        except Exception:
            return None

    model_shifts = {_shift(path): path for path in models}
    benchmark_shifts = {_shift(path): path for path in benches}
    pairs: List[Tuple[str, str, int]] = []
    for shift_value in sorted(
        set(model_shifts.keys()) & set(benchmark_shifts.keys())
    ):
        if shift_value is None:
            continue
        pairs.append(
            (
                model_shifts[shift_value],
                benchmark_shifts[shift_value],
                shift_value,
            )
        )
    return pairs


def list_pair_names(pairs: Iterable[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Return [(basename(model), basename(bench)), ...] for display/logging."""
    formatted_pairs: List[Tuple[str, str]] = []
    for pair in pairs:
        model_path, benchmark_path = pair[0], pair[1]
        formatted_pairs.append(
            (
                os.path.basename(model_path),
                os.path.basename(benchmark_path),
            )
        )
    return formatted_pairs


def _summarize_results(results: List[dict]) -> dict:
    """Summarize sample-wide HAC comparison results."""
    results_dataframe = pd.DataFrame(results)
    if results_dataframe.empty:
        return {
            "n_pairs": 0,
            "n_significant_one_sided": 0,
            "share_significant": 0.0,
            "t_min": None,
            "t_median": None,
            "t_max": None,
            "mean_diff_min": None,
            "mean_diff_median": None,
            "mean_diff_max": None,
        }
    significant_count = int(
        results_dataframe["better_than_benchmark"].astype(bool).sum()
    )
    return {
        "n_pairs": int(len(results)),
        "n_significant_one_sided": significant_count,
        "share_significant": (
            float(significant_count) / float(len(results)) if results else 0.0
        ),
        "t_min": float(results_dataframe["t_stat"].min()),
        "t_median": float(results_dataframe["t_stat"].median()),
        "t_max": float(results_dataframe["t_stat"].max()),
        "mean_diff_min": float(results_dataframe["mean_diff_logret"].min()),
        "mean_diff_median": float(
            results_dataframe["mean_diff_logret"].median()
        ),
        "mean_diff_max": float(results_dataframe["mean_diff_logret"].max()),
    }


def _maybe_save(
    results: List[dict],
    summary: dict,
    out_json: Optional[str] = None,
    out_csv: Optional[str] = None,
):
    """Optionally persist comparison outputs."""
    results_dataframe = pd.DataFrame(results)
    if out_json:
        blob = {"results": results, "summary": summary}
        with open(out_json, "w", encoding="utf-8") as file_handle:
            file_handle.write(json.dumps(blob, indent=2) + "\n")
    if out_csv:
        results_dataframe.to_csv(out_csv, index=False)
    return results_dataframe


def compare_from_list(
    pairs: Iterable[Tuple[str, str]],
    alpha: float = 0.05,
    bandwidth_lags: Optional[int] = "auto",
    out_json: Optional[str] = None,
    out_csv: Optional[str] = None,
):
    """
    Compare a list of (model_path, benchmark_path) pairs.
    Returns (results_list, summary_dict, results_df).
    Optionally saves to out_json (full blob) and/or out_csv (row-wise results).
    """
    if isinstance(bandwidth_lags, str) and bandwidth_lags.lower() != "auto":
        bandwidth_lags = int(bandwidth_lags)

    comparison_results: List[dict] = []
    for model_path, benchmark_path in pairs:
        comparison_record = compare_files(
            model_path,
            benchmark_path,
            alpha=alpha,
            bandwidth_lags=bandwidth_lags,
        )
        comparison_results.append(comparison_record)

    summary = _summarize_results(comparison_results)
    results_dataframe = _maybe_save(
        comparison_results,
        summary,
        out_json=out_json,
        out_csv=out_csv,
    )
    return comparison_results, summary, results_dataframe


def results_to_dataframe(results: List[dict]) -> pd.DataFrame:
    """Convert results list to DataFrame."""
    return pd.DataFrame(results)


def show_pairs(pairs: Iterable[Tuple[str, str]]):
    """Return display-friendly DataFrame of paired file names."""
    names = list_pair_names(pairs)
    return pd.DataFrame(names, columns=["model", "benchmark"])


# %% Durbin-Watson diagnostics helper (Notebook cell)
def report_durbin_watson(model_result, lower: float = 1.5, upper: float = 2.5):
    """
    Notebook-friendly Durbin-Watson check for an OLS model result.
    Prints guidance and, when needed, returns a HAC-refitted summary.
    """
    if model_result is None:
        raise ValueError("model_result cannot be None.")

    dw_stat = float(durbin_watson(model_result.resid))
    print(f"[model] Durbin–Watson: {dw_stat:.3f}")

    if lower <= dw_stat <= upper:
        print(
            "[model] No strong evidence of autocorrelation; "
            "using conventional SEs."
        )
        return {"durbin_watson": dw_stat, "hac_summary": None, "maxlags": None}

    nobs = getattr(model_result, "nobs", None) or 100
    try:
        maxlags = max(1, int(round(4 * ((nobs) / 100.0) ** (2.0 / 9.0))))
    except Exception:
        maxlags = 3

    try:
        hac_model = model_result.get_robustcov_results(
            cov_type="HAC",
            maxlags=maxlags,
        )
    except Exception as exc:
        print(
            f"[model] HAC recomputation failed with maxlags={maxlags}: {exc}"
        )
        return {
            "durbin_watson": dw_stat,
            "hac_summary": None,
            "maxlags": maxlags,
            "error": str(exc),
        }

    print(
        "[model] Recomputed robust SEs with HAC (Newey–West), "
        f"maxlags={maxlags}"
    )
    try:
        from IPython.display import display

        display(hac_model.summary())
    except Exception:
        # Fall back to plain text summary when IPython display is unavailable.
        print(hac_model.summary())

    return {
        "durbin_watson": dw_stat,
        "hac_summary": hac_model.summary(),
        "maxlags": maxlags,
    }
