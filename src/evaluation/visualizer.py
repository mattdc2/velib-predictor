"""Visualisation utilities for the Velib evaluation framework.

All functions return :class:`matplotlib.figure.Figure` objects — they do not
call ``plt.show()`` so that callers control display (notebook, file save, etc.).

Available plots
---------------
* :func:`plot_predictions`      — actual vs predicted time series
* :func:`plot_residuals`        — residual distribution (histogram + KDE)
* :func:`plot_metric_comparison`— bar chart of MAE / RMSE / MAPE / R² across models
* :func:`plot_error_by_hour`    — mean absolute error broken down by hour of day
* :func:`plot_dashboard`        — 2×2 composite of the four plots above

Usage::

    from src.evaluation.visualizer import plot_dashboard
    fig = plot_dashboard(evaluator, timestamp_col="time")
    fig.savefig("eval_dashboard.png", dpi=150, bbox_inches="tight")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd
from loguru import logger

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from src.evaluation.evaluator import EvaluationResult, ModelEvaluator

# Colour palette — consistent across all plots
_PALETTE = [
    "#4C72B0",
    "#DD8452",
    "#55A868",
    "#C44E52",
    "#8172B2",
    "#937860",
    "#DA8BC3",
    "#8C8C8C",
]


def _get_mpl():
    """Lazy import of matplotlib to avoid hard dependency at module level."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker

        return plt, mticker
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "matplotlib is required for visualisation. Install it with: pip install matplotlib"
        ) from exc


# ---------------------------------------------------------------------------
# Individual plot functions
# ---------------------------------------------------------------------------


def plot_predictions(
    result: "EvaluationResult",
    timestamp_col: Optional[str] = None,
    max_points: int = 500,
    title: Optional[str] = None,
) -> "Figure":
    """Plot actual vs predicted values for a single model.

    Args:
        result: An :class:`~src.evaluation.evaluator.EvaluationResult`.
        timestamp_col: Column name for the x-axis label.  When ``None`` the
            integer index is used.
        max_points: Downsample to at most this many points for readability
            (default: ``500``).
        title: Optional plot title; defaults to ``"<model_name>: Actual vs Predicted"``.

    Returns:
        :class:`matplotlib.figure.Figure`
    """
    plt, _ = _get_mpl()

    y_true = result.y_true.reset_index(drop=True)
    y_pred = result.predictions.reset_index(drop=True)

    if len(y_true) > max_points:
        step = max(1, len(y_true) // max_points)
        y_true = y_true.iloc[::step]
        y_pred = y_pred.iloc[::step]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(y_true.index, y_true.values, label="Actual", color=_PALETTE[0], linewidth=1.2)
    ax.plot(
        y_pred.index,
        y_pred.values,
        label="Predicted",
        color=_PALETTE[1],
        linewidth=1.0,
        linestyle="--",
        alpha=0.85,
    )
    ax.set_xlabel(timestamp_col or "Index")
    ax.set_ylabel("Bikes available")
    ax.set_title(title or f"{result.model_name}: Actual vs Predicted")
    ax.legend(framealpha=0.8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_residuals(
    result: "EvaluationResult",
    bins: int = 40,
    title: Optional[str] = None,
) -> "Figure":
    """Plot the residual (error) distribution for a single model.

    Displays a histogram with an overlaid KDE and vertical lines at ±1 MAE.

    Args:
        result: An :class:`~src.evaluation.evaluator.EvaluationResult`.
        bins: Number of histogram bins (default: ``40``).
        title: Optional plot title.

    Returns:
        :class:`matplotlib.figure.Figure`
    """
    plt, _ = _get_mpl()

    residuals = result.residuals.dropna()
    mae = result.metrics.get("mae", float("nan"))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(residuals, bins=bins, color=_PALETTE[0], alpha=0.7, density=True, label="Residuals")

    # KDE overlay via numpy
    if len(residuals) > 1:
        kde_x = np.linspace(residuals.min(), residuals.max(), 300)
        bw = 1.06 * residuals.std() * len(residuals) ** (-0.2)  # Silverman's rule
        kde_y = np.array(
            [
                np.mean(np.exp(-0.5 * ((x_val - residuals) / bw) ** 2) / (bw * np.sqrt(2 * np.pi)))
                for x_val in kde_x
            ]
        )
        ax.plot(kde_x, kde_y, color=_PALETTE[1], linewidth=2, label="KDE")

    if not np.isnan(mae):
        for sign, lbl in [(1, f"+MAE={mae:.2f}"), (-1, f"−MAE={mae:.2f}")]:
            ax.axvline(sign * mae, color=_PALETTE[2], linestyle="--", linewidth=1.2, label=lbl)

    ax.axvline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Residual (predicted − actual)")
    ax.set_ylabel("Density")
    ax.set_title(title or f"{result.model_name}: Residual Distribution")
    ax.legend(framealpha=0.8, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_metric_comparison(
    evaluator: "ModelEvaluator",
    metrics: Optional[list[str]] = None,
    title: str = "Model Comparison",
) -> "Figure":
    """Bar chart comparing all registered models across one or more metrics.

    Args:
        evaluator: A :class:`~src.evaluation.evaluator.ModelEvaluator` after
            :meth:`~src.evaluation.evaluator.ModelEvaluator.fit_evaluate` has
            been called.
        metrics: List of metrics to show.  Defaults to
            ``["mae", "rmse", "mape", "r2"]``.
        title: Figure suptitle.

    Returns:
        :class:`matplotlib.figure.Figure`
    """
    plt, _ = _get_mpl()

    if metrics is None:
        metrics = ["mae", "rmse", "mape", "r2"]

    comparison = evaluator.compare(sort_by="mae")
    n_metrics = len(metrics)
    n_cols = min(n_metrics, 2)
    n_rows = (n_metrics + 1) // 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), squeeze=False)
    colours = _PALETTE[: len(comparison)]

    for idx, metric in enumerate(metrics):
        ax = axes[idx // n_cols][idx % n_cols]
        values = comparison[metric].values
        bars = ax.bar(comparison["model"], values, color=colours, edgecolor="white", width=0.6)

        # Value labels on bars
        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.01,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        ax.set_title(metric.upper())
        ax.set_xlabel("")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.3)

    # Hide any unused subplot panels
    for idx in range(n_metrics, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_error_by_hour(
    evaluator: "ModelEvaluator",
    timestamp_col: str = "time",
    title: str = "Mean Absolute Error by Hour of Day",
) -> "Figure":
    """Line plot of mean absolute error broken down by hour of day.

    Useful for spotting systematic patterns (e.g. harder at rush hour).

    Args:
        evaluator: A :class:`~src.evaluation.evaluator.ModelEvaluator` after
            :meth:`~src.evaluation.evaluator.ModelEvaluator.fit_evaluate`.
        timestamp_col: Name of the timestamp column.  Must be present in
            ``y_true``'s index as a level named *timestamp_col*, or the default
            integer index is used to recover hours (falls back to empty plot).
        title: Plot title.

    Returns:
        :class:`matplotlib.figure.Figure`
    """
    plt, mticker = _get_mpl()

    fig, ax = plt.subplots(figsize=(9, 4))

    for idx, (name, result) in enumerate(evaluator.results.items()):
        # Attempt to extract hour from index
        try:
            times = pd.to_datetime(result.y_true.index.get_level_values(timestamp_col))
        except (KeyError, TypeError):
            logger.warning(  # type: ignore[attr-defined]
                "plot_error_by_hour: cannot extract '%s' from index for model '%s'; skipping.",
                timestamp_col,
                name,
            )
            continue

        abs_err = result.residuals.abs()
        hourly = (
            pd.DataFrame({"hour": times.hour, "abs_err": abs_err.values})
            .groupby("hour")["abs_err"]
            .mean()
        )
        ax.plot(
            hourly.index,
            hourly.values,
            marker="o",
            markersize=4,
            color=_PALETTE[idx % len(_PALETTE)],
            label=name,
            linewidth=1.5,
        )

    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(1))
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Mean absolute error")
    ax.set_title(title)
    ax.legend(framealpha=0.8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------


def plot_dashboard(
    evaluator: "ModelEvaluator",
    best_model_name: Optional[str] = None,
    timestamp_col: str = "time",
    title: str = "Model Evaluation Dashboard",
) -> "Figure":
    """2×2 composite dashboard with the four standard evaluation plots.

    Layout::

        ┌───────────────────────┬───────────────────────┐
        │  Actual vs Predicted  │  Metric Comparison    │
        │  (best / chosen model)│  (all models)         │
        ├───────────────────────┼───────────────────────┤
        │  Residual Distribution│  MAE by Hour of Day   │
        │  (best / chosen model)│  (all models)         │
        └───────────────────────┴───────────────────────┘

    Args:
        evaluator: A :class:`~src.evaluation.evaluator.ModelEvaluator` after
            :meth:`~src.evaluation.evaluator.ModelEvaluator.fit_evaluate`.
        best_model_name: Model to use for the per-model panels (top-left and
            bottom-left).  Defaults to the model with lowest MAE.
        timestamp_col: Timestamp column name for the hour-of-day plot.
        title: Figure suptitle.

    Returns:
        :class:`matplotlib.figure.Figure`
    """
    plt, mticker = _get_mpl()

    if best_model_name is None:
        best_model_name, _ = evaluator.best_model(metric="mae")

    best_result = evaluator.results[best_model_name]
    comparison = evaluator.compare(sort_by="mae")

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.40, wspace=0.30)

    # ── Top-left: actual vs predicted ──────────────────────────────────
    ax_pred = fig.add_subplot(gs[0, 0])
    y_true = best_result.y_true.reset_index(drop=True)
    y_pred = best_result.predictions.reset_index(drop=True)
    n = len(y_true)
    step = max(1, n // 500)
    ax_pred.plot(y_true.iloc[::step].values, label="Actual", color=_PALETTE[0], linewidth=1.0)
    ax_pred.plot(
        y_pred.iloc[::step].values,
        label="Predicted",
        color=_PALETTE[1],
        linewidth=0.9,
        linestyle="--",
        alpha=0.85,
    )
    ax_pred.set_title(f"{best_model_name}: Actual vs Predicted")
    ax_pred.set_xlabel("Index")
    ax_pred.set_ylabel("Bikes available")
    ax_pred.legend(fontsize=8, framealpha=0.8)
    ax_pred.grid(axis="y", alpha=0.3)

    # ── Top-right: metric comparison bar chart ─────────────────────────
    ax_bar = fig.add_subplot(gs[0, 1])
    models = comparison["model"].tolist()
    mae_vals = comparison["mae"].tolist()
    rmse_vals = comparison["rmse"].tolist()
    x = np.arange(len(models))
    w = 0.35
    bars1 = ax_bar.bar(x - w / 2, mae_vals, w, label="MAE", color=_PALETTE[0], edgecolor="white")
    bars2 = ax_bar.bar(x + w / 2, rmse_vals, w, label="RMSE", color=_PALETTE[1], edgecolor="white")
    for bars in (bars1, bars2):
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h):
                ax_bar.text(
                    bar.get_x() + bar.get_width() / 2,
                    h * 1.01,
                    f"{h:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(models, rotation=15, ha="right", fontsize=9)
    ax_bar.set_title("MAE & RMSE by Model")
    ax_bar.set_ylabel("Error")
    ax_bar.legend(fontsize=8, framealpha=0.8)
    ax_bar.grid(axis="y", alpha=0.3)

    # ── Bottom-left: residual distribution ─────────────────────────────
    ax_res = fig.add_subplot(gs[1, 0])
    residuals = best_result.residuals.dropna()
    mae_val = best_result.metrics.get("mae", float("nan"))
    ax_res.hist(residuals, bins=35, color=_PALETTE[0], alpha=0.7, density=True)
    if len(residuals) > 1:
        bw = 1.06 * residuals.std() * len(residuals) ** (-0.2)
        kde_x = np.linspace(residuals.min(), residuals.max(), 300)
        kde_y = np.array(
            [
                np.mean(np.exp(-0.5 * ((x_val - residuals) / bw) ** 2) / (bw * np.sqrt(2 * np.pi)))
                for x_val in kde_x
            ]
        )
        ax_res.plot(kde_x, kde_y, color=_PALETTE[1], linewidth=1.8)
    if not np.isnan(mae_val):
        ax_res.axvline(mae_val, color=_PALETTE[2], linestyle="--", linewidth=1.1)
        ax_res.axvline(-mae_val, color=_PALETTE[2], linestyle="--", linewidth=1.1)
    ax_res.axvline(0, color="black", linewidth=0.8, alpha=0.5)
    ax_res.set_title(f"{best_model_name}: Residual Distribution")
    ax_res.set_xlabel("Residual")
    ax_res.set_ylabel("Density")
    ax_res.grid(axis="y", alpha=0.3)

    # ── Bottom-right: error by hour ────────────────────────────────────
    ax_hour = fig.add_subplot(gs[1, 1])
    for idx, (name, result) in enumerate(evaluator.results.items()):
        try:
            times = pd.to_datetime(result.y_true.index.get_level_values(timestamp_col))
            abs_err = result.residuals.abs()
            hourly = (
                pd.DataFrame({"hour": times.hour, "abs_err": abs_err.values})
                .groupby("hour")["abs_err"]
                .mean()
            )
            ax_hour.plot(
                hourly.index,
                hourly.values,
                marker="o",
                markersize=3,
                color=_PALETTE[idx % len(_PALETTE)],
                label=name,
                linewidth=1.2,
            )
        except (KeyError, TypeError):
            pass

    ax_hour.xaxis.set_major_locator(mticker.MultipleLocator(4))
    ax_hour.set_xlabel("Hour of day")
    ax_hour.set_ylabel("MAE")
    ax_hour.set_title("MAE by Hour of Day")
    ax_hour.legend(fontsize=8, framealpha=0.8)
    ax_hour.grid(alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    return fig
