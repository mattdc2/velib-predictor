"""Evaluation demo: synthetic dataset → multiple models → comparison.

Generates a realistic synthetic Velib dataset, engineers features, trains
several models from the baseline suite, and compares them using the
evaluation framework.

Run from the project root::

    uv run python scripts/run_evaluation_demo.py

Optional: pass --save-plots to write PNG files instead of displaying them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. Synthetic dataset
# ---------------------------------------------------------------------------

def make_synthetic_dataset(
    n_stations: int = 8,
    n_hours: int = 24 * 60,  # 60 days
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic multi-station Velib-like dataset.

    Each station has a capacity drawn at random.  The number of available
    bikes follows a cyclical daily pattern (rush-hour dips) plus station-
    specific offsets, weekday effects, and Gaussian noise.
    """
    rng = np.random.default_rng(seed)
    times = pd.date_range("2024-01-01", periods=n_hours, freq="1h")
    capacities = rng.integers(15, 45, size=n_stations)

    records = []
    for sid in range(n_stations):
        cap = capacities[sid]
        for i, ts in enumerate(times):
            hour = ts.hour
            dow = ts.dayofweek  # 0=Mon … 6=Sun
            is_weekend = int(dow >= 5)

            # Cyclical availability pattern: low at morning/evening rush
            daily_pattern = (
                0.55
                + 0.20 * np.cos(2 * np.pi * (hour - 14) / 24)   # afternoon peak
                - 0.15 * np.exp(-((hour - 8) ** 2) / 4)          # morning dip
                - 0.10 * np.exp(-((hour - 18) ** 2) / 3)         # evening dip
                + 0.05 * is_weekend                               # slightly fuller on weekends
            )
            station_bias = rng.normal(0, 0.05)                    # per-station shift
            noise = rng.normal(0, 0.06)

            occupancy = float(np.clip(daily_pattern + station_bias + noise, 0.05, 0.95))
            num_bikes = round(occupancy * cap)

            records.append(
                {
                    "time": ts,
                    "station_id": sid,
                    "station_capacity": cap,
                    "num_bikes_available": num_bikes,
                }
            )

    df = pd.DataFrame(records).sort_values(["time", "station_id"]).reset_index(drop=True)
    print(f"Dataset: {len(df):,} rows  |  {n_stations} stations  |  {n_hours} hours")
    return df


# ---------------------------------------------------------------------------
# 2. Feature engineering
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add temporal, cyclical, lag, and rolling features in-place."""
    df = df.copy()
    ts = pd.to_datetime(df["time"])

    # Temporal
    df["hour"] = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["month"] = ts.dt.month

    # Cyclical encodings
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # Lag and rolling features per station
    df = df.sort_values(["station_id", "time"])
    grp = df.groupby("station_id")["num_bikes_available"]
    df["lag_1h"] = grp.shift(1)
    df["lag_2h"] = grp.shift(2)
    df["lag_24h"] = grp.shift(24)
    df["rolling_mean_3h"] = grp.transform(lambda s: s.shift(1).rolling(3).mean())
    df["rolling_mean_6h"] = grp.transform(lambda s: s.shift(1).rolling(6).mean())

    # Capacity normalised occupancy lag
    df["occupancy_lag_1h"] = df["lag_1h"] / df["station_capacity"].clip(lower=1)

    return df.sort_values(["time", "station_id"]).reset_index(drop=True)


FEATURE_COLS = [
    "station_capacity",
    "hour", "day_of_week", "is_weekend", "month",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "lag_1h", "lag_2h", "lag_24h",
    "rolling_mean_3h", "rolling_mean_6h",
    "occupancy_lag_1h",
]

TARGET = "num_bikes_available"


# ---------------------------------------------------------------------------
# 3. Model registration
# ---------------------------------------------------------------------------

def build_evaluator(feature_cols: list[str]) -> "ModelEvaluator":  # type: ignore[name-defined]
    from src.evaluation.evaluator import ModelEvaluator
    from src.models.baseline.linear import LassoModel, LinearModel, RidgeModel
    from src.models.baseline.persistence import HistoricalAverageModel, PersistenceModel
    from src.models.baseline.tree_models import LightGBMModel, PerStationTreeModel, XGBoostModel

    ev = ModelEvaluator(target_col=TARGET)

    ev.register("persistence",     PersistenceModel(value_col=TARGET, timestamp_col="time"))
    ev.register("hist_avg_hourly", HistoricalAverageModel(value_col=TARGET, timestamp_col="time"))
    ev.register("hist_avg_dow",    HistoricalAverageModel(value_col=TARGET, timestamp_col="time",
                                                          use_day_of_week=True))
    ev.register("linear",          LinearModel(feature_cols=feature_cols, target_col=TARGET))
    ev.register("ridge",           RidgeModel(feature_cols=feature_cols, target_col=TARGET))
    ev.register("lasso",           LassoModel(feature_cols=feature_cols, target_col=TARGET))
    ev.register("xgboost",         XGBoostModel(feature_cols=feature_cols, target_col=TARGET,
                                                 n_estimators=200, max_depth=5, learning_rate=0.05))
    ev.register("lightgbm",        LightGBMModel(feature_cols=feature_cols, target_col=TARGET,
                                                  n_estimators=200, num_leaves=31))
    ev.register("xgb_per_station", PerStationTreeModel(
        XGBoostModel, feature_cols=feature_cols, target_col=TARGET,
        model_kwargs={"n_estimators": 100, "max_depth": 4},
    ))

    return ev


# ---------------------------------------------------------------------------
# 4. Pretty-print comparison table
# ---------------------------------------------------------------------------

def print_comparison(ev: "ModelEvaluator") -> None:  # type: ignore[name-defined]
    df = ev.compare(sort_by="mae")
    best_name, _ = ev.best_model(metric="mae")

    col_w = max(len(n) for n in df["model"]) + 2
    header = f"{'Model':<{col_w}}  {'MAE':>7}  {'RMSE':>7}  {'MAPE%':>7}  {'R²':>7}"
    sep = "─" * len(header)

    print(f"\n{sep}")
    print(" Model Comparison — test-set metrics (sorted by MAE)")
    print(sep)
    print(header)
    print(sep)
    for _, row in df.iterrows():
        marker = " ◀ best" if row["model"] == best_name else ""
        print(
            f"{row['model']:<{col_w}}  "
            f"{row['mae']:>7.3f}  "
            f"{row['rmse']:>7.3f}  "
            f"{row['mape']:>7.2f}  "
            f"{row['r2']:>7.3f}"
            f"{marker}"
        )
    print(sep)
    print(f"\nTrain rows: {df['train_size'].iloc[0]:,}   Test rows: {df['test_size'].iloc[0]:,}\n")


# ---------------------------------------------------------------------------
# 5. Feature importance summary
# ---------------------------------------------------------------------------

def print_feature_importance(ev: "ModelEvaluator", model_name: str = "xgboost") -> None:  # type: ignore[name-defined]
    from src.models.baseline.tree_models import XGBoostModel, LightGBMModel, PerStationTreeModel

    result = ev.results.get(model_name)
    if result is None:
        return

    model = ev._registry[model_name]
    if not hasattr(model, "feature_importance"):
        return

    fi = model.feature_importance().head(10)
    print(f"Top-10 features  [{model_name}]")
    print(f"{'Feature':<25}  {'Importance':>10}")
    print("─" * 38)
    for _, row in fi.iterrows():
        print(f"  {row['feature']:<23}  {row['importance']:>10.4f}")
    print()


# ---------------------------------------------------------------------------
# 6. Plots
# ---------------------------------------------------------------------------

def make_plots(ev: "ModelEvaluator", save: bool, out_dir: Path) -> None:  # type: ignore[name-defined]
    from src.evaluation.visualizer import (
        plot_dashboard,
        plot_metric_comparison,
        plot_predictions,
    )

    best_name, _ = ev.best_model(metric="mae")

    figures = {
        "dashboard": plot_dashboard(ev, title="Velib Evaluation Dashboard"),
        "metric_comparison": plot_metric_comparison(ev, title="Model Comparison"),
        "predictions_best": plot_predictions(ev.results[best_name],
                                              title=f"{best_name}: Actual vs Predicted"),
    }

    if save:
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, fig in figures.items():
            path = out_dir / f"{name}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"  Saved: {path}")
    else:
        try:
            import matplotlib.pyplot as plt
            plt.show()
        except Exception:
            print("  (display unavailable; use --save-plots to write PNG files)")

    import matplotlib.pyplot as plt
    for fig in figures.values():
        plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Velib model evaluation demo")
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save plots to plots/ instead of displaying them",
    )
    parser.add_argument("--n-stations", type=int, default=8)
    parser.add_argument("--n-hours",    type=int, default=24 * 60)
    args = parser.parse_args()

    # ── Step 1: data ──────────────────────────────────────────────────
    print("\n[1/5] Generating synthetic dataset …")
    raw_df = make_synthetic_dataset(n_stations=args.n_stations, n_hours=args.n_hours)

    # ── Step 2: features ──────────────────────────────────────────────
    print("[2/5] Engineering features …")
    df = engineer_features(raw_df)
    available_features = [c for c in FEATURE_COLS if c in df.columns]
    n_dropped = df[available_features + [TARGET]].isna().any(axis=1).sum()
    df = df.dropna(subset=available_features + [TARGET]).reset_index(drop=True)
    print(f"       Features: {len(available_features)}  |  Rows after NaN drop: {len(df):,}  "
          f"({n_dropped} dropped)")

    # ── Step 3: split ─────────────────────────────────────────────────
    print("[3/5] Splitting dataset (70 / 15 / 15) …")
    from src.evaluation.evaluator import temporal_split
    split = temporal_split(df, train_frac=0.70, val_frac=0.15)

    # ── Step 4: train & evaluate ──────────────────────────────────────
    print("[4/5] Training and evaluating models …")
    ev = build_evaluator(available_features)
    ev.fit_evaluate(split)

    # ── Step 5: results ───────────────────────────────────────────────
    print("[5/5] Results\n")
    print_comparison(ev)
    print_feature_importance(ev, model_name="xgboost")
    print_feature_importance(ev, model_name="xgb_per_station")

    print("Generating plots …")
    make_plots(ev, save=args.save_plots, out_dir=Path("plots"))

    sys.exit(0)


if __name__ == "__main__":
    main()
