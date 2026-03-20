"""Tree-based regression models for Velib demand forecasting.

Two gradient-boosted tree variants sharing a common interface:

* :class:`XGBoostModel`  — XGBoost gradient boosting
* :class:`LightGBMModel` — LightGBM gradient boosting (faster, leaf-wise growth)

Per-station training:

* :class:`PerStationTreeModel` — wraps any tree model class to train one
  estimator per station, with automatic fallback to a global model for unseen
  stations or stations with insufficient training data.

Utilities:

* :func:`tune_hyperparams`              — manual grid search with
  :class:`~sklearn.model_selection.TimeSeriesSplit` CV
* :func:`cross_validate_tree_model`     — time-series k-fold CV (mirrors
  :func:`~src.models.baseline.linear.cross_validate_model`)
* :func:`compare_per_station_vs_global` — side-by-side evaluation of the two
  training strategies on a temporal hold-out split
"""

from __future__ import annotations

import itertools
from typing import Any, Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import TimeSeriesSplit  # type: ignore[import-untyped]

from src.models.base import VelibBaseModel
from src.models.baseline.persistence import compute_metrics

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_feature_matrix(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract feature matrix and target vector, dropping rows with any NaN."""
    subset = df[feature_cols + [target_col]].dropna()
    X = subset[feature_cols].to_numpy(dtype=float)
    y = subset[target_col].to_numpy(dtype=float)
    return X, y


# ---------------------------------------------------------------------------
# Shared base for XGBoost and LightGBM
# ---------------------------------------------------------------------------


class _TreeBase(VelibBaseModel):
    """Internal mixin — not part of the public API.

    Provides shared logic for:
    - ``fit``               (NaN-dropping + estimator training)
    - ``predict``           (NaN-safe, returns NaN for rows with missing features)
    - ``feature_importance``  (gain-based importances from ``feature_importances_``)
    """

    def __init__(
        self,
        feature_cols: Optional[list[str]] = None,
        target_col: str = "num_bikes_available",
        **estimator_kwargs: Any,
    ) -> None:
        self.feature_cols = feature_cols
        self.target_col = target_col
        self._estimator_kwargs = estimator_kwargs

        self._estimator: Any = None
        self._fitted_feature_cols: list[str] = []

    # ------------------------------------------------------------------
    def _make_estimator(self) -> Any:
        raise NotImplementedError

    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> "_TreeBase":
        """Fit the tree estimator on *df*.

        When ``feature_cols`` was not provided at construction time, every
        numeric column except *target_col* is used.

        Args:
            df: Training DataFrame.

        Returns:
            *self*
        """
        if self.feature_cols is None:
            self._fitted_feature_cols = [
                c for c in df.select_dtypes(include=[np.number]).columns if c != self.target_col
            ]
        else:
            self._fitted_feature_cols = list(self.feature_cols)

        X, y = _to_feature_matrix(df, self._fitted_feature_cols, self.target_col)
        self._estimator = self._make_estimator()
        self._estimator.fit(X, y)

        logger.info(
            f"{type(self).__name__} fitted on {len(y)} samples, {len(self._fitted_feature_cols)} features."
        )
        return self

    # ------------------------------------------------------------------
    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Generate predictions for every row in *df*.

        Rows with any NaN in the feature columns receive a NaN prediction
        rather than raising an error.

        Args:
            df: DataFrame to predict on.

        Returns:
            Series named ``"prediction"`` aligned with ``df.index``.

        Raises:
            RuntimeError: If :meth:`fit` has not been called.
        """
        if self._estimator is None:
            raise RuntimeError("Call fit() before predict().")

        result = pd.Series(np.nan, index=df.index, name="prediction", dtype=float)

        available = [c for c in self._fitted_feature_cols if c in df.columns]
        valid_mask = df[available].notna().all(axis=1)

        if valid_mask.any():
            X = df.loc[valid_mask, available].to_numpy(dtype=float)
            result.loc[valid_mask] = self._estimator.predict(X)

        return result

    # ------------------------------------------------------------------
    def feature_importance(self) -> pd.DataFrame:
        """Return gain-based feature importances sorted descending.

        Tree models compute importance as the total gain attributed to each
        feature across all splits.

        Returns:
            DataFrame with columns ``["feature", "importance"]``, sorted
            descending by ``importance``.

        Raises:
            RuntimeError: If :meth:`fit` has not been called.
        """
        if self._estimator is None:
            raise RuntimeError("Call fit() before feature_importance().")

        importances = np.asarray(self._estimator.feature_importances_).flatten()
        return (
            pd.DataFrame({"feature": self._fitted_feature_cols, "importance": importances})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )


# ---------------------------------------------------------------------------
# Public model classes
# ---------------------------------------------------------------------------


class XGBoostModel(_TreeBase):
    """XGBoost gradient boosting regressor.

    Args:
        n_estimators: Number of boosting rounds (default: ``100``).
        max_depth: Maximum tree depth (default: ``6``).
        learning_rate: Step-size shrinkage after each round (default: ``0.1``).
        subsample: Row sub-sampling ratio per tree (default: ``0.8``).
        colsample_bytree: Feature sub-sampling ratio per tree (default: ``0.8``).
        feature_cols: Explicit list of feature columns. Auto-detected from
            numeric columns when ``None``.
        target_col: Target variable column name.
        **kwargs: Forwarded to :class:`xgboost.XGBRegressor`.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        feature_cols: Optional[list[str]] = None,
        target_col: str = "num_bikes_available",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            feature_cols=feature_cols,
            target_col=target_col,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            **kwargs,
        )

    def _make_estimator(self) -> Any:
        from xgboost import XGBRegressor  # type: ignore[import-untyped]

        return XGBRegressor(random_state=42, verbosity=0, **self._estimator_kwargs)


class LightGBMModel(_TreeBase):
    """LightGBM gradient boosting regressor.

    LightGBM uses leaf-wise (best-first) tree growth instead of level-wise,
    which tends to converge faster and use less memory on large datasets.

    Args:
        n_estimators: Number of boosting rounds (default: ``100``).
        max_depth: Maximum tree depth; ``-1`` means no explicit limit
            (default: ``-1``).
        learning_rate: Step-size shrinkage (default: ``0.1``).
        num_leaves: Maximum number of leaves per tree (default: ``31``).
        subsample: Row sub-sampling ratio per tree (default: ``0.8``).
        feature_cols: Explicit list of feature columns. Auto-detected when
            ``None``.
        target_col: Target variable column name.
        **kwargs: Forwarded to :class:`lightgbm.LGBMRegressor`.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = -1,
        learning_rate: float = 0.1,
        num_leaves: int = 31,
        subsample: float = 0.8,
        feature_cols: Optional[list[str]] = None,
        target_col: str = "num_bikes_available",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            feature_cols=feature_cols,
            target_col=target_col,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            subsample=subsample,
            **kwargs,
        )

    def _make_estimator(self) -> Any:
        from lightgbm import LGBMRegressor  # type: ignore[import-untyped]

        return LGBMRegressor(random_state=42, verbose=-1, **self._estimator_kwargs)


# ---------------------------------------------------------------------------
# Per-station model
# ---------------------------------------------------------------------------


class PerStationTreeModel(VelibBaseModel):
    """Train one tree model per station, with global-model fallback.

    Stations that have fewer than *min_train_samples* rows in the training set
    — or that are completely unseen at prediction time — are handled by a
    single global model trained on all stations.

    Args:
        model_class: Tree model class to instantiate — either
            :class:`XGBoostModel` or :class:`LightGBMModel`.
        station_col: Column identifying each station.
        target_col: Target variable column name.
        feature_cols: Explicit feature columns. Auto-detected when ``None``.
        min_train_samples: Minimum training rows for a per-station model
            (default: ``30``).
        model_kwargs: Extra keyword arguments forwarded to every *model_class*
            instance.
    """

    def __init__(
        self,
        model_class: type[_TreeBase],
        station_col: str = "station_id",
        target_col: str = "num_bikes_available",
        feature_cols: Optional[list[str]] = None,
        min_train_samples: int = 30,
        model_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        self.model_class = model_class
        self.station_col = station_col
        self.target_col = target_col
        self.feature_cols = feature_cols
        self.min_train_samples = min_train_samples
        self.model_kwargs = model_kwargs or {}

        self._station_models: dict[Any, _TreeBase] = {}
        self._global_model: Optional[_TreeBase] = None
        self._fitted_feature_cols: list[str] = []

    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> "PerStationTreeModel":
        """Fit a global fallback model and one model per qualifying station.

        Args:
            df: Training DataFrame.

        Returns:
            *self*
        """
        if self.feature_cols is None:
            self._fitted_feature_cols = [
                c for c in df.select_dtypes(include=[np.number]).columns if c != self.target_col
            ]
        else:
            self._fitted_feature_cols = list(self.feature_cols)

        # Global fallback — trained on all data
        self._global_model = self.model_class(
            feature_cols=self._fitted_feature_cols,
            target_col=self.target_col,
            **self.model_kwargs,
        ).fit(df)

        trained, skipped = 0, 0
        for station_id, station_df in df.groupby(self.station_col):
            if len(station_df) >= self.min_train_samples:
                self._station_models[station_id] = self.model_class(
                    feature_cols=self._fitted_feature_cols,
                    target_col=self.target_col,
                    **self.model_kwargs,
                ).fit(station_df)
                trained += 1
            else:
                skipped += 1

        logger.info(
            f"PerStationTreeModel fitted: {trained} per-station models, {skipped} used global fallback (min_train_samples={self.min_train_samples})."
        )
        return self

    # ------------------------------------------------------------------
    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Predict using the per-station model when available, else global.

        Args:
            df: DataFrame to forecast.

        Returns:
            Series named ``"prediction"`` aligned with ``df.index``.

        Raises:
            RuntimeError: If :meth:`fit` has not been called.
        """
        if self._global_model is None:
            raise RuntimeError("Call fit() before predict().")

        result = pd.Series(np.nan, index=df.index, name="prediction", dtype=float)

        for station_id, station_df in df.groupby(self.station_col):
            model = self._station_models.get(station_id, self._global_model)
            preds = model.predict(station_df)
            result.loc[station_df.index] = preds.values

        return result

    # ------------------------------------------------------------------
    def feature_importance(self) -> pd.DataFrame:
        """Return mean feature importance averaged over all per-station models.

        Falls back to the global model's importance when no per-station models
        were trained.

        Returns:
            DataFrame with columns ``["feature", "importance"]``, sorted
            descending by ``importance``.

        Raises:
            RuntimeError: If :meth:`fit` has not been called.
        """
        if self._global_model is None:
            raise RuntimeError("Call fit() before feature_importance().")

        if not self._station_models:
            return self._global_model.feature_importance()

        all_fi = [m.feature_importance() for m in self._station_models.values()]
        combined = pd.concat(all_fi, ignore_index=True)
        return (
            combined.groupby("feature")["importance"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )


# ---------------------------------------------------------------------------
# Hyperparameter tuning
# ---------------------------------------------------------------------------


def tune_hyperparams(
    model_class: type[_TreeBase],
    df: pd.DataFrame,
    feature_cols: list[str],
    param_grid: dict[str, list[Any]],
    target_col: str = "num_bikes_available",
    n_splits: int = 3,
) -> dict[str, Any]:
    """Grid search over hyperparameters using time-series cross-validation.

    All combinations in *param_grid* are evaluated with
    :class:`~sklearn.model_selection.TimeSeriesSplit`.  The combination with
    the lowest mean MAE across folds is returned as ``"best_params"``.

    Args:
        model_class: :class:`XGBoostModel` or :class:`LightGBMModel`.
        df: Full dataset (must already be sorted by time).
        feature_cols: Feature columns passed to the model.
        param_grid: Mapping from parameter name to list of candidate values,
            e.g. ``{"n_estimators": [50, 100], "max_depth": [3, 6]}``.
        target_col: Target column name.
        n_splits: Number of CV folds (default: ``3``).

    Returns:
        A dict with keys:

        * ``"best_params"``  — dict of the best hyperparameter combination
        * ``"best_score"``   — mean MAE of the best combination
        * ``"all_results"``  — list of ``{"params", "mean_mae", "fold_maes"}``
          dicts for every combination tried
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    indices = np.arange(len(df))

    param_names = list(param_grid.keys())
    combinations = list(itertools.product(*param_grid.values()))

    all_results: list[dict[str, Any]] = []
    best_score = float("inf")
    best_params: dict[str, Any] = {}

    for combo in combinations:
        params = dict(zip(param_names, combo))
        fold_maes: list[float] = []

        for train_idx, val_idx in tscv.split(indices):
            train_df = df.iloc[train_idx]
            val_df = df.iloc[val_idx]

            model = model_class(feature_cols=feature_cols, target_col=target_col, **params)
            model.fit(train_df)
            metrics = compute_metrics(val_df[target_col], model.predict(val_df))
            fold_maes.append(metrics["mae"])

        mean_mae = float(np.nanmean(fold_maes))
        all_results.append({"params": params, "mean_mae": mean_mae, "fold_maes": fold_maes})

        logger.debug(f"tune_hyperparams: params={params} → mean_mae={mean_mae:.4f}")

        if mean_mae < best_score:
            best_score = mean_mae
            best_params = params

    logger.info(
        f"tune_hyperparams: best={best_params} (mean_mae={best_score:.4f}) over {len(combinations)} combinations."
    )
    return {"best_params": best_params, "best_score": best_score, "all_results": all_results}


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------


def cross_validate_tree_model(
    model_class: type[_TreeBase],
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "num_bikes_available",
    n_splits: int = 5,
    model_kwargs: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Time-series cross-validation for any tree model class.

    Uses :class:`~sklearn.model_selection.TimeSeriesSplit` so that training
    folds always precede validation folds — no data leakage.

    Args:
        model_class: :class:`XGBoostModel` or :class:`LightGBMModel`.
        df: Full dataset (must already be sorted by time).
        feature_cols: Feature columns to pass to the model.
        target_col: Target column name.
        n_splits: Number of CV folds (default: ``5``).
        model_kwargs: Extra keyword arguments forwarded to *model_class*.

    Returns:
        A dict with keys:

        * ``"fold_metrics"`` — list of per-fold ``{mae, rmse, mape}`` dicts
        * ``"mean_mae"``  / ``"std_mae"``
        * ``"mean_rmse"`` / ``"std_rmse"``
        * ``"mean_mape"`` / ``"std_mape"``
    """
    kwargs = model_kwargs or {}
    tscv = TimeSeriesSplit(n_splits=n_splits)
    indices = np.arange(len(df))

    fold_metrics: list[dict[str, float]] = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(indices)):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]

        model = model_class(feature_cols=feature_cols, target_col=target_col, **kwargs)
        model.fit(train_df)
        metrics = compute_metrics(val_df[target_col], model.predict(val_df))
        fold_metrics.append(metrics)
        logger.debug(f"CV fold {fold + 1} — MAE: {metrics['mae']:.3f}  RMSE: {metrics['rmse']:.3f}")

    maes = [m["mae"] for m in fold_metrics]
    rmses = [m["rmse"] for m in fold_metrics]
    mapes = [m["mape"] for m in fold_metrics]

    summary: dict[str, Any] = {
        "fold_metrics": fold_metrics,
        "mean_mae": float(np.nanmean(maes)),
        "std_mae": float(np.nanstd(maes)),
        "mean_rmse": float(np.nanmean(rmses)),
        "std_rmse": float(np.nanstd(rmses)),
        "mean_mape": float(np.nanmean(mapes)),
        "std_mape": float(np.nanstd(mapes)),
    }

    logger.info(
        f"{model_class.__name__} CV ({n_splits} folds) — MAE {summary['mean_mae']:.3f}±{summary['std_mae']:.3f}  RMSE {summary['mean_rmse']:.3f}±{summary['std_rmse']:.3f}"
    )
    return summary


# ---------------------------------------------------------------------------
# Per-station vs global comparison
# ---------------------------------------------------------------------------


def compare_per_station_vs_global(
    model_class: type[_TreeBase],
    df: pd.DataFrame,
    feature_cols: list[str],
    station_col: str = "station_id",
    target_col: str = "num_bikes_available",
    test_fraction: float = 0.2,
    model_kwargs: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Evaluate and compare per-station vs global tree models.

    The dataset is split temporally: the first ``(1 - test_fraction)`` rows
    are used for training and the remainder for evaluation.  Both a global
    model (all stations, single estimator) and a
    :class:`PerStationTreeModel` are trained on the same split.

    Args:
        model_class: :class:`XGBoostModel` or :class:`LightGBMModel`.
        df: Full dataset (must be sorted by time).
        feature_cols: Feature columns.
        station_col: Station identifier column.
        target_col: Target column name.
        test_fraction: Fraction of rows to hold out as the test set
            (default: ``0.2``).
        model_kwargs: Extra kwargs forwarded to every model instance.

    Returns:
        A dict with keys:

        * ``"global_metrics"``      — ``{mae, rmse, mape}`` for the global model
        * ``"per_station_metrics"`` — ``{mae, rmse, mape}`` for the per-station model
        * ``"station_comparison"``  — :class:`pandas.DataFrame` with per-station
          ``{station_id, global_mae, per_station_mae, mae_delta, n_test_rows}``
          sorted by ``mae_delta`` (negative = per-station wins)
        * ``"winner"``              — ``"global"`` or ``"per_station"`` (lower MAE)
    """
    kwargs = model_kwargs or {}
    split_idx = int(len(df) * (1 - test_fraction))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    # Global model
    global_model = model_class(feature_cols=feature_cols, target_col=target_col, **kwargs).fit(
        train_df
    )
    global_metrics = compute_metrics(test_df[target_col], global_model.predict(test_df))

    # Per-station model
    per_station = PerStationTreeModel(
        model_class=model_class,
        station_col=station_col,
        target_col=target_col,
        feature_cols=feature_cols,
        model_kwargs=kwargs,
    ).fit(train_df)
    per_station_metrics = compute_metrics(test_df[target_col], per_station.predict(test_df))

    # Per-station breakdown
    rows = []
    for station_id, sdf in test_df.groupby(station_col):
        g_m = compute_metrics(sdf[target_col], global_model.predict(sdf))
        ps_model = per_station._station_models.get(station_id, per_station._global_model)
        ps_m = compute_metrics(sdf[target_col], ps_model.predict(sdf))  # type: ignore[union-attr]
        rows.append(
            {
                "station_id": station_id,
                "global_mae": g_m["mae"],
                "per_station_mae": ps_m["mae"],
                "mae_delta": ps_m["mae"] - g_m["mae"],  # negative means per-station wins
                "n_test_rows": len(sdf),
            }
        )

    station_comparison = pd.DataFrame(rows).sort_values("mae_delta").reset_index(drop=True)
    winner = "per_station" if per_station_metrics["mae"] < global_metrics["mae"] else "global"

    logger.info(
        f"compare_per_station_vs_global: global MAE={global_metrics['mae']:.3f}, per-station MAE={per_station_metrics['mae']:.3f} → {winner} wins."
    )
    return {
        "global_metrics": global_metrics,
        "per_station_metrics": per_station_metrics,
        "station_comparison": station_comparison,
        "winner": winner,
    }
