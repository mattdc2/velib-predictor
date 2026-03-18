"""Linear regression baseline models for Velib demand forecasting.

Three model variants sharing a common interface:

* :class:`LinearModel`  — ordinary least-squares regression
* :class:`RidgeModel`   — L2-regularised regression (ridge)
* :class:`LassoModel`   — L1-regularised regression (lasso, sparse coefficients)

All three extend :class:`~src.models.base.VelibBaseModel` and therefore expose
the same ``fit / predict / evaluate`` contract.

Feature importance
------------------
Each model exposes :meth:`feature_importance` which returns a
:class:`pandas.DataFrame` of ``(feature, coefficient, abs_coefficient)`` rows
sorted by absolute magnitude — a transparent proxy for importance in linear
models.

Cross-validation
----------------
The module-level function :func:`cross_validate_model` runs time-series-aware
k-fold CV (via :class:`sklearn.model_selection.TimeSeriesSplit`) and returns
per-fold and aggregated metric summaries.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import Lasso, LinearRegression, Ridge  # type: ignore[import-untyped]
from sklearn.model_selection import TimeSeriesSplit  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

from src.models.base import VelibBaseModel
from src.models.baseline.persistence import compute_metrics

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_numpy(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract feature matrix and target vector, dropping NaN rows."""
    subset = df[feature_cols + [target_col]].dropna()
    X = subset[feature_cols].to_numpy(dtype=float)
    y = subset[target_col].to_numpy(dtype=float)
    return X, y


# ---------------------------------------------------------------------------
# Base mixin shared by all three linear variants
# ---------------------------------------------------------------------------


class _LinearBase(VelibBaseModel):
    """Internal mixin — not part of the public API.

    Provides shared logic for:
    - ``fit``  (scaling + sklearn estimator training)
    - ``predict``  (scaling + inference, NaN pass-through)
    - ``feature_importance``
    """

    def __init__(
        self,
        feature_cols: Optional[list[str]] = None,
        target_col: str = "num_bikes_available",
        scale_features: bool = True,
        **estimator_kwargs: Any,
    ) -> None:
        self.feature_cols = feature_cols  # resolved at fit time when None
        self.target_col = target_col
        self.scale_features = scale_features
        self._estimator_kwargs = estimator_kwargs

        self._estimator: Optional[LinearRegression | Ridge | Lasso] = None
        self._scaler: Optional[StandardScaler] = None
        self._fitted_feature_cols: list[str] = []

    # ------------------------------------------------------------------
    def _make_estimator(self) -> LinearRegression | Ridge | Lasso:
        raise NotImplementedError

    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> "_LinearBase":
        """Fit the scaler (optional) and linear estimator on *df*.

        When ``feature_cols`` was not provided at construction time every
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

        X, y = _to_numpy(df, self._fitted_feature_cols, self.target_col)

        if self.scale_features:
            self._scaler = StandardScaler()
            X = self._scaler.fit_transform(X)

        self._estimator = self._make_estimator()
        self._estimator.fit(X, y)

        logger.info(
            "%s fitted on %d samples, %d features.",
            type(self).__name__,
            len(y),
            len(self._fitted_feature_cols),
        )
        return self

    # ------------------------------------------------------------------
    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Generate predictions for every row in *df*.

        Rows with any NaN in the feature columns receive a NaN prediction
        rather than raising an error, so evaluation on partially-filled data
        degrades gracefully.

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

        # Only predict rows where all feature columns are present and not NaN
        available_features = [c for c in self._fitted_feature_cols if c in df.columns]
        valid_mask = df[available_features].notna().all(axis=1)

        if valid_mask.any():
            X = df.loc[valid_mask, available_features].to_numpy(dtype=float)
            if self._scaler is not None:
                X = self._scaler.transform(X)
            result.loc[valid_mask] = self._estimator.predict(X)

        return result

    # ------------------------------------------------------------------
    def feature_importance(self) -> pd.DataFrame:
        """Return model coefficients sorted by absolute magnitude.

        For linear models, coefficient magnitude is the direct measure of
        feature importance (after standard scaling, if enabled).

        Returns:
            DataFrame with columns ``["feature", "coefficient",
            "abs_coefficient"]``, sorted descending by ``abs_coefficient``.

        Raises:
            RuntimeError: If :meth:`fit` has not been called.
        """
        if self._estimator is None:
            raise RuntimeError("Call fit() before feature_importance().")

        coefs = np.asarray(self._estimator.coef_).flatten()
        importance = pd.DataFrame(
            {
                "feature": self._fitted_feature_cols,
                "coefficient": coefs,
                "abs_coefficient": np.abs(coefs),
            }
        ).sort_values("abs_coefficient", ascending=False)

        return importance.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Public model classes
# ---------------------------------------------------------------------------


class LinearModel(_LinearBase):
    """Ordinary least-squares linear regression.

    Args:
        feature_cols: Explicit list of feature columns.  Auto-detected from
            numeric columns when ``None``.
        target_col: Target variable column name.
        scale_features: Whether to z-score features before fitting
            (recommended; default: ``True``).
        **kwargs: Forwarded to :class:`sklearn.linear_model.LinearRegression`.
    """

    def _make_estimator(self) -> LinearRegression:
        return LinearRegression(**self._estimator_kwargs)


class RidgeModel(_LinearBase):
    """L2-regularised (ridge) linear regression.

    Args:
        alpha: Regularisation strength (default: ``1.0``).  Larger values
            penalise large coefficients more heavily.
        feature_cols: Explicit list of feature columns.
        target_col: Target variable column name.
        scale_features: Whether to z-score features (default: ``True``).
        **kwargs: Forwarded to :class:`sklearn.linear_model.Ridge`.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        feature_cols: Optional[list[str]] = None,
        target_col: str = "num_bikes_available",
        scale_features: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(feature_cols, target_col, scale_features, alpha=alpha, **kwargs)

    def _make_estimator(self) -> Ridge:
        return Ridge(**self._estimator_kwargs)


class LassoModel(_LinearBase):
    """L1-regularised (lasso) linear regression.

    Lasso tends to produce sparse solutions — irrelevant features are driven
    to exactly zero, making :meth:`feature_importance` directly interpretable
    as feature selection.

    Args:
        alpha: Regularisation strength (default: ``0.1``).
        feature_cols: Explicit list of feature columns.
        target_col: Target variable column name.
        scale_features: Whether to z-score features (default: ``True``).
        max_iter: Maximum solver iterations (default: ``10_000``).
        **kwargs: Forwarded to :class:`sklearn.linear_model.Lasso`.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        feature_cols: Optional[list[str]] = None,
        target_col: str = "num_bikes_available",
        scale_features: bool = True,
        max_iter: int = 10_000,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            feature_cols, target_col, scale_features, alpha=alpha, max_iter=max_iter, **kwargs
        )

    def _make_estimator(self) -> Lasso:
        return Lasso(**self._estimator_kwargs)


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------


def cross_validate_model(
    model_class: type[_LinearBase],
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "num_bikes_available",
    n_splits: int = 5,
    model_kwargs: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Time-series cross-validation for any linear model class.

    Uses :class:`sklearn.model_selection.TimeSeriesSplit` so that training
    folds always precede validation folds — no data leakage.

    Args:
        model_class: One of :class:`LinearModel`, :class:`RidgeModel`, or
            :class:`LassoModel`.
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
        y_pred = model.predict(val_df)
        y_true = val_df[target_col]
        metrics = compute_metrics(y_true, y_pred)
        fold_metrics.append(metrics)
        logger.debug(
            "CV fold %d — MAE: %.3f  RMSE: %.3f", fold + 1, metrics["mae"], metrics["rmse"]
        )

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
        "%s CV (%d folds) — MAE %.3f±%.3f  RMSE %.3f±%.3f",
        model_class.__name__,
        n_splits,
        summary["mean_mae"],
        summary["std_mae"],
        summary["mean_rmse"],
        summary["std_rmse"],
    )
    return summary
