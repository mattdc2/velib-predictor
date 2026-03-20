"""Standardized evaluation framework for Velib forecasting models.

This module provides:

* :func:`compute_metrics`  — MAE, RMSE, MAPE and R² from any pair of series
* :func:`temporal_split`   — reproducible train / val / test splits by time
* :class:`EvaluationResult` — typed container for one model's evaluation output
* :class:`ModelEvaluator`   — register models, run evaluation, compare results

Typical usage
-------------
::

    from src.evaluation.evaluator import ModelEvaluator, temporal_split
    from src.models.baseline.persistence import PersistenceModel
    from src.models.baseline.linear import RidgeModel
    from src.models.baseline.tree_models import XGBoostModel

    split = temporal_split(df, train_frac=0.7, val_frac=0.15)

    evaluator = ModelEvaluator(target_col="num_bikes_available")
    evaluator.register("persistence", PersistenceModel())
    evaluator.register("ridge", RidgeModel())
    evaluator.register("xgboost", XGBoostModel())

    results = evaluator.fit_evaluate(split)
    print(evaluator.compare())        # DataFrame sorted by MAE
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.models.base import VelibBaseModel

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

_MAPE_EPS = 1.0  # avoids ÷0 for empty stations; natural unit = 1 bike


def compute_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    mape_eps: float = _MAPE_EPS,
) -> dict[str, float]:
    """Compute MAE, RMSE, MAPE and R² between two aligned series.

    MAPE uses ``max(|y_true|, mape_eps)`` as denominator to avoid division by
    zero for empty stations.  R² is the coefficient of determination; it can be
    negative when the model is worse than predicting the mean.

    Args:
        y_true: Ground-truth observations.
        y_pred: Model predictions, aligned with *y_true*.
        mape_eps: Minimum denominator value for MAPE (default: ``1.0``).

    Returns:
        ``{"mae": float, "rmse": float, "mape": float, "r2": float}``
        All values are ``nan`` when no valid (non-NaN) pairs exist.

    Raises:
        ValueError: If *y_true* and *y_pred* have different lengths.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred must have the same length ({len(y_true)} vs {len(y_pred)})."
        )

    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)

    mask = ~(np.isnan(yt) | np.isnan(yp))
    if mask.sum() == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "mape": float("nan"), "r2": float("nan")}

    yt, yp = yt[mask], yp[mask]
    err = yp - yt

    mae = float(np.mean(np.abs(err)))
    rmse = float(math.sqrt(np.mean(err**2)))
    mape = float(np.mean(np.abs(err) / np.maximum(np.abs(yt), mape_eps)) * 100)

    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {"mae": mae, "rmse": rmse, "mape": mape, "r2": r2}


# ---------------------------------------------------------------------------
# Temporal split
# ---------------------------------------------------------------------------


@dataclass
class TemporalSplit:
    """Result of :func:`temporal_split`.

    Attributes:
        train: Training portion of the data.
        val: Validation portion (may be empty when ``val_frac=0``).
        test: Test (hold-out) portion.
        train_frac: Fraction used for training.
        val_frac: Fraction used for validation.
        test_frac: Fraction used for testing.
    """

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    train_frac: float
    val_frac: float
    test_frac: float

    @property
    def train_val(self) -> pd.DataFrame:
        """Combined train + validation set (for final re-fit before test evaluation)."""
        return pd.concat([self.train, self.val], ignore_index=True)


def temporal_split(
    df: pd.DataFrame,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: Optional[float] = None,
) -> TemporalSplit:
    """Split *df* into contiguous train / val / test portions.

    The split is purely positional — it preserves the row order of *df* and
    does **not** shuffle, ensuring no future information leaks into training.

    Args:
        df: Input DataFrame (should be sorted by time before calling).
        train_frac: Fraction of rows allocated to training (default: ``0.7``).
        val_frac: Fraction of rows allocated to validation (default: ``0.15``).
        test_frac: Fraction for the test set.  When ``None`` it is inferred as
            ``1 - train_frac - val_frac``.

    Returns:
        A :class:`TemporalSplit` with ``train``, ``val``, and ``test`` DataFrames.

    Raises:
        ValueError: If the fractions do not sum to approximately 1.0 or are
            individually out of ``[0, 1]``.
    """
    if test_frac is None:
        test_frac = 1.0 - train_frac - val_frac

    total = train_frac + val_frac + test_frac
    if not math.isclose(total, 1.0, abs_tol=1e-6):
        raise ValueError(f"train_frac + val_frac + test_frac must sum to 1.0, got {total:.6f}.")
    for name, frac in [
        ("train_frac", train_frac),
        ("val_frac", val_frac),
        ("test_frac", test_frac),
    ]:
        if not (0.0 <= frac <= 1.0):
            raise ValueError(f"{name} must be in [0, 1], got {frac}.")

    n = len(df)
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)

    train = df.iloc[:train_end].reset_index(drop=True)
    val = df.iloc[train_end:val_end].reset_index(drop=True)
    test = df.iloc[val_end:].reset_index(drop=True)

    logger.info(
        f"temporal_split: total={n}  train={len(train)} ({train_frac * 100:.0f}%)  val={len(val)} ({val_frac * 100:.0f}%)  test={len(test)} ({test_frac * 100:.0f}%)"
    )
    return TemporalSplit(
        train=train,
        val=val,
        test=test,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
    )


# ---------------------------------------------------------------------------
# Evaluation result container
# ---------------------------------------------------------------------------


@dataclass
class EvaluationResult:
    """Evaluation output for a single model on a single split.

    Attributes:
        model_name: Human-readable label for the model.
        metrics: ``{"mae", "rmse", "mape", "r2"}`` on the *test* set.
        predictions: Predicted values aligned with *y_true*.
        y_true: Ground-truth values from the test set.
        train_size: Number of training rows seen during ``fit``.
        test_size: Number of test rows evaluated.
    """

    model_name: str
    metrics: dict[str, float]
    predictions: pd.Series
    y_true: pd.Series
    train_size: int
    test_size: int

    @property
    def residuals(self) -> pd.Series:
        """Signed prediction errors (predicted − actual)."""
        return (self.predictions - self.y_true).rename("residual")

    def metrics_series(self) -> pd.Series:
        """Return metrics as a named :class:`pandas.Series` for easy concatenation."""
        return pd.Series(self.metrics, name=self.model_name)


# ---------------------------------------------------------------------------
# ModelEvaluator
# ---------------------------------------------------------------------------


class ModelEvaluator:
    """Register, fit, evaluate and compare multiple models on a common split.

    Models are fitted on the **training** portion of a :class:`TemporalSplit`
    and evaluated on the **test** portion.  The validation set is available on
    the split object itself for hyperparameter selection prior to final evaluation.

    Args:
        target_col: Column holding the ground-truth target values.
        fit_on_train_val: When ``True``, models are fitted on the combined
            train + validation set before test evaluation (useful after
            hyperparameters have been selected via the validation set).
            Default: ``False`` (fit on train only).

    Example::

        evaluator = ModelEvaluator()
        evaluator.register("persistence", PersistenceModel())
        evaluator.register("xgb", XGBoostModel(feature_cols=FEATURES))
        results = evaluator.fit_evaluate(split)
        print(evaluator.compare())
    """

    def __init__(
        self,
        target_col: str = "num_bikes_available",
        fit_on_train_val: bool = False,
    ) -> None:
        self.target_col = target_col
        self.fit_on_train_val = fit_on_train_val

        self._registry: dict[str, VelibBaseModel] = {}
        self._results: dict[str, EvaluationResult] = {}

    # ------------------------------------------------------------------
    def register(self, name: str, model: VelibBaseModel) -> "ModelEvaluator":
        """Register a model under *name*.

        Args:
            name: Unique label used in comparison tables and plots.
            model: An unfitted (or pre-fitted) :class:`~src.models.base.VelibBaseModel`.

        Returns:
            *self* — enables chaining multiple ``register`` calls.

        Raises:
            ValueError: If *name* is already registered.
        """
        if name in self._registry:
            raise ValueError(f"A model named '{name}' is already registered. Use a unique name.")
        self._registry[name] = model
        logger.debug(f"ModelEvaluator: registered '{name}'.")
        return self

    # ------------------------------------------------------------------
    def fit_evaluate(self, split: TemporalSplit) -> dict[str, EvaluationResult]:
        """Fit every registered model and evaluate it on the test set.

        Args:
            split: A :class:`TemporalSplit` produced by :func:`temporal_split`.

        Returns:
            Mapping from model name to :class:`EvaluationResult`.  The same
            dict is also stored in :attr:`results` for later access.

        Raises:
            RuntimeError: If no models have been registered.
        """
        if not self._registry:
            raise RuntimeError("No models registered. Call register() first.")

        train_df = split.train_val if self.fit_on_train_val else split.train
        test_df = split.test

        self._results = {}
        for name, model in self._registry.items():
            logger.info(f"ModelEvaluator: fitting '{name}' on {len(train_df)} rows …")
            model.fit(train_df)

            y_pred = model.predict(test_df)
            y_true = test_df[self.target_col]
            metrics = compute_metrics(y_true, y_pred)

            self._results[name] = EvaluationResult(
                model_name=name,
                metrics=metrics,
                predictions=y_pred,
                y_true=y_true,
                train_size=len(train_df),
                test_size=len(test_df),
            )
            logger.info(
                f"ModelEvaluator: '{name}' — MAE={metrics['mae']:.3f}  RMSE={metrics['rmse']:.3f}  R²={metrics['r2']:.3f}"
            )

        return self._results

    # ------------------------------------------------------------------
    @property
    def results(self) -> dict[str, EvaluationResult]:
        """Evaluation results from the last :meth:`fit_evaluate` call."""
        return self._results

    # ------------------------------------------------------------------
    def compare(
        self,
        sort_by: str = "mae",
        ascending: bool = True,
    ) -> pd.DataFrame:
        """Return a DataFrame comparing all models by their test-set metrics.

        Args:
            sort_by: Metric column to sort by (default: ``"mae"``).
            ascending: Sort ascending when ``True`` (default).  Set to
                ``False`` to rank by R² (higher is better).

        Returns:
            DataFrame with one row per model and columns
            ``["model", "mae", "rmse", "mape", "r2", "train_size", "test_size"]``,
            sorted by *sort_by*.

        Raises:
            RuntimeError: If :meth:`fit_evaluate` has not been called yet.
        """
        if not self._results:
            raise RuntimeError("No results available. Call fit_evaluate() first.")

        rows = []
        for name, result in self._results.items():
            rows.append(
                {
                    "model": name,
                    **result.metrics,
                    "train_size": result.train_size,
                    "test_size": result.test_size,
                }
            )

        comparison = pd.DataFrame(rows).sort_values(sort_by, ascending=ascending)
        return comparison.reset_index(drop=True)

    # ------------------------------------------------------------------
    def best_model(self, metric: str = "mae") -> tuple[str, EvaluationResult]:
        """Return the name and result of the best model by *metric*.

        For MAE/RMSE/MAPE lower is better; for R² higher is better.

        Args:
            metric: One of ``"mae"``, ``"rmse"``, ``"mape"``, ``"r2"``.

        Returns:
            ``(name, EvaluationResult)`` of the winning model.

        Raises:
            RuntimeError: If :meth:`fit_evaluate` has not been called.
            ValueError: If *metric* is not recognised.
        """
        if not self._results:
            raise RuntimeError("No results available. Call fit_evaluate() first.")
        if metric not in ("mae", "rmse", "mape", "r2"):
            raise ValueError(f"Unknown metric '{metric}'. Choose from mae, rmse, mape, r2.")

        higher_is_better = metric == "r2"
        best_name = min(
            self._results,
            key=lambda n: (
                -self._results[n].metrics[metric]
                if higher_is_better
                else self._results[n].metrics[metric]
            ),
        )
        return best_name, self._results[best_name]

    # ------------------------------------------------------------------
    def per_station_metrics(
        self,
        station_col: str = "station_id",
    ) -> pd.DataFrame:
        """Break down test-set MAE/RMSE per station for every model.

        Useful for identifying which stations are hardest to predict.

        Args:
            station_col: Column identifying each station in the test data.

        Returns:
            Long-format DataFrame with columns
            ``["station_id", "model", "mae", "rmse", "mape", "r2", "n_rows"]``.

        Raises:
            RuntimeError: If :meth:`fit_evaluate` has not been called.
        """
        if not self._results:
            raise RuntimeError("No results available. Call fit_evaluate() first.")

        rows = []
        first_result = next(iter(self._results.values()))
        # Reconstruct test df index from y_true (station_col lives in the index or test_df)
        y_true_index = first_result.y_true.index

        for name, result in self._results.items():
            # We need station information — it must have been retained in y_true's index
            # or be accessible via result.y_true.  We attempt a best-effort lookup.
            try:
                stations = result.y_true.index.get_level_values(station_col)
            except KeyError:
                logger.warning(
                    f"per_station_metrics: '{station_col}' column not in index; set df index before passing to fit_evaluate."
                )
                return pd.DataFrame()

            combined = pd.DataFrame(
                {"y_true": result.y_true.values, "y_pred": result.predictions.values},
                index=y_true_index,
            )
            combined[station_col] = stations

            for station_id, grp in combined.groupby(station_col):
                m = compute_metrics(
                    pd.Series(grp["y_true"].values),
                    pd.Series(grp["y_pred"].values),
                )
                rows.append(
                    {
                        station_col: station_id,
                        "model": name,
                        **m,
                        "n_rows": len(grp),
                    }
                )

        return pd.DataFrame(rows)
