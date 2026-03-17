"""Persistence and historical-average baseline models for Velib forecasting.

Two naïve baselines that establish a performance floor:

* :class:`PersistenceModel`  — "tomorrow = today": predicts that the next
  observation equals the most recent one (naïve persistence / random-walk
  forecast).

* :class:`HistoricalAverageModel` — predicts the per-station average bikes
  available for a given hour (and optionally day-of-week), computed from
  the training set.

Both expose the same ``fit / predict / evaluate`` interface defined in
:class:`~src.models.base.VelibBaseModel`.

Standalone helper
-----------------
:func:`compute_metrics` can be used independently to evaluate any pair of
true / predicted series.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.models.base import VelibBaseModel

# ---------------------------------------------------------------------------
# Standalone metric computation
# ---------------------------------------------------------------------------

_MAPE_EPS = 1.0  # avoids ÷0 for empty stations; natural unit = 1 bike


def compute_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    mape_eps: float = _MAPE_EPS,
) -> dict[str, float]:
    """Compute MAE, RMSE and MAPE between two aligned series.

    MAPE uses ``max(|y_true|, mape_eps)`` as the denominator to avoid
    division by zero when a station is empty.  For integer bike counts the
    natural floor of 1 bike is a sensible default.

    Args:
        y_true: Ground-truth observations.
        y_pred: Model predictions, aligned with *y_true*.
        mape_eps: Minimum denominator value for MAPE (default: 1.0).

    Returns:
        ``{"mae": float, "rmse": float, "mape": float}`` where MAPE is
        expressed as a percentage.

    Raises:
        ValueError: If *y_true* and *y_pred* have different lengths.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred must have the same length " f"({len(y_true)} vs {len(y_pred)})."
        )

    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    # Ignore rows where either value is NaN (e.g. first row of each station
    # for lag-based models).
    mask = ~(np.isnan(y_true_arr) | np.isnan(y_pred_arr))
    if mask.sum() == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "mape": float("nan")}

    yt = y_true_arr[mask]
    yp = y_pred_arr[mask]
    err = yp - yt

    mae = float(np.mean(np.abs(err)))
    rmse = float(math.sqrt(np.mean(err**2)))
    mape = float(np.mean(np.abs(err) / np.maximum(np.abs(yt), mape_eps)) * 100)

    logger.debug("Metrics — MAE: %.3f  RMSE: %.3f  MAPE: %.2f%%", mae, rmse, mape)
    return {"mae": mae, "rmse": rmse, "mape": mape}


# ---------------------------------------------------------------------------
# PersistenceModel
# ---------------------------------------------------------------------------


class PersistenceModel(VelibBaseModel):
    """Naïve persistence baseline: predict ``t+1 = t``.

    At prediction time the model uses ``{value_col}_lag_1`` when that column
    is present in the input DataFrame (i.e. the feature pipeline has already
    been applied).  Otherwise it derives the lag by sorting each station's
    rows by timestamp and shifting by one observation.

    ``fit()`` stores only the last observed value per station so that the
    very first prediction for each station in a new batch is not ``NaN``.

    Args:
        value_col: Column holding the target variable.
        station_col: Column identifying each station.
        timestamp_col: Column holding the observation timestamp.
    """

    def __init__(
        self,
        value_col: str = "num_bikes_available",
        station_col: str = "station_id",
        timestamp_col: str = "time",
    ) -> None:
        self.value_col = value_col
        self.station_col = station_col
        self.timestamp_col = timestamp_col
        # Populated by fit(): maps station_id → last known value
        self._last_known: dict[str | int, float] = {}

    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> "PersistenceModel":
        """Memorise the last observed value for each station.

        Args:
            df: Training DataFrame.

        Returns:
            *self*
        """
        last = (
            df.sort_values([self.station_col, self.timestamp_col])
            .groupby(self.station_col)[self.value_col]
            .last()
        )
        self._last_known = dict(zip(last.index.astype(object), last.values.astype(float)))
        logger.info(
            "PersistenceModel fitted on %d rows, %d stations.",
            len(df),
            len(self._last_known),
        )
        return self

    # ------------------------------------------------------------------
    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Return the last-known value as the forecast for every row.

        Prediction logic (in order of preference):

        1. If ``{value_col}_lag_1`` exists in *df*, use it directly.
        2. Otherwise sort by ``(station_col, timestamp_col)`` and compute
           the lag in-place; the first row of each station is filled from
           ``self._last_known`` (learned during ``fit``).

        Args:
            df: DataFrame to forecast.

        Returns:
            Series named ``"prediction"`` aligned with ``df.index``.
        """
        lag_col = f"{self.value_col}_lag_1"

        if lag_col in df.columns:
            predictions = df[lag_col].rename("prediction")
            logger.debug("PersistenceModel: using pre-computed %s column.", lag_col)
            return predictions

        # Derive lag on the fly
        sorted_df = df.sort_values([self.station_col, self.timestamp_col]).copy()
        sorted_df["prediction"] = sorted_df.groupby(self.station_col)[self.value_col].shift(1)

        # Fill the first NaN of each station from the training set
        if self._last_known:
            first_per_station_mask = sorted_df["prediction"].isna()
            sorted_df.loc[first_per_station_mask, "prediction"] = sorted_df.loc[
                first_per_station_mask, self.station_col
            ].map(self._last_known)

        return sorted_df["prediction"].reindex(df.index)


# ---------------------------------------------------------------------------
# HistoricalAverageModel
# ---------------------------------------------------------------------------


class HistoricalAverageModel(VelibBaseModel):
    """Forecast using the per-station historical average for the same hour.

    The model builds a lookup table mapping ``(station_id, hour)`` — or
    optionally ``(station_id, hour, day_of_week)`` — to the mean bikes
    available observed during training.  Unknown combinations fall back to the
    per-station mean, then to the global mean.

    Args:
        value_col: Column holding the target variable.
        station_col: Column identifying each station.
        timestamp_col: Column holding the observation timestamp.
        use_day_of_week: When ``True``, group by ``(station, hour,
            day_of_week)`` for finer-grained averages.
    """

    def __init__(
        self,
        value_col: str = "num_bikes_available",
        station_col: str = "station_id",
        timestamp_col: str = "time",
        use_day_of_week: bool = False,
    ) -> None:
        self.value_col = value_col
        self.station_col = station_col
        self.timestamp_col = timestamp_col
        self.use_day_of_week = use_day_of_week

        self._lookup: Optional[pd.DataFrame] = None  # hourly averages
        self._station_mean: Optional[pd.Series] = None  # fallback per station
        self._global_mean: float = float("nan")  # ultimate fallback

    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> "HistoricalAverageModel":
        """Compute per-station hourly (and optionally daily) averages.

        Args:
            df: Training DataFrame.

        Returns:
            *self*
        """
        ts = pd.to_datetime(df[self.timestamp_col])
        work = df[[self.station_col, self.value_col]].copy()
        work["_hour"] = ts.dt.hour

        group_keys = [self.station_col, "_hour"]
        if self.use_day_of_week:
            work["_dow"] = ts.dt.dayofweek
            group_keys.append("_dow")

        self._lookup = work.groupby(group_keys)[self.value_col].mean().rename("_avg").reset_index()
        self._station_mean = df.groupby(self.station_col)[self.value_col].mean()
        self._global_mean = float(df[self.value_col].mean())

        logger.info(
            "HistoricalAverageModel fitted — %d lookup entries, global mean %.2f.",
            len(self._lookup),
            self._global_mean,
        )
        return self

    # ------------------------------------------------------------------
    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Look up the historical average for each row's station and hour.

        Falls back to per-station mean, then to global mean, for unseen
        combinations.

        Args:
            df: DataFrame to forecast.

        Returns:
            Series named ``"prediction"`` aligned with ``df.index``.

        Raises:
            RuntimeError: If :meth:`fit` has not been called yet.
        """
        if self._lookup is None or self._station_mean is None:
            raise RuntimeError("Call fit() before predict().")

        ts = pd.to_datetime(df[self.timestamp_col])
        work = df[[self.station_col]].copy()
        work.index = df.index
        work["_hour"] = ts.values  # keep as datetime temporarily
        work["_hour"] = ts.dt.hour

        merge_keys = [self.station_col, "_hour"]
        if self.use_day_of_week:
            work["_dow"] = ts.dt.dayofweek
            merge_keys.append("_dow")

        merged = work.merge(self._lookup, on=merge_keys, how="left")
        merged.index = df.index

        predictions = merged["_avg"].copy()

        # Station-level fallback for unknown (station, hour) pairs
        unknown_mask = predictions.isna()
        if unknown_mask.any():
            predictions.loc[unknown_mask] = df.loc[unknown_mask, self.station_col].map(
                self._station_mean
            )

        # Global fallback for entirely unseen stations
        predictions = predictions.fillna(self._global_mean)

        return predictions.rename("prediction")
