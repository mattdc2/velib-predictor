"""Lagged and rolling feature engineering for Velib station time series data.

Provides functions to enrich a station-status DataFrame with history-based features:
  - Lag features  — value at t-1 … t-N (previous observations)
  - Rolling means — mean over 1 h, 2 h, 6 h, 24 h time windows
  - Capacity gap  — difference between station capacity and current bikes
  - Rate of change — absolute and percentage change since the prior observation

All functions expect a DataFrame with one row per (station, timestamp) pair,
sorted or sortable by ``station_col`` and ``timestamp_col``.  They return a
**copy** of the input — the original is never modified.
"""

from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

# Default column names matching the station_status database schema
_DEFAULT_VALUE_COL = "num_bikes_available"
_DEFAULT_STATION_COL = "station_id"
_DEFAULT_TIMESTAMP_COL = "time"
_DEFAULT_DOCKS_COL = "num_docks_available"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def add_lag_features(
    df: pd.DataFrame,
    value_col: str = _DEFAULT_VALUE_COL,
    lags: tuple[int, ...] = (1, 2, 3, 4),
    station_col: str = _DEFAULT_STATION_COL,
    timestamp_col: str = _DEFAULT_TIMESTAMP_COL,
) -> pd.DataFrame:
    """Add lagged observations of *value_col* for each station.

    Each lag N produces the value observed N steps before the current row
    (within each station group).  The first N rows of each station will be
    ``NaN`` for lag N.

    Added columns: ``{value_col}_lag_{N}`` for each N in *lags*.

    Args:
        df: Input DataFrame with one row per (station, timestamp). Not modified
            in place.
        value_col: Column whose lagged values to compute.
        lags: Lag steps to compute (default: ``(1, 2, 3, 4)`` — last four
            observations, i.e. approximately the last hour at 15-min cadence).
        station_col: Station identifier column.
        timestamp_col: Timestamp column.

    Returns:
        A copy of *df* with ``len(lags)`` new columns appended, in the same
        row order as the input.

    Raises:
        ValueError: If *value_col* or *station_col* are not present in *df*.
    """
    for col in (value_col, station_col, timestamp_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    result = df.sort_values([station_col, timestamp_col]).copy()

    for lag in lags:
        result[f"{value_col}_lag_{lag}"] = result.groupby(station_col)[value_col].shift(lag)
        logger.debug(f"Added lag feature: {value_col}_lag_{lag}")

    return result.sort_index()


def add_rolling_means(
    df: pd.DataFrame,
    value_col: str = _DEFAULT_VALUE_COL,
    windows: tuple[str, ...] = ("1h", "2h", "6h", "24h"),
    station_col: str = _DEFAULT_STATION_COL,
    timestamp_col: str = _DEFAULT_TIMESTAMP_COL,
    min_periods: int = 1,
) -> pd.DataFrame:
    """Add time-based rolling mean features per station.

    Uses offset-string windows (e.g. ``"1h"``) so the window size adapts
    correctly to irregular sampling intervals.  The rolling window is
    **closed on the left** (i.e. the current observation is excluded from its
    own rolling mean, giving a purely historical feature).

    Added columns: ``{value_col}_rolling_mean_{window}`` for each window.

    Args:
        df: Input DataFrame with one row per (station, timestamp). Not modified
            in place.
        value_col: Column to compute rolling means for.
        windows: Time-based window sizes as offset strings (default:
            ``("1h", "2h", "6h", "24h")``).
        station_col: Station identifier column.
        timestamp_col: Timestamp column (must be convertible to
            ``pd.Timestamp``).
        min_periods: Minimum number of observations required to produce a
            non-NaN result (default: 1).

    Returns:
        A copy of *df* with ``len(windows)`` new columns appended, in the same
        row order as the input.

    Raises:
        ValueError: If required columns are not present in *df*.
    """
    for col in (value_col, station_col, timestamp_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    result = df.copy()
    result[timestamp_col] = pd.to_datetime(result[timestamp_col])
    result = result.sort_values([station_col, timestamp_col])

    # Set timestamp as index: required for time-based rolling windows
    ts_indexed = result.set_index(timestamp_col)

    for window in windows:
        col_name = f"{value_col}_rolling_mean_{window}"
        means = (
            ts_indexed.groupby(station_col)[value_col]
            .rolling(window, min_periods=min_periods, closed="left")
            .mean()
            .reset_index(level=0, drop=True)  # drop station_id from MultiIndex
        )
        # means is aligned with result (both sorted by station + timestamp)
        result[col_name] = means.values
        logger.debug(f"Added rolling mean feature: {col_name}")

    return result.sort_index()


def add_capacity_gap(
    df: pd.DataFrame,
    value_col: str = _DEFAULT_VALUE_COL,
    capacity_col: Optional[str] = None,
    docks_col: str = _DEFAULT_DOCKS_COL,
    gap_col: str = "capacity_gap",
    fill_rate_col: str = "fill_rate",
) -> pd.DataFrame:
    """Add the gap between station capacity and current bike count.

    Two features are produced:

    - **capacity_gap** — ``capacity - bikes_available`` (number of empty slots)
    - **fill_rate** — ``bikes_available / capacity`` (fraction in [0, 1])

    If *capacity_col* is provided (e.g. from a join with ``station_information``),
    it is used directly.  Otherwise capacity is derived as
    ``bikes_available + docks_available``, which equals the total number of
    functional dock slots at the time of observation.

    Args:
        df: Input DataFrame. Not modified in place.
        value_col: Column holding the current bike count.
        capacity_col: Optional column holding the station's design capacity.
            When ``None``, capacity is derived from *value_col* + *docks_col*.
        docks_col: Column holding available docks (used when *capacity_col* is
            ``None``).
        gap_col: Name of the gap output column.
        fill_rate_col: Name of the fill-rate output column.

    Returns:
        A copy of *df* with *gap_col* and *fill_rate_col* appended.

    Raises:
        ValueError: If required columns are not present in *df*.
    """
    required = [value_col]
    if capacity_col is not None:
        required.append(capacity_col)
    else:
        required.append(docks_col)
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    result = df.copy()

    if capacity_col is not None:
        capacity = result[capacity_col].astype(float)
    else:
        capacity = (result[value_col] + result[docks_col]).astype(float)

    result[gap_col] = capacity - result[value_col].astype(float)
    # Guard against zero-capacity stations to avoid division by zero
    result[fill_rate_col] = result[value_col].astype(float) / capacity.replace(0, np.nan)

    logger.debug(f"Added capacity gap features: {gap_col}, {fill_rate_col}")
    return result


def add_rate_of_change(
    df: pd.DataFrame,
    value_col: str = _DEFAULT_VALUE_COL,
    station_col: str = _DEFAULT_STATION_COL,
    timestamp_col: str = _DEFAULT_TIMESTAMP_COL,
    diff_col: Optional[str] = None,
    pct_change_col: Optional[str] = None,
) -> pd.DataFrame:
    """Add absolute and percentage rate-of-change features per station.

    Both features are computed between a row and the **immediately preceding**
    observation for the same station (after sorting by timestamp).

    Added columns:

    - ``{value_col}_diff``       — ``value[t] - value[t-1]`` (absolute change)
    - ``{value_col}_pct_change`` — ``(value[t] - value[t-1]) / |value[t-1]|``
      (signed percentage; ``NaN`` when the previous value is zero)

    The first row of each station group will be ``NaN`` for both features.

    Args:
        df: Input DataFrame with one row per (station, timestamp). Not modified
            in place.
        value_col: Column to compute the rate of change for.
        station_col: Station identifier column.
        timestamp_col: Timestamp column.
        diff_col: Override name for the absolute-change output column.
        pct_change_col: Override name for the percentage-change output column.

    Returns:
        A copy of *df* with two new columns appended, in the same row order as
        the input.

    Raises:
        ValueError: If required columns are not present in *df*.
    """
    for col in (value_col, station_col, timestamp_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    _diff_col = diff_col or f"{value_col}_diff"
    _pct_col = pct_change_col or f"{value_col}_pct_change"

    result = df.sort_values([station_col, timestamp_col]).copy()

    grouped = result.groupby(station_col)[value_col]
    result[_diff_col] = grouped.diff()

    prev = grouped.shift(1)
    result[_pct_col] = result[_diff_col] / prev.abs().replace(0, np.nan)

    logger.debug(f"Added rate-of-change features: {_diff_col}, {_pct_col}")
    return result.sort_index()


def add_all_lagged_features(
    df: pd.DataFrame,
    value_col: str = _DEFAULT_VALUE_COL,
    lags: tuple[int, ...] = (1, 2, 3, 4),
    windows: tuple[str, ...] = ("1h", "2h", "6h", "24h"),
    station_col: str = _DEFAULT_STATION_COL,
    timestamp_col: str = _DEFAULT_TIMESTAMP_COL,
    capacity_col: Optional[str] = None,
    docks_col: str = _DEFAULT_DOCKS_COL,
    min_periods: int = 1,
) -> pd.DataFrame:
    """Apply all lagged feature transformations in a single call.

    Convenience wrapper that sequentially calls:
        1. :func:`add_lag_features`
        2. :func:`add_rolling_means`
        3. :func:`add_capacity_gap`
        4. :func:`add_rate_of_change`

    Args:
        df: Input DataFrame with one row per (station, timestamp). Not modified
            in place.
        value_col: Primary numeric column (e.g. ``"num_bikes_available"``).
        lags: Lag steps for :func:`add_lag_features`.
        windows: Time windows for :func:`add_rolling_means`.
        station_col: Station identifier column.
        timestamp_col: Timestamp column.
        capacity_col: Optional explicit capacity column; if ``None``, capacity
            is derived from *docks_col*.
        docks_col: Available-docks column (used when *capacity_col* is
            ``None``).
        min_periods: Minimum periods for rolling means.

    Returns:
        A copy of *df* with all lagged features appended.
    """
    result = add_lag_features(df, value_col, lags, station_col, timestamp_col)
    result = add_rolling_means(result, value_col, windows, station_col, timestamp_col, min_periods)
    result = add_capacity_gap(result, value_col, capacity_col, docks_col)
    result = add_rate_of_change(result, value_col, station_col, timestamp_col)
    logger.info("All lagged features added successfully")
    return result
