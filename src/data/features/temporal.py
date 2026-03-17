"""Temporal feature engineering for Velib station time series data.

Provides functions to enrich a DataFrame with time-based features:
  - Cyclical hour-of-day encoding (sin/cos)
  - Day of week, weekend and public holiday indicators
  - Rush hour indicators (Paris commute patterns)
  - Time elapsed since the previous observation
"""

import math
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

try:
    import holidays as holidays_lib

    HAS_HOLIDAYS = True
except ImportError:
    HAS_HOLIDAYS = False
    logger.warning("holidays package not available; is_holiday will always be 0")

# Paris rush-hour windows (inclusive, hour-of-day integers)
MORNING_RUSH_START = 7
MORNING_RUSH_END = 9
EVENING_RUSH_START = 17
EVENING_RUSH_END = 19


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_datetime_index(df: pd.DataFrame, timestamp_col: str) -> pd.DatetimeIndex:
    """Return a DatetimeIndex extracted from a column or the DataFrame index.

    Args:
        df: Input DataFrame.
        timestamp_col: Name of the datetime column to look for.

    Returns:
        A pandas DatetimeIndex aligned with the rows of *df*.

    Raises:
        ValueError: If *timestamp_col* is not found and the index is not a DatetimeIndex.
    """
    if timestamp_col in df.columns:
        return pd.DatetimeIndex(df[timestamp_col])
    if isinstance(df.index, pd.DatetimeIndex):
        return df.index
    raise ValueError(
        f"Column '{timestamp_col}' not found and the DataFrame index is not a DatetimeIndex."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def add_hour_features(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    """Add cyclical hour-of-day features using sin/cos encoding.

    Encodes the hour on a circle so that hour 23 and hour 0 are adjacent,
    which is important for distance-based models.

    Added columns:
        - ``hour``: integer hour of day (0–23)
        - ``hour_sin``: sin(2π * hour / 24)
        - ``hour_cos``: cos(2π * hour / 24)

    Args:
        df: Input DataFrame. Not modified in place.
        timestamp_col: Name of the datetime column (or ``"timestamp"``).

    Returns:
        A copy of *df* with the three new columns appended.

    Raises:
        ValueError: If the timestamp source cannot be determined.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({"timestamp": pd.to_datetime(["2024-01-08 06:00:00"])})
        >>> add_hour_features(df)[["hour", "hour_sin", "hour_cos"]]
           hour  hour_sin  hour_cos
        0     6       1.0       0.0
    """
    dt = _get_datetime_index(df, timestamp_col)
    result = df.copy()
    result["hour"] = dt.hour
    result["hour_sin"] = np.sin(2 * math.pi * dt.hour / 24)
    result["hour_cos"] = np.cos(2 * math.pi * dt.hour / 24)
    logger.debug("Added hour features: hour, hour_sin, hour_cos")
    return result


def add_day_features(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    country: str = "FR",
    years: Optional[list[int]] = None,
) -> pd.DataFrame:
    """Add day-of-week, weekend, and public holiday indicator features.

    Added columns:
        - ``day_of_week``: 0 = Monday … 6 = Sunday
        - ``is_weekend``: 1 if Saturday or Sunday, else 0
        - ``is_holiday``: 1 if the date is a public holiday in *country*, else 0

    Args:
        df: Input DataFrame. Not modified in place.
        timestamp_col: Name of the datetime column.
        country: ISO 3166-1 alpha-2 country code used for holiday lookup
            (default ``"FR"`` for France).
        years: List of years to pre-load holidays for. Auto-detected from the
            data when ``None``.

    Returns:
        A copy of *df* with the three new columns appended.

    Raises:
        ValueError: If the timestamp source cannot be determined.
    """
    dt = _get_datetime_index(df, timestamp_col)
    result = df.copy()

    result["day_of_week"] = dt.dayofweek
    result["is_weekend"] = (dt.dayofweek >= 5).astype(int)

    if HAS_HOLIDAYS:
        if years is None:
            years = list(dt.year.unique())
        country_holidays = holidays_lib.country_holidays(country, years=years)
        result["is_holiday"] = (
            pd.Series(dt.date, index=result.index).isin(country_holidays).astype(int)
        )
    else:
        result["is_holiday"] = 0

    logger.debug("Added day features: day_of_week, is_weekend, is_holiday")
    return result


def add_rush_hour_features(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    """Add rush hour indicator features for Paris commute patterns.

    Rush hour windows (inclusive):
        - Morning: 07:00–09:00
        - Evening: 17:00–19:00

    Added columns:
        - ``is_morning_rush``: 1 during the morning rush window, else 0
        - ``is_evening_rush``: 1 during the evening rush window, else 0
        - ``is_rush_hour``: 1 if either rush window is active, else 0

    Args:
        df: Input DataFrame. Not modified in place.
        timestamp_col: Name of the datetime column.

    Returns:
        A copy of *df* with the three new columns appended.

    Raises:
        ValueError: If the timestamp source cannot be determined.
    """
    dt = _get_datetime_index(df, timestamp_col)
    result = df.copy()

    hour = dt.hour
    result["is_morning_rush"] = ((hour >= MORNING_RUSH_START) & (hour <= MORNING_RUSH_END)).astype(
        int
    )
    result["is_evening_rush"] = ((hour >= EVENING_RUSH_START) & (hour <= EVENING_RUSH_END)).astype(
        int
    )
    result["is_rush_hour"] = (result["is_morning_rush"] | result["is_evening_rush"]).astype(int)

    logger.debug("Added rush hour features: is_morning_rush, is_evening_rush, is_rush_hour")
    return result


def add_time_since_last_observation(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    group_col: Optional[str] = None,
    output_col: str = "seconds_since_last_obs",
) -> pd.DataFrame:
    """Add the elapsed time (in seconds) since the previous observation.

    When *group_col* is provided (e.g. ``"station_id"``), the delta is
    computed independently per group so station boundaries are not crossed.
    The first row of each group will be ``NaN``.

    Args:
        df: Input DataFrame, expected to be sorted by *timestamp_col*
            (and by *group_col* if provided).
        timestamp_col: Name of the datetime column.
        group_col: Optional column to group by before computing the delta.
        output_col: Name of the output column (default
            ``"seconds_since_last_obs"``).

    Returns:
        A copy of *df* with *output_col* appended.

    Raises:
        ValueError: If the timestamp source cannot be determined.
    """
    result = df.copy()

    if timestamp_col in result.columns:
        ts = pd.to_datetime(result[timestamp_col])
    elif isinstance(result.index, pd.DatetimeIndex):
        ts = result.index.to_series().reset_index(drop=True)
        result = result.reset_index(drop=False)
    else:
        raise ValueError(
            f"Column '{timestamp_col}' not found and the DataFrame index is not a DatetimeIndex."
        )

    if group_col is not None:
        result[output_col] = ts.groupby(result[group_col]).diff().dt.total_seconds()
    else:
        result[output_col] = ts.diff().dt.total_seconds()

    logger.debug("Added time-since-last-observation feature: %s", output_col)
    return result


def add_all_temporal_features(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    country: str = "FR",
    group_col: Optional[str] = None,
) -> pd.DataFrame:
    """Apply all temporal feature transformations in a single call.

    Convenience wrapper that sequentially calls:
        1. :func:`add_hour_features`
        2. :func:`add_day_features`
        3. :func:`add_rush_hour_features`
        4. :func:`add_time_since_last_observation`

    Args:
        df: Input DataFrame. Not modified in place.
        timestamp_col: Name of the datetime column.
        country: Country code for holiday detection (default ``"FR"``).
        group_col: Optional column for per-group time-since-last-obs.

    Returns:
        A copy of *df* with all temporal features appended.
    """
    result = add_hour_features(df, timestamp_col)
    result = add_day_features(result, timestamp_col, country=country)
    result = add_rush_hour_features(result, timestamp_col)
    result = add_time_since_last_observation(result, timestamp_col, group_col=group_col)
    logger.info("All temporal features added successfully")
    return result
