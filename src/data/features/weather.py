"""Weather feature engineering for Velib demand forecasting.

Provides functions to enrich a weather DataFrame with model-ready features:
  - Weather lag features  — key columns shifted 1 h and 2 h back in time
  - Rain indicator        — binary flag from precipitation threshold
  - Temperature category  — ordered label (cold / mild / warm / hot)
  - Comfort index         — scalar score capturing cycling comfort

Column names default to those produced by
:class:`~src.data.weather_collector.WeatherData` / the ``weather_data``
database table.
"""

from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

# ---------------------------------------------------------------------------
# Domain constants
# ---------------------------------------------------------------------------

#: Precipitation threshold (mm/h) above which it is considered to be raining.
RAIN_THRESHOLD_MM: float = 0.1

#: Temperature bin edges (°C) for cold / mild / warm / hot categories.
TEMP_BINS: list[float] = [-np.inf, 5.0, 15.0, 25.0, np.inf]
TEMP_LABELS: list[str] = ["cold", "mild", "warm", "hot"]

#: Wind speed (km/h) below which no wind penalty is applied to the comfort index.
COMFORT_WIND_FREE_KMH: float = 10.0
#: Comfort penalty per km/h of wind above the free threshold.
COMFORT_WIND_COEFF: float = 0.3
#: Comfort penalty per mm/h of precipitation.
COMFORT_RAIN_COEFF: float = 2.0

# Default column names matching the WeatherData model / weather_data table.
_DEFAULT_TIMESTAMP_COL = "time"
_DEFAULT_TEMP_COL = "temperature"
_DEFAULT_APPARENT_TEMP_COL = "apparent_temperature"
_DEFAULT_RAIN_COL = "rain"
_DEFAULT_PRECIP_COL = "precipitation"
_DEFAULT_WIND_SPEED_COL = "wind_speed"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def add_weather_lags(
    df: pd.DataFrame,
    lag_cols: list[str],
    lags_hours: tuple[int, ...] = (1, 2),
    timestamp_col: str = _DEFAULT_TIMESTAMP_COL,
    tolerance_minutes: int = 30,
) -> pd.DataFrame:
    """Add lagged values of weather columns at specified hour offsets.

    Uses :func:`pandas.merge_asof` so the lookup is correct even when the
    weather time series has gaps (e.g. missing hourly records).  A lag value
    is set to ``NaN`` when no record falls within *tolerance_minutes* of the
    expected lag time.

    Added columns: ``{col}_lag_{N}h`` for each col in *lag_cols* and each N in
    *lags_hours*.

    Args:
        df: Input weather DataFrame with one row per timestamp. Not modified in
            place.
        lag_cols: Columns whose lagged values to compute.
        lags_hours: Hour offsets to compute (default: ``(1, 2)``).
        timestamp_col: Timestamp column name.
        tolerance_minutes: Maximum deviation (minutes) from the exact lag time
            that is still considered a valid match.

    Returns:
        A copy of *df* with ``len(lag_cols) × len(lags_hours)`` new columns
        appended, in the original row order.

    Raises:
        ValueError: If *timestamp_col* or any column in *lag_cols* is not
            present in *df*.
    """
    if timestamp_col not in df.columns:
        raise ValueError(f"Timestamp column '{timestamp_col}' not found in DataFrame.")
    missing = [c for c in lag_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in DataFrame: {missing}")

    result = df.copy()
    result[timestamp_col] = pd.to_datetime(result[timestamp_col])
    sorted_result = result.sort_values(timestamp_col)

    tolerance = pd.Timedelta(minutes=tolerance_minutes)

    for h in lags_hours:
        lag_delta = pd.Timedelta(hours=h)

        # Build a lookup table: shift each timestamp *forward* by h so that
        # merge_asof at time t finds the original record at time t - h.
        lookup = sorted_result[[timestamp_col] + lag_cols].copy()
        lookup[timestamp_col] = lookup[timestamp_col] + lag_delta

        merged = pd.merge_asof(
            sorted_result[[timestamp_col]],
            lookup,
            on=timestamp_col,
            tolerance=tolerance,
            direction="nearest",
        )

        for col in lag_cols:
            out_col = f"{col}_lag_{h}h"
            sorted_result[out_col] = merged[col].values
            logger.debug(f"Added weather lag feature: {out_col}")

    return sorted_result.sort_index()


def add_is_raining(
    df: pd.DataFrame,
    rain_col: str = _DEFAULT_RAIN_COL,
    precipitation_col: str = _DEFAULT_PRECIP_COL,
    threshold: float = RAIN_THRESHOLD_MM,
    output_col: str = "is_raining",
) -> pd.DataFrame:
    """Add a binary rain indicator column.

    The flag is ``1`` when either *rain_col* or *precipitation_col* (whichever
    is present) exceeds *threshold* mm/h.  If both columns are present, the
    flag is ``1`` when **either** exceeds the threshold.

    Args:
        df: Input DataFrame. Not modified in place.
        rain_col: Column holding liquid rain in mm/h.
        precipitation_col: Column holding total precipitation in mm/h.
        threshold: Minimum mm/h value to classify as raining (default: 0.1).
        output_col: Name of the output column.

    Returns:
        A copy of *df* with *output_col* appended (integer 0/1).

    Raises:
        ValueError: If neither *rain_col* nor *precipitation_col* is present.
    """
    available = [c for c in (rain_col, precipitation_col) if c in df.columns]
    if not available:
        raise ValueError(f"Neither '{rain_col}' nor '{precipitation_col}' found in DataFrame.")

    result = df.copy()
    raining = pd.Series(False, index=result.index)
    for col in available:
        raining = raining | (result[col] > threshold)

    result[output_col] = raining.astype(int)
    logger.debug(f"Added rain indicator: {output_col} (threshold={threshold:.2f} mm/h)")
    return result


def add_temp_category(
    df: pd.DataFrame,
    temp_col: str = _DEFAULT_TEMP_COL,
    output_col: str = "temp_category",
    bins: Optional[list[float]] = None,
    labels: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Add an ordered temperature category column.

    Default bins (°C):

    +-----------+--------------------+
    | Category  | Range              |
    +===========+====================+
    | cold      | < 5 °C             |
    +-----------+--------------------+
    | mild      | 5 °C – 15 °C       |
    +-----------+--------------------+
    | warm      | 15 °C – 25 °C      |
    +-----------+--------------------+
    | hot       | ≥ 25 °C            |
    +-----------+--------------------+

    Args:
        df: Input DataFrame. Not modified in place.
        temp_col: Column holding temperature in °C.
        output_col: Name of the output column.
        bins: Custom bin edges (overrides defaults).
        labels: Custom category labels matching ``len(bins) - 1`` entries.

    Returns:
        A copy of *df* with *output_col* appended as an ordered
        ``pd.Categorical``.

    Raises:
        ValueError: If *temp_col* is not present in *df*.
    """
    if temp_col not in df.columns:
        raise ValueError(f"Column '{temp_col}' not found in DataFrame.")

    _bins = bins if bins is not None else TEMP_BINS
    _labels = labels if labels is not None else TEMP_LABELS

    result = df.copy()
    result[output_col] = pd.cut(
        result[temp_col],
        bins=_bins,
        labels=_labels,
        ordered=True,
    )
    logger.debug(f"Added temperature category: {output_col}")
    return result


def add_comfort_index(
    df: pd.DataFrame,
    apparent_temp_col: str = _DEFAULT_APPARENT_TEMP_COL,
    temp_col: str = _DEFAULT_TEMP_COL,
    precipitation_col: str = _DEFAULT_PRECIP_COL,
    wind_speed_col: str = _DEFAULT_WIND_SPEED_COL,
    output_col: str = "comfort_index",
) -> pd.DataFrame:
    """Add a cycling comfort index.

    The index starts from the *apparent temperature* (which already incorporates
    humidity and wind-chill effects) and applies additional penalties for
    precipitation and strong wind::

        comfort_index = apparent_temp
                      - COMFORT_RAIN_COEFF  × precipitation          (mm/h)
                      - COMFORT_WIND_COEFF  × max(0, wind_speed - 10)  (km/h)

    *apparent_temperature* falls back to *temperature* when the column is
    absent or fully ``NaN``.  Precipitation and wind speed default to 0 when
    absent.

    Higher values mean more comfortable conditions for cycling.  Typical Paris
    range: roughly −20 (heavy rain, strong wind, cold) to +35 (hot summer day,
    no wind).

    Args:
        df: Input DataFrame. Not modified in place.
        apparent_temp_col: Column holding apparent (feels-like) temperature in °C.
        temp_col: Fallback temperature column (°C).
        precipitation_col: Column holding precipitation in mm/h.
        wind_speed_col: Column holding wind speed in km/h.
        output_col: Name of the output column.

    Returns:
        A copy of *df* with *output_col* appended.

    Raises:
        ValueError: If neither *apparent_temp_col* nor *temp_col* is present.
    """
    if apparent_temp_col not in df.columns and temp_col not in df.columns:
        raise ValueError(f"Neither '{apparent_temp_col}' nor '{temp_col}' found in DataFrame.")

    result = df.copy()

    # Base temperature: prefer apparent, fall back to dry-bulb
    if apparent_temp_col in result.columns:
        base_temp = result[apparent_temp_col].fillna(result.get(temp_col, 0.0))
    else:
        base_temp = result[temp_col].astype(float)

    # Precipitation penalty
    if precipitation_col in result.columns:
        precip = result[precipitation_col].fillna(0.0)
    else:
        precip = pd.Series(0.0, index=result.index)

    # Wind penalty above the free threshold
    if wind_speed_col in result.columns:
        wind = result[wind_speed_col].fillna(0.0)
    else:
        wind = pd.Series(0.0, index=result.index)
    wind_excess = (wind - COMFORT_WIND_FREE_KMH).clip(lower=0.0)

    result[output_col] = base_temp - COMFORT_RAIN_COEFF * precip - COMFORT_WIND_COEFF * wind_excess

    logger.debug(f"Added comfort index: {output_col}")
    return result


def add_all_weather_features(
    df: pd.DataFrame,
    lag_cols: Optional[list[str]] = None,
    lags_hours: tuple[int, ...] = (1, 2),
    timestamp_col: str = _DEFAULT_TIMESTAMP_COL,
    temp_col: str = _DEFAULT_TEMP_COL,
    apparent_temp_col: str = _DEFAULT_APPARENT_TEMP_COL,
    rain_col: str = _DEFAULT_RAIN_COL,
    precipitation_col: str = _DEFAULT_PRECIP_COL,
    wind_speed_col: str = _DEFAULT_WIND_SPEED_COL,
    rain_threshold: float = RAIN_THRESHOLD_MM,
    tolerance_minutes: int = 30,
) -> pd.DataFrame:
    """Apply all weather feature transformations in a single call.

    Convenience wrapper that sequentially calls:
        1. :func:`add_weather_lags` (when *lag_cols* is provided or inferred)
        2. :func:`add_is_raining`
        3. :func:`add_temp_category`
        4. :func:`add_comfort_index`

    When *lag_cols* is ``None``, lags are computed for all numeric weather
    columns present in *df* (excluding the timestamp).

    Args:
        df: Input weather DataFrame. Not modified in place.
        lag_cols: Columns to lag.  Auto-detected when ``None``.
        lags_hours: Hour offsets for lag features.
        timestamp_col: Timestamp column name.
        temp_col: Temperature column name.
        apparent_temp_col: Apparent temperature column name.
        rain_col: Rain column name.
        precipitation_col: Precipitation column name.
        wind_speed_col: Wind speed column name.
        rain_threshold: Threshold for :func:`add_is_raining`.
        tolerance_minutes: Tolerance for :func:`add_weather_lags`.

    Returns:
        A copy of *df* with all weather features appended.
    """
    # Infer lag columns: numeric columns except the timestamp
    if lag_cols is None:
        lag_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != timestamp_col]

    result = add_weather_lags(df, lag_cols, lags_hours, timestamp_col, tolerance_minutes)
    result = add_is_raining(result, rain_col, precipitation_col, threshold=rain_threshold)
    result = add_temp_category(result, temp_col)
    result = add_comfort_index(
        result, apparent_temp_col, temp_col, precipitation_col, wind_speed_col
    )
    logger.info("All weather features added successfully")
    return result
