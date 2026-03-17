"""Feature pipeline for Velib demand forecasting.

Orchestrates all feature modules (temporal, spatial, lagged, weather) into a
single ``fit / transform`` interface, with optional disk caching and a helper
for creating supervised-learning datasets.

Typical usage::

    from src.data.features.pipeline import FeaturePipeline, PipelineConfig

    config = PipelineConfig.from_yaml("config/features.yaml")
    pipeline = FeaturePipeline(config)

    # Fit on station metadata (computes & caches spatial features)
    pipeline.fit(station_df)

    # Enrich the time-series data
    features_df = pipeline.transform(status_df, weather_df)

    # Build a labeled training dataset
    X, y = pipeline.create_training_dataset(features_df, horizon_steps=1)
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
import yaml  # type: ignore[import-untyped]
from loguru import logger
from pydantic import BaseModel, Field

from src.data.features.lagged import add_all_lagged_features
from src.data.features.spatial import add_all_spatial_features
from src.data.features.temporal import add_all_temporal_features
from src.data.features.weather import add_all_weather_features

# ---------------------------------------------------------------------------
# Configuration models
# ---------------------------------------------------------------------------


class TemporalConfig(BaseModel):
    timestamp_col: str = "time"
    station_col: str = "station_id"
    country: str = "FR"


class SpatialConfig(BaseModel):
    lat_col: str = "lat"
    lon_col: str = "lon"
    id_col: str = "station_id"
    # Pydantic coerces YAML lists to tuples automatically
    ks: tuple[int, ...] = (5, 10)
    n_clusters: int = 8


class LaggedConfig(BaseModel):
    value_col: str = "num_bikes_available"
    lags: tuple[int, ...] = (1, 2, 3, 4)
    windows: tuple[str, ...] = ("1h", "2h", "6h", "24h")
    docks_col: str = "num_docks_available"


class WeatherConfig(BaseModel):
    timestamp_col: str = "time"
    lag_cols: list[str] = Field(
        default_factory=lambda: [
            "temperature",
            "apparent_temperature",
            "precipitation",
            "rain",
            "wind_speed",
        ]
    )
    lags_hours: tuple[int, ...] = (1, 2)
    rain_threshold: float = 0.1
    merge_tolerance_minutes: int = 30


class GeneralConfig(BaseModel):
    target_col: str = "num_bikes_available"
    missing_value_strategy: Literal["forward_fill", "drop", "none"] = "forward_fill"
    max_missing_fraction: float = Field(default=0.3, ge=0.0, le=1.0)
    drop_knn_id_cols: bool = True
    cache_enabled: bool = True
    cache_dir: str = ".feature_cache"


class PipelineConfig(BaseModel):
    """Full pipeline configuration, loadable from a YAML file.

    Each section maps to a dedicated sub-config model.  All fields have
    sensible defaults so ``PipelineConfig()`` works without any arguments.
    """

    temporal: TemporalConfig = Field(default_factory=TemporalConfig)
    spatial: SpatialConfig = Field(default_factory=SpatialConfig)
    lagged: LaggedConfig = Field(default_factory=LaggedConfig)
    weather: WeatherConfig = Field(default_factory=WeatherConfig)
    pipeline: GeneralConfig = Field(default_factory=GeneralConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        """Load and validate configuration from a YAML file.

        Pydantic validates all field types and coerces YAML lists to tuples
        where needed.  Unknown keys are silently ignored.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            A validated :class:`PipelineConfig` instance.
        """
        with open(path) as fh:
            raw = yaml.safe_load(fh) or {}
        return cls.model_validate(raw)

    @classmethod
    def default(cls) -> "PipelineConfig":
        """Return a :class:`PipelineConfig` populated with default values."""
        return cls()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _hash_df(df: pd.DataFrame) -> str:
    """Return a 12-character hex hash of a DataFrame's contents (not its index)."""
    hashed = pd.util.hash_pandas_object(df, index=False).values
    return hashlib.md5(hashed.tobytes()).hexdigest()[:12]  # type: ignore[union-attr]


def _is_list_column(series: pd.Series) -> bool:
    """Return True if *series* contains Python list values (not storable in Parquet)."""
    if series.dtype != object:
        return False
    sample = series.dropna()
    if sample.empty:
        return False
    return isinstance(sample.iloc[0], list)


def _merge_weather(
    status_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    timestamp_col: str,
    tolerance_minutes: int,
) -> pd.DataFrame:
    """Left-join weather features onto the status DataFrame by nearest timestamp.

    Weather has one row per timestamp; status has one row per (station, time).
    :func:`pandas.merge_asof` maps each status row to the nearest weather
    record within *tolerance_minutes*.

    Args:
        status_df: Status DataFrame (many rows per timestamp). Must have
            *timestamp_col* as a datetime column.
        weather_df: Enriched weather DataFrame (one row per timestamp).
        timestamp_col: Column name used as the merge key in both DataFrames.
        tolerance_minutes: Maximum allowed time gap for a valid match.

    Returns:
        Status DataFrame with weather feature columns appended.
    """
    # Only bring in columns that are not already in status_df
    new_weather_cols = [
        c for c in weather_df.columns if c not in status_df.columns or c == timestamp_col
    ]
    weather_slim = weather_df[new_weather_cols].copy()
    weather_slim[timestamp_col] = pd.to_datetime(weather_slim[timestamp_col])

    sorted_status = status_df.copy()
    sorted_status[timestamp_col] = pd.to_datetime(sorted_status[timestamp_col])
    sorted_status = sorted_status.sort_values(timestamp_col)

    weather_slim = weather_slim.sort_values(timestamp_col)

    merged = pd.merge_asof(
        sorted_status,
        weather_slim,
        on=timestamp_col,
        tolerance=pd.Timedelta(minutes=tolerance_minutes),
        direction="nearest",
    )
    return merged.sort_index()


def _handle_missing(
    df: pd.DataFrame,
    strategy: str,
    station_col: str,
    timestamp_col: str,
    max_missing_fraction: float,
) -> pd.DataFrame:
    """Apply missing-value strategy to the feature DataFrame.

    Args:
        df: Input DataFrame.
        strategy: One of ``"forward_fill"``, ``"drop"``, or ``"none"``.
        station_col: Station identifier column (used for per-station ffill).
        timestamp_col: Timestamp column (used for sorting before ffill).
        max_missing_fraction: Rows with a higher fraction of NaN numeric
            features are dropped (applied after the fill strategy).

    Returns:
        Cleaned DataFrame (may have fewer rows than *df*).
    """
    result = df.copy()

    if strategy == "forward_fill":
        numeric_cols: list[str] = result.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            result = result.sort_values([station_col, timestamp_col])
            result[numeric_cols] = result.groupby(station_col)[numeric_cols].ffill()
            result = result.sort_index()
        logger.debug("Applied forward-fill to %d numeric columns", len(numeric_cols))

    elif strategy == "drop":
        n_before = len(result)
        result = result.dropna()
        logger.debug("Dropped %d rows with any NaN value", n_before - len(result))

    # Row-level missing fraction filter
    if max_missing_fraction < 1.0:
        numeric_cols = result.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols):
            missing_frac = result[numeric_cols].isna().mean(axis=1)
            n_before = len(result)
            result = result[missing_frac <= max_missing_fraction]
            n_dropped = n_before - len(result)
            if n_dropped:
                logger.warning(
                    "Dropped %d rows with > %.0f%% missing features",
                    n_dropped,
                    max_missing_fraction * 100,
                )

    return result


# ---------------------------------------------------------------------------
# FeaturePipeline
# ---------------------------------------------------------------------------


class FeaturePipeline:
    """Orchestrates temporal, spatial, lagged, and weather feature engineering.

    The pipeline follows a ``fit / transform`` pattern:

    * :meth:`fit` computes **static** features (spatial) from the station
      metadata table and caches them to disk.
    * :meth:`transform` applies all feature transformations to a time-series
      status DataFrame and optionally merges weather features.
    * :meth:`create_training_dataset` converts the enriched DataFrame into
      ``(X, y)`` pairs suitable for supervised learning.

    Args:
        config: Pipeline configuration.  Defaults to :meth:`PipelineConfig.default`.
    """

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig.default()
        self._spatial_features: Optional[pd.DataFrame] = None
        self._cache_dir = Path(self.config.pipeline.cache_dir)

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_path(self, key: str) -> Path:
        return self._cache_dir / f"{key}.parquet"

    def _try_load_cache(self, key: str) -> Optional[pd.DataFrame]:
        path = self._cache_path(key)
        if path.exists():
            try:
                df = pd.read_parquet(path)
                logger.debug("Cache hit: %s", path.name)
                return df
            except Exception as exc:
                logger.warning("Cache read failed (%s): %s", path.name, exc)
        return None

    def _save_cache(self, df: pd.DataFrame, key: str) -> None:
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        path = self._cache_path(key)
        # List-valued columns cannot be stored in Parquet — drop them
        list_cols = [c for c in df.columns if _is_list_column(df[c])]
        cacheable = df.drop(columns=list_cols)
        try:
            cacheable.to_parquet(path, index=True)
            logger.debug("Saved cache: %s (%d cols)", path.name, len(cacheable.columns))
        except Exception as exc:
            logger.warning("Cache write failed (%s): %s", path.name, exc)

    def clear_cache(self) -> None:
        """Delete all cached Parquet files written by this pipeline."""
        if self._cache_dir.exists():
            shutil.rmtree(self._cache_dir)
            logger.info("Cleared feature cache: %s", self._cache_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, station_df: pd.DataFrame) -> "FeaturePipeline":
        """Compute static (spatial) features from station metadata.

        Spatial features depend only on station coordinates and are stable
        across time, so they are computed once and cached to disk.  Subsequent
        calls with the same station data are served from cache.

        Args:
            station_df: One row per station with at least the columns specified
                in :attr:`config.spatial` (``lat_col``, ``lon_col``, ``id_col``).

        Returns:
            *self*, for method chaining.
        """
        cfg = self.config.spatial
        logger.info("Fitting spatial features on %d stations", len(station_df))

        # Cache key: hash of the station coordinates used for spatial computation
        coord_cols = [cfg.lat_col, cfg.lon_col, cfg.id_col]
        cache_key = "spatial_" + _hash_df(station_df[coord_cols])

        if self.config.pipeline.cache_enabled:
            cached = self._try_load_cache(cache_key)
            if cached is not None:
                self._spatial_features = cached
                return self

        original_cols = set(station_df.columns)
        enriched = add_all_spatial_features(
            station_df,
            ks=cfg.ks,
            n_clusters=cfg.n_clusters,
            lat_col=cfg.lat_col,
            lon_col=cfg.lon_col,
            id_col=cfg.id_col,
        )

        # Keep only the station id + newly added feature columns
        new_cols = [c for c in enriched.columns if c not in original_cols]
        if self.config.pipeline.drop_knn_id_cols:
            new_cols = [c for c in new_cols if not c.endswith("_ids")]
        self._spatial_features = enriched[[cfg.id_col] + new_cols]

        if self.config.pipeline.cache_enabled:
            self._save_cache(self._spatial_features, cache_key)

        logger.info("Spatial features computed: %s", new_cols)
        return self

    def transform(
        self,
        status_df: pd.DataFrame,
        weather_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Apply all feature transformations to a station-status DataFrame.

        Steps applied in order:

        1. **Temporal** — hour sin/cos, day-of-week, rush-hour, time-since-last
        2. **Lagged** — lag features, rolling means, capacity gap, rate of change
        3. **Spatial** — merge pre-computed spatial features by ``station_id``
        4. **Weather** — enrich weather df then left-join onto status by time
        5. **Missing data** — forward-fill and/or row-level NaN filtering

        The result is optionally saved to a Parquet cache keyed by the hash of
        the input data, so repeated calls with identical inputs are instant.

        Args:
            status_df: One row per (station, timestamp).  Must have the columns
                referenced in the config (e.g. ``time``, ``station_id``,
                ``num_bikes_available``, ``num_docks_available``).
            weather_df: Optional weather time series (one row per timestamp).

        Returns:
            Enriched DataFrame with all feature columns appended.

        Raises:
            RuntimeError: If :meth:`fit` has not been called yet.
        """
        if self._spatial_features is None:
            raise RuntimeError("Call fit() before transform().")

        # Full-pipeline cache check
        if self.config.pipeline.cache_enabled:
            inputs_hash = _hash_df(status_df)
            if weather_df is not None:
                inputs_hash += _hash_df(weather_df)
            cache_key = "pipeline_" + hashlib.md5(inputs_hash.encode()).hexdigest()[:12]
            cached = self._try_load_cache(cache_key)
            if cached is not None:
                return cached

        cfg_t = self.config.temporal
        cfg_l = self.config.lagged

        result = status_df.copy()

        # 1. Temporal features
        result = add_all_temporal_features(
            result,
            timestamp_col=cfg_t.timestamp_col,
            country=cfg_t.country,
            group_col=cfg_t.station_col,
        )

        # 2. Lagged features
        result = add_all_lagged_features(
            result,
            value_col=cfg_l.value_col,
            lags=cfg_l.lags,
            windows=cfg_l.windows,
            station_col=cfg_t.station_col,
            timestamp_col=cfg_t.timestamp_col,
            docks_col=cfg_l.docks_col,
        )

        # 3. Spatial features — left join, skip columns already present
        spatial_cols_to_merge = [
            c
            for c in self._spatial_features.columns
            if c == self.config.spatial.id_col or c not in result.columns
        ]
        result = result.merge(
            self._spatial_features[spatial_cols_to_merge],
            on=self.config.spatial.id_col,
            how="left",
        )

        # 4. Weather features
        if weather_df is not None:
            cfg_w = self.config.weather
            weather_enriched = add_all_weather_features(
                weather_df,
                lag_cols=[c for c in cfg_w.lag_cols if c in weather_df.columns],
                lags_hours=cfg_w.lags_hours,
                timestamp_col=cfg_w.timestamp_col,
                rain_threshold=cfg_w.rain_threshold,
                tolerance_minutes=cfg_w.merge_tolerance_minutes,
            )
            result = _merge_weather(
                result,
                weather_enriched,
                timestamp_col=cfg_t.timestamp_col,
                tolerance_minutes=cfg_w.merge_tolerance_minutes,
            )

        # 5. Handle missing values
        result = _handle_missing(
            result,
            strategy=self.config.pipeline.missing_value_strategy,
            station_col=cfg_t.station_col,
            timestamp_col=cfg_t.timestamp_col,
            max_missing_fraction=self.config.pipeline.max_missing_fraction,
        )

        logger.info(
            "Pipeline transform complete: %d rows, %d columns", len(result), len(result.columns)
        )

        if self.config.pipeline.cache_enabled:
            self._save_cache(result, cache_key)

        return result

    def fit_transform(
        self,
        station_df: pd.DataFrame,
        status_df: pd.DataFrame,
        weather_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Fit on station metadata and transform the status data in one call.

        Equivalent to ``pipeline.fit(station_df).transform(status_df, weather_df)``.

        Args:
            station_df: Station metadata (one row per station).
            status_df: Station status time series.
            weather_df: Optional weather time series.

        Returns:
            Enriched feature DataFrame.
        """
        return self.fit(station_df).transform(status_df, weather_df)

    def create_training_dataset(
        self,
        df: pd.DataFrame,
        horizon_steps: int = 1,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Convert an enriched feature DataFrame into a supervised learning dataset.

        For each (station, timestamp) row the **target** is the value of
        ``target_col`` *horizon_steps* observations into the future (within the
        same station).  Rows at the end of each station's series (where the
        future value is unavailable) are dropped.

        List-valued columns (e.g. ``knn_5_ids``) and the raw target column are
        removed from ``X`` since they are not directly consumable by ML models.

        Args:
            df: Enriched DataFrame produced by :meth:`transform`.
            horizon_steps: Number of observation steps ahead to predict
                (default: 1, i.e. next observation).

        Returns:
            A ``(X, y)`` tuple where ``X`` is a DataFrame of features and ``y``
            is a Series of target values aligned by index.

        Raises:
            ValueError: If the target column is not present in *df*.
        """
        target_col = self.config.pipeline.target_col
        station_col = self.config.temporal.station_col
        timestamp_col = self.config.temporal.timestamp_col

        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

        sorted_df = df.sort_values([station_col, timestamp_col]).copy()

        # Shift target backward to get the future value
        y = sorted_df.groupby(station_col)[target_col].shift(-horizon_steps)

        # Drop rows where the future target is unavailable
        valid = ~y.isna()
        X = sorted_df[valid].copy()
        y = y[valid]

        # Remove columns not suitable as ML features
        drop_cols = [target_col]
        list_cols = [c for c in X.columns if _is_list_column(X[c])]
        drop_cols.extend(list_cols)
        X = X.drop(columns=[c for c in drop_cols if c in X.columns])

        logger.info(
            "Training dataset: %d samples, %d features, horizon=%d steps",
            len(X),
            len(X.columns),
            horizon_steps,
        )
        return X, y
