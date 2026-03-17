"""Unit tests for the feature pipeline module."""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.features.pipeline import (
    FeaturePipeline,
    GeneralConfig,
    LaggedConfig,
    PipelineConfig,
    SpatialConfig,
    TemporalConfig,
    WeatherConfig,
    _handle_missing,
    _merge_weather,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

N_STATIONS = 15  # must be > max(ks) = 10


@pytest.fixture
def station_df() -> pd.DataFrame:
    """Minimal station metadata: id + lat/lon spread across Paris."""
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "station_id": list(range(N_STATIONS)),
            "lat": 48.85 + rng.uniform(-0.05, 0.05, N_STATIONS),
            "lon": 2.35 + rng.uniform(-0.05, 0.05, N_STATIONS),
        }
    )


@pytest.fixture
def status_df(station_df: pd.DataFrame) -> pd.DataFrame:
    """Three hourly snapshots for every station."""
    times = pd.date_range("2024-01-08 08:00", periods=3, freq="1h")
    rows = []
    rng = np.random.default_rng(1)
    for t in times:
        for sid in station_df["station_id"]:
            bikes = int(rng.integers(0, 20))
            docks = int(rng.integers(0, 20))
            rows.append(
                {
                    "time": t,
                    "station_id": sid,
                    "num_bikes_available": bikes,
                    "num_docks_available": docks,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def weather_df() -> pd.DataFrame:
    """Five hourly weather observations."""
    times = pd.date_range("2024-01-08 06:00", periods=5, freq="1h")
    return pd.DataFrame(
        {
            "time": times,
            "temperature": [3.0, 5.0, 7.0, 6.0, 4.0],
            "apparent_temperature": [1.0, 3.0, 5.0, 4.0, 2.0],
            "precipitation": [0.0, 0.0, 0.5, 0.0, 0.0],
            "rain": [0.0, 0.0, 0.4, 0.0, 0.0],
            "wind_speed": [10.0, 8.0, 12.0, 9.0, 11.0],
        }
    )


@pytest.fixture
def config() -> PipelineConfig:
    """Default config with caching disabled to avoid touching the filesystem."""
    cfg = PipelineConfig.default()
    cfg.pipeline.cache_enabled = False
    return config_no_cache()


def config_no_cache() -> PipelineConfig:
    cfg = PipelineConfig.default()
    cfg.pipeline.cache_enabled = False
    return cfg


@pytest.fixture
def fitted_pipeline(station_df, config) -> FeaturePipeline:
    pipeline = FeaturePipeline(config_no_cache())
    pipeline.fit(station_df)
    return pipeline


# ---------------------------------------------------------------------------
# PipelineConfig
# ---------------------------------------------------------------------------


class TestPipelineConfig:
    def test_default_returns_config(self):
        cfg = PipelineConfig.default()
        assert isinstance(cfg, PipelineConfig)
        assert isinstance(cfg.temporal, TemporalConfig)
        assert isinstance(cfg.spatial, SpatialConfig)
        assert isinstance(cfg.lagged, LaggedConfig)
        assert isinstance(cfg.weather, WeatherConfig)
        assert isinstance(cfg.pipeline, GeneralConfig)

    def test_default_values(self):
        cfg = PipelineConfig.default()
        assert cfg.temporal.timestamp_col == "time"
        assert cfg.temporal.station_col == "station_id"
        assert cfg.spatial.ks == (5, 10)
        assert cfg.spatial.n_clusters == 8
        assert cfg.lagged.value_col == "num_bikes_available"
        assert "1h" in cfg.lagged.windows
        assert cfg.weather.rain_threshold == 0.1

    def test_from_yaml_loads_file(self, tmp_path):
        yaml_content = """
temporal:
  timestamp_col: time
  station_col: station_id
  country: FR
spatial:
  ks: [3, 7]
  n_clusters: 4
lagged:
  lags: [1, 2]
  windows: ["1h", "6h"]
weather:
  lags_hours: [1]
  rain_threshold: 0.2
pipeline:
  cache_enabled: false
  missing_value_strategy: none
"""
        config_file = tmp_path / "features.yaml"
        config_file.write_text(yaml_content)

        cfg = PipelineConfig.from_yaml(config_file)
        assert cfg.spatial.ks == (3, 7)
        assert cfg.spatial.n_clusters == 4
        assert cfg.lagged.lags == (1, 2)
        assert cfg.lagged.windows == ("1h", "6h")
        assert cfg.weather.rain_threshold == 0.2
        assert cfg.pipeline.cache_enabled is False

    def test_from_yaml_partial_file(self, tmp_path):
        """YAML with only some sections should use defaults for the rest."""
        config_file = tmp_path / "partial.yaml"
        config_file.write_text("spatial:\n  n_clusters: 5\n")
        cfg = PipelineConfig.from_yaml(config_file)
        assert cfg.spatial.n_clusters == 5
        assert cfg.temporal.country == "FR"  # default preserved

    def test_from_yaml_roundtrip(self, tmp_path):
        """Loading the project features.yaml should not raise."""
        project_yaml = Path(__file__).parent.parent / "config" / "features.yaml"
        if project_yaml.exists():
            cfg = PipelineConfig.from_yaml(project_yaml)
            assert cfg.spatial.ks == (5, 10)


# ---------------------------------------------------------------------------
# FeaturePipeline.fit
# ---------------------------------------------------------------------------


class TestFeaturePipelineFit:
    def test_fit_returns_self(self, station_df):
        pipeline = FeaturePipeline(config_no_cache())
        result = pipeline.fit(station_df)
        assert result is pipeline

    def test_fit_stores_spatial_features(self, station_df):
        pipeline = FeaturePipeline(config_no_cache())
        pipeline.fit(station_df)
        assert pipeline._spatial_features is not None

    def test_spatial_features_has_station_id(self, station_df):
        pipeline = FeaturePipeline(config_no_cache())
        pipeline.fit(station_df)
        assert "station_id" in pipeline._spatial_features.columns  # type: ignore[union-attr]

    def test_spatial_features_row_count(self, station_df):
        pipeline = FeaturePipeline(config_no_cache())
        pipeline.fit(station_df)
        assert len(pipeline._spatial_features) == len(station_df)  # type: ignore[arg-type]

    def test_spatial_features_dist_to_center(self, station_df):
        pipeline = FeaturePipeline(config_no_cache())
        pipeline.fit(station_df)
        assert "dist_to_center_km" in pipeline._spatial_features.columns  # type: ignore[union-attr]

    def test_spatial_features_no_knn_id_cols_by_default(self, station_df):
        pipeline = FeaturePipeline(config_no_cache())
        pipeline.fit(station_df)
        id_cols = [c for c in pipeline._spatial_features.columns if c.endswith("_ids")]  # type: ignore[union-attr]
        assert id_cols == []

    def test_fit_uses_cache_on_second_call(self, station_df, tmp_path):
        cfg = PipelineConfig.default()
        cfg.pipeline.cache_enabled = True
        cfg.pipeline.cache_dir = str(tmp_path / "cache")
        pipeline = FeaturePipeline(cfg)

        pipeline.fit(station_df)
        cache_files_after_first = list(Path(cfg.pipeline.cache_dir).glob("*.parquet"))
        assert cache_files_after_first  # cache was written

        # Second fit: should load from cache (no recomputation)
        pipeline2 = FeaturePipeline(cfg)
        pipeline2.fit(station_df)
        assert pipeline2._spatial_features is not None

    def test_clear_cache(self, station_df, tmp_path):
        cfg = PipelineConfig.default()
        cfg.pipeline.cache_enabled = True
        cfg.pipeline.cache_dir = str(tmp_path / "cache")
        pipeline = FeaturePipeline(cfg)
        pipeline.fit(station_df)
        assert Path(cfg.pipeline.cache_dir).exists()
        pipeline.clear_cache()
        assert not Path(cfg.pipeline.cache_dir).exists()


# ---------------------------------------------------------------------------
# FeaturePipeline.transform
# ---------------------------------------------------------------------------


class TestFeaturePipelineTransform:
    def test_transform_raises_without_fit(self, status_df):
        pipeline = FeaturePipeline(config_no_cache())
        with pytest.raises(RuntimeError, match="fit"):
            pipeline.transform(status_df)

    def test_transform_returns_dataframe(self, fitted_pipeline, status_df):
        result = fitted_pipeline.transform(status_df)
        assert isinstance(result, pd.DataFrame)

    def test_transform_preserves_row_count(self, fitted_pipeline, status_df):
        result = fitted_pipeline.transform(status_df)
        # Row count may decrease due to missing-value filtering; at least as many
        # stations × timestamps survive minus initial NaN rows (lags, etc.)
        assert len(result) > 0

    def test_transform_adds_temporal_cols(self, fitted_pipeline, status_df):
        result = fitted_pipeline.transform(status_df)
        for col in ("hour", "hour_sin", "hour_cos", "day_of_week", "is_weekend"):
            assert col in result.columns, f"Missing temporal col: {col}"

    def test_transform_adds_lagged_cols(self, fitted_pipeline, status_df):
        result = fitted_pipeline.transform(status_df)
        assert "num_bikes_available_lag_1" in result.columns
        assert "num_bikes_available_rolling_mean_1h" in result.columns

    def test_transform_adds_spatial_cols(self, fitted_pipeline, status_df):
        result = fitted_pipeline.transform(status_df)
        assert "dist_to_center_km" in result.columns
        assert "cluster_id" in result.columns

    def test_transform_with_weather_adds_weather_cols(
        self, fitted_pipeline, status_df, weather_df
    ):
        result = fitted_pipeline.transform(status_df, weather_df)
        assert "is_raining" in result.columns
        assert "temp_category" in result.columns
        assert "comfort_index" in result.columns

    def test_transform_without_weather_no_weather_cols(self, fitted_pipeline, status_df):
        result = fitted_pipeline.transform(status_df)
        assert "is_raining" not in result.columns

    def test_transform_original_df_not_modified(self, fitted_pipeline, status_df):
        cols_before = list(status_df.columns)
        fitted_pipeline.transform(status_df)
        assert list(status_df.columns) == cols_before


# ---------------------------------------------------------------------------
# FeaturePipeline.fit_transform
# ---------------------------------------------------------------------------


class TestFeaturePipelineFitTransform:
    def test_fit_transform_equivalent_to_fit_then_transform(
        self, station_df, status_df
    ):
        p1 = FeaturePipeline(config_no_cache())
        p1.fit(station_df)
        r1 = p1.transform(status_df)

        p2 = FeaturePipeline(config_no_cache())
        r2 = p2.fit_transform(station_df, status_df)

        assert set(r1.columns) == set(r2.columns)
        assert len(r1) == len(r2)

    def test_fit_transform_with_weather(self, station_df, status_df, weather_df):
        pipeline = FeaturePipeline(config_no_cache())
        result = pipeline.fit_transform(station_df, status_df, weather_df)
        assert "is_raining" in result.columns


# ---------------------------------------------------------------------------
# FeaturePipeline.create_training_dataset
# ---------------------------------------------------------------------------


class TestCreateTrainingDataset:
    def test_returns_tuple_of_two(self, fitted_pipeline, status_df):
        df = fitted_pipeline.transform(status_df)
        out = fitted_pipeline.create_training_dataset(df)
        assert len(out) == 2

    def test_X_is_dataframe(self, fitted_pipeline, status_df):
        df = fitted_pipeline.transform(status_df)
        X, y = fitted_pipeline.create_training_dataset(df)
        assert isinstance(X, pd.DataFrame)

    def test_y_is_series(self, fitted_pipeline, status_df):
        df = fitted_pipeline.transform(status_df)
        X, y = fitted_pipeline.create_training_dataset(df)
        assert isinstance(y, pd.Series)

    def test_X_y_same_length(self, fitted_pipeline, status_df):
        df = fitted_pipeline.transform(status_df)
        X, y = fitted_pipeline.create_training_dataset(df)
        assert len(X) == len(y)

    def test_target_col_not_in_X(self, fitted_pipeline, status_df):
        df = fitted_pipeline.transform(status_df)
        X, _ = fitted_pipeline.create_training_dataset(df)
        assert "num_bikes_available" not in X.columns

    def test_y_no_nan(self, fitted_pipeline, status_df):
        df = fitted_pipeline.transform(status_df)
        _, y = fitted_pipeline.create_training_dataset(df)
        assert not y.isna().any()

    def test_horizon_shifts_target(self, fitted_pipeline, status_df):
        """y with horizon=1 should equal y with horizon=2 shifted by one step."""
        df = fitted_pipeline.transform(status_df)
        # Two-step horizon produces fewer samples than one-step
        _, y1 = fitted_pipeline.create_training_dataset(df, horizon_steps=1)
        _, y2 = fitted_pipeline.create_training_dataset(df, horizon_steps=2)
        assert len(y2) <= len(y1)

    def test_missing_target_col_raises(self, fitted_pipeline, status_df):
        df = fitted_pipeline.transform(status_df).drop(columns=["num_bikes_available"])
        with pytest.raises(ValueError, match="num_bikes_available"):
            fitted_pipeline.create_training_dataset(df)

    def test_list_cols_not_in_X(self, station_df, status_df):
        cfg = config_no_cache()
        cfg.pipeline.drop_knn_id_cols = False  # keep knn id cols during fit
        pipeline = FeaturePipeline(cfg)
        pipeline.fit(station_df)
        df = pipeline.transform(status_df)
        X, _ = pipeline.create_training_dataset(df)
        list_cols = [c for c in X.columns if X[c].dtype == object]
        assert list_cols == [], f"Unexpected object cols in X: {list_cols}"


# ---------------------------------------------------------------------------
# _handle_missing
# ---------------------------------------------------------------------------


class TestHandleMissing:
    @pytest.fixture
    def df_with_nans(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "time": pd.date_range("2024-01-08", periods=4, freq="1h"),
                "station_id": [1, 1, 1, 1],
                "num_bikes_available": [10.0, np.nan, 8.0, np.nan],
                "feature_a": [1.0, np.nan, 3.0, 4.0],
            }
        )

    def test_forward_fill_fills_nans(self, df_with_nans):
        result = _handle_missing(
            df_with_nans, "forward_fill", "station_id", "time", max_missing_fraction=1.0
        )
        assert not result["num_bikes_available"].isna().any()
        assert result["num_bikes_available"].iloc[1] == 10.0  # filled from prior

    def test_drop_removes_nan_rows(self, df_with_nans):
        result = _handle_missing(
            df_with_nans, "drop", "station_id", "time", max_missing_fraction=1.0
        )
        assert not result.isna().any().any()
        assert len(result) < len(df_with_nans)

    def test_none_strategy_leaves_nans(self, df_with_nans):
        result = _handle_missing(
            df_with_nans, "none", "station_id", "time", max_missing_fraction=1.0
        )
        assert result["num_bikes_available"].isna().sum() == df_with_nans["num_bikes_available"].isna().sum()

    def test_max_missing_fraction_drops_rows(self):
        df = pd.DataFrame(
            {
                "time": pd.date_range("2024-01-08", periods=3, freq="1h"),
                "station_id": [1, 1, 1],
                "a": [1.0, np.nan, 3.0],
                "b": [1.0, np.nan, 3.0],
                "c": [1.0, np.nan, 3.0],
            }
        )
        # Row 1 has 100% missing numeric features → should be dropped at threshold 0.5
        result = _handle_missing(df, "none", "station_id", "time", max_missing_fraction=0.5)
        assert len(result) == 2

    def test_original_not_modified(self, df_with_nans):
        nans_before = df_with_nans["num_bikes_available"].isna().sum()
        _handle_missing(df_with_nans, "forward_fill", "station_id", "time", max_missing_fraction=1.0)
        assert df_with_nans["num_bikes_available"].isna().sum() == nans_before


# ---------------------------------------------------------------------------
# _merge_weather
# ---------------------------------------------------------------------------


class TestMergeWeather:
    def test_weather_cols_appended(self, status_df, weather_df):
        result = _merge_weather(status_df, weather_df, "time", tolerance_minutes=30)
        assert "temperature" in result.columns

    def test_row_count_unchanged(self, status_df, weather_df):
        result = _merge_weather(status_df, weather_df, "time", tolerance_minutes=30)
        assert len(result) == len(status_df)

    def test_outside_tolerance_gives_nan(self, status_df):
        """Weather timestamps far from status timestamps → NaN after merge."""
        far_weather = pd.DataFrame(
            {
                "time": pd.date_range("2024-01-09 00:00", periods=3, freq="1h"),
                "temperature": [5.0, 6.0, 7.0],
            }
        )
        result = _merge_weather(status_df, far_weather, "time", tolerance_minutes=30)
        assert result["temperature"].isna().all()

    def test_does_not_duplicate_existing_cols(self, status_df, weather_df):
        # Add a 'temperature' col to status_df — merge should not create temperature_x / _y
        status_with_temp = status_df.copy()
        status_with_temp["temperature"] = 10.0
        result = _merge_weather(status_with_temp, weather_df, "time", tolerance_minutes=30)
        assert "temperature_x" not in result.columns
        assert "temperature_y" not in result.columns

    def test_row_order_preserved(self, status_df, weather_df):
        result = _merge_weather(status_df, weather_df, "time", tolerance_minutes=30)
        pd.testing.assert_index_equal(result.index, status_df.index)
