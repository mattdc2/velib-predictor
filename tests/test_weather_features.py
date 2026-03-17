"""Unit tests for weather feature engineering module."""

import numpy as np
import pandas as pd
import pytest

from src.data.features.weather import (
    COMFORT_RAIN_COEFF,
    COMFORT_WIND_COEFF,
    COMFORT_WIND_FREE_KMH,
    TEMP_LABELS,
    add_all_weather_features,
    add_comfort_index,
    add_is_raining,
    add_temp_category,
    add_weather_lags,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def weather_df() -> pd.DataFrame:
    """Six hourly weather observations covering a range of conditions."""
    times = pd.date_range("2024-01-08 08:00", periods=6, freq="1h")
    return pd.DataFrame(
        {
            "time": times,
            "temperature": [-2.0, 3.0, 12.0, 20.0, 27.0, 32.0],
            "apparent_temperature": [-5.0, 1.0, 11.0, 19.0, 29.0, 34.0],
            "precipitation": [0.0, 0.5, 0.0, 2.0, 0.0, 0.0],
            "rain": [0.0, 0.4, 0.0, 1.8, 0.0, 0.0],
            "wind_speed": [5.0, 10.0, 15.0, 8.0, 20.0, 3.0],
            "wind_gusts": [8.0, 15.0, 22.0, 12.0, 30.0, 5.0],
        }
    )


@pytest.fixture
def minimal_df() -> pd.DataFrame:
    """Minimal DataFrame with only temperature and precipitation."""
    return pd.DataFrame(
        {
            "time": pd.date_range("2024-01-08 08:00", periods=3, freq="1h"),
            "temperature": [10.0, 15.0, 20.0],
            "precipitation": [0.0, 1.0, 0.0],
        }
    )


# ---------------------------------------------------------------------------
# add_weather_lags
# ---------------------------------------------------------------------------


class TestAddWeatherLags:
    def test_columns_added(self, weather_df):
        result = add_weather_lags(weather_df, lag_cols=["temperature"], lags_hours=(1, 2))
        assert "temperature_lag_1h" in result.columns
        assert "temperature_lag_2h" in result.columns

    def test_lag1h_value(self, weather_df):
        result = add_weather_lags(weather_df, lag_cols=["temperature"], lags_hours=(1,))
        # Row 1 (09:00) should see row 0 (08:00) temperature
        assert result["temperature_lag_1h"].iloc[1] == pytest.approx(
            weather_df["temperature"].iloc[0]
        )

    def test_lag2h_value(self, weather_df):
        result = add_weather_lags(weather_df, lag_cols=["temperature"], lags_hours=(2,))
        # Row 2 (10:00) should see row 0 (08:00) temperature
        assert result["temperature_lag_2h"].iloc[2] == pytest.approx(
            weather_df["temperature"].iloc[0]
        )

    def test_first_rows_are_nan(self, weather_df):
        result = add_weather_lags(weather_df, lag_cols=["temperature"], lags_hours=(1, 2))
        assert pd.isna(result["temperature_lag_1h"].iloc[0])
        assert pd.isna(result["temperature_lag_2h"].iloc[0])
        assert pd.isna(result["temperature_lag_2h"].iloc[1])

    def test_multiple_lag_cols(self, weather_df):
        result = add_weather_lags(
            weather_df, lag_cols=["temperature", "precipitation"], lags_hours=(1,)
        )
        assert "temperature_lag_1h" in result.columns
        assert "precipitation_lag_1h" in result.columns

    def test_outside_tolerance_gives_nan(self, weather_df):
        """Gap larger than tolerance should produce NaN for that lag."""
        # Remove row 2 to create a 2-hour gap between rows 1 and 3
        gapped = weather_df.drop(index=2).reset_index(drop=True)
        result = add_weather_lags(
            gapped, lag_cols=["temperature"], lags_hours=(1,), tolerance_minutes=30
        )
        # Row that was at 11:00 now has row at 09:00 as its 2nd prior row,
        # so 1h lag points to 10:00 which doesn't exist → NaN
        assert pd.isna(result["temperature_lag_1h"].iloc[2])

    def test_row_count_unchanged(self, weather_df):
        result = add_weather_lags(weather_df, lag_cols=["temperature"])
        assert len(result) == len(weather_df)

    def test_original_df_not_modified(self, weather_df):
        cols = list(weather_df.columns)
        add_weather_lags(weather_df, lag_cols=["temperature"])
        assert list(weather_df.columns) == cols

    def test_row_order_preserved(self, weather_df):
        result = add_weather_lags(weather_df, lag_cols=["temperature"])
        pd.testing.assert_index_equal(result.index, weather_df.index)

    def test_missing_timestamp_col_raises(self, weather_df):
        with pytest.raises(ValueError, match="bad_ts"):
            add_weather_lags(weather_df, lag_cols=["temperature"], timestamp_col="bad_ts")

    def test_missing_lag_col_raises(self, weather_df):
        with pytest.raises(ValueError, match="nonexistent"):
            add_weather_lags(weather_df, lag_cols=["nonexistent"])


# ---------------------------------------------------------------------------
# add_is_raining
# ---------------------------------------------------------------------------


class TestAddIsRaining:
    def test_column_added(self, weather_df):
        result = add_is_raining(weather_df)
        assert "is_raining" in result.columns

    def test_dry_rows_flagged_0(self, weather_df):
        result = add_is_raining(weather_df)
        assert result["is_raining"].iloc[0] == 0  # precipitation=0, rain=0

    def test_wet_rows_flagged_1(self, weather_df):
        result = add_is_raining(weather_df)
        assert result["is_raining"].iloc[1] == 1  # precipitation=0.5, rain=0.4
        assert result["is_raining"].iloc[3] == 1  # precipitation=2.0

    def test_threshold_respected(self, weather_df):
        # With threshold=3.0, row 3 (precip=2.0) should NOT be flagged
        result = add_is_raining(weather_df, threshold=3.0)
        assert result["is_raining"].iloc[3] == 0

    def test_only_rain_col_present(self, weather_df):
        df = weather_df.drop(columns=["precipitation"])
        result = add_is_raining(df)
        assert result["is_raining"].iloc[1] == 1

    def test_only_precip_col_present(self, weather_df):
        df = weather_df.drop(columns=["rain"])
        result = add_is_raining(df)
        assert result["is_raining"].iloc[1] == 1

    def test_neither_col_raises(self, weather_df):
        df = weather_df.drop(columns=["rain", "precipitation"])
        with pytest.raises(ValueError):
            add_is_raining(df)

    def test_custom_output_col(self, weather_df):
        result = add_is_raining(weather_df, output_col="raining")
        assert "raining" in result.columns

    def test_output_is_integer(self, weather_df):
        result = add_is_raining(weather_df)
        assert np.issubdtype(result["is_raining"].dtype, np.integer)

    def test_original_df_not_modified(self, weather_df):
        cols = list(weather_df.columns)
        add_is_raining(weather_df)
        assert list(weather_df.columns) == cols


# ---------------------------------------------------------------------------
# add_temp_category
# ---------------------------------------------------------------------------


class TestAddTempCategory:
    def test_column_added(self, weather_df):
        result = add_temp_category(weather_df)
        assert "temp_category" in result.columns

    def test_cold_below_5(self, weather_df):
        result = add_temp_category(weather_df)
        # -2°C and 3°C → cold
        assert result["temp_category"].iloc[0] == "cold"
        assert result["temp_category"].iloc[1] == "cold"

    def test_mild_5_to_15(self, weather_df):
        result = add_temp_category(weather_df)
        # 12°C → mild
        assert result["temp_category"].iloc[2] == "mild"

    def test_warm_15_to_25(self, weather_df):
        result = add_temp_category(weather_df)
        # 20°C → warm
        assert result["temp_category"].iloc[3] == "warm"

    def test_hot_above_25(self, weather_df):
        result = add_temp_category(weather_df)
        # 27°C and 32°C → hot
        assert result["temp_category"].iloc[4] == "hot"
        assert result["temp_category"].iloc[5] == "hot"

    def test_all_four_categories_present(self, weather_df):
        result = add_temp_category(weather_df)
        assert set(result["temp_category"].dropna().unique()) == set(TEMP_LABELS)

    def test_ordered_categorical(self, weather_df):
        result = add_temp_category(weather_df)
        assert hasattr(result["temp_category"], "cat")
        assert result["temp_category"].cat.ordered

    def test_cold_less_than_warm(self, weather_df):
        result = add_temp_category(weather_df)
        assert result["temp_category"].iloc[0] < result["temp_category"].iloc[3]

    def test_custom_bins_and_labels(self, weather_df):
        result = add_temp_category(
            weather_df,
            bins=[-np.inf, 10.0, np.inf],
            labels=["below10", "above10"],
        )
        assert result["temp_category"].iloc[0] == "below10"
        assert result["temp_category"].iloc[3] == "above10"

    def test_missing_temp_col_raises(self, weather_df):
        with pytest.raises(ValueError, match="no_temp"):
            add_temp_category(weather_df, temp_col="no_temp")

    def test_original_df_not_modified(self, weather_df):
        cols = list(weather_df.columns)
        add_temp_category(weather_df)
        assert list(weather_df.columns) == cols


# ---------------------------------------------------------------------------
# add_comfort_index
# ---------------------------------------------------------------------------


class TestAddComfortIndex:
    def test_column_added(self, weather_df):
        result = add_comfort_index(weather_df)
        assert "comfort_index" in result.columns

    def test_no_wind_no_rain_equals_apparent_temp(self, weather_df):
        # Row 0: wind_speed=5 (<10 free threshold), precipitation=0
        result = add_comfort_index(weather_df)
        expected = weather_df["apparent_temperature"].iloc[0]
        assert result["comfort_index"].iloc[0] == pytest.approx(expected)

    def test_rain_reduces_comfort(self, weather_df):
        result = add_comfort_index(weather_df)
        # Row 3: precip=2.0, penalty = 2.0 * 2.0 = 4.0
        base = weather_df["apparent_temperature"].iloc[3]
        assert result["comfort_index"].iloc[3] < base

    def test_rain_penalty_magnitude(self, weather_df):
        result = add_comfort_index(weather_df)
        # Row 3: wind_speed=8 (<10, no wind penalty), precip=2.0
        base = weather_df["apparent_temperature"].iloc[3]
        expected = base - COMFORT_RAIN_COEFF * 2.0
        assert result["comfort_index"].iloc[3] == pytest.approx(expected)

    def test_wind_penalty_above_threshold(self, weather_df):
        result = add_comfort_index(weather_df)
        # Row 4: wind_speed=20, excess=10, penalty=0.3*10=3; precip=0
        base = weather_df["apparent_temperature"].iloc[4]
        expected = base - COMFORT_WIND_COEFF * (20.0 - COMFORT_WIND_FREE_KMH)
        assert result["comfort_index"].iloc[4] == pytest.approx(expected)

    def test_apparent_temp_fallback_to_temperature(self, minimal_df):
        result = add_comfort_index(minimal_df)
        # apparent_temperature absent → falls back to temperature; no wind col
        assert result["comfort_index"].iloc[0] == pytest.approx(minimal_df["temperature"].iloc[0])

    def test_neither_temp_col_raises(self, weather_df):
        df = weather_df.drop(columns=["temperature", "apparent_temperature"])
        with pytest.raises(ValueError):
            add_comfort_index(df)

    def test_custom_output_col(self, weather_df):
        result = add_comfort_index(weather_df, output_col="cycling_comfort")
        assert "cycling_comfort" in result.columns

    def test_original_df_not_modified(self, weather_df):
        cols = list(weather_df.columns)
        add_comfort_index(weather_df)
        assert list(weather_df.columns) == cols


# ---------------------------------------------------------------------------
# add_all_weather_features
# ---------------------------------------------------------------------------


class TestAddAllWeatherFeatures:
    DERIVED_COLS = ["is_raining", "temp_category", "comfort_index"]

    def test_derived_cols_present(self, weather_df):
        result = add_all_weather_features(weather_df)
        for col in self.DERIVED_COLS:
            assert col in result.columns, f"Missing: {col}"

    def test_lag_cols_present(self, weather_df):
        result = add_all_weather_features(weather_df, lag_cols=["temperature"], lags_hours=(1,))
        assert "temperature_lag_1h" in result.columns

    def test_auto_lag_cols(self, weather_df):
        result = add_all_weather_features(weather_df, lags_hours=(1,))
        # All numeric columns should have lag variants
        numeric_cols = weather_df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            assert f"{col}_lag_1h" in result.columns

    def test_original_columns_preserved(self, weather_df):
        result = add_all_weather_features(weather_df)
        for col in weather_df.columns:
            assert col in result.columns

    def test_row_count_unchanged(self, weather_df):
        assert len(add_all_weather_features(weather_df)) == len(weather_df)

    def test_original_df_not_modified(self, weather_df):
        cols = list(weather_df.columns)
        add_all_weather_features(weather_df)
        assert list(weather_df.columns) == cols

    def test_row_order_preserved(self, weather_df):
        result = add_all_weather_features(weather_df)
        pd.testing.assert_index_equal(result.index, weather_df.index)
