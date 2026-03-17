"""Unit tests for temporal feature engineering module."""

import numpy as np
import pandas as pd
import pytest

from src.data.features.temporal import (
    EVENING_RUSH_END,
    EVENING_RUSH_START,
    MORNING_RUSH_END,
    MORNING_RUSH_START,
    add_all_temporal_features,
    add_day_features,
    add_hour_features,
    add_rush_hour_features,
    add_time_since_last_observation,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """DataFrame covering multiple hours, days, and a French public holiday."""
    timestamps = [
        "2024-01-01 00:00:00",  # New Year's Day (FR holiday), Monday, midnight
        "2024-01-01 07:30:00",  # New Year's Day, morning rush
        "2024-01-06 08:00:00",  # Saturday (weekend), morning rush
        "2024-01-06 17:30:00",  # Saturday (weekend), evening rush
        "2024-01-08 12:00:00",  # Monday, midday (not rush)
        "2024-01-08 18:00:00",  # Monday, evening rush
        "2024-01-09 22:00:00",  # Tuesday, night (not rush)
    ]
    return pd.DataFrame({"timestamp": pd.to_datetime(timestamps), "value": range(7)})


@pytest.fixture
def multi_station_df() -> pd.DataFrame:
    """DataFrame with two stations for grouped time-delta tests."""
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2024-01-08 10:00:00",
                    "2024-01-08 10:05:00",
                    "2024-01-08 10:10:00",
                    "2024-01-08 10:00:00",
                    "2024-01-08 10:15:00",
                ]
            ),
            "station_id": ["A", "A", "A", "B", "B"],
            "value": range(5),
        }
    )


# ---------------------------------------------------------------------------
# add_hour_features
# ---------------------------------------------------------------------------


class TestAddHourFeatures:
    def test_columns_added(self, sample_df):
        result = add_hour_features(sample_df)
        assert {"hour", "hour_sin", "hour_cos"}.issubset(result.columns)

    def test_hour_values(self, sample_df):
        result = add_hour_features(sample_df)
        assert result["hour"].iloc[0] == 0  # midnight
        assert result["hour"].iloc[1] == 7  # 07:30

    def test_cyclical_at_midnight(self, sample_df):
        result = add_hour_features(sample_df)
        # hour 0 → sin=0, cos=1
        assert result["hour_sin"].iloc[0] == pytest.approx(0.0, abs=1e-9)
        assert result["hour_cos"].iloc[0] == pytest.approx(1.0, abs=1e-9)

    def test_cyclical_at_noon(self):
        df = pd.DataFrame({"timestamp": pd.to_datetime(["2024-01-08 12:00:00"])})
        result = add_hour_features(df)
        # hour 12 → sin(π)≈0, cos(π)=-1
        assert result["hour_sin"].iloc[0] == pytest.approx(0.0, abs=1e-9)
        assert result["hour_cos"].iloc[0] == pytest.approx(-1.0, abs=1e-9)

    def test_cyclical_at_hour6(self):
        df = pd.DataFrame({"timestamp": pd.to_datetime(["2024-01-08 06:00:00"])})
        result = add_hour_features(df)
        # hour 6 → sin(π/2)=1, cos(π/2)≈0
        assert result["hour_sin"].iloc[0] == pytest.approx(1.0, abs=1e-9)
        assert result["hour_cos"].iloc[0] == pytest.approx(0.0, abs=1e-9)

    def test_sin_cos_in_range(self, sample_df):
        result = add_hour_features(sample_df)
        assert result["hour_sin"].between(-1, 1).all()
        assert result["hour_cos"].between(-1, 1).all()

    def test_original_df_not_modified(self, sample_df):
        original_cols = list(sample_df.columns)
        add_hour_features(sample_df)
        assert list(sample_df.columns) == original_cols

    def test_datetime_index_supported(self, sample_df):
        df_indexed = sample_df.set_index("timestamp")
        result = add_hour_features(df_indexed, timestamp_col="timestamp")
        assert "hour" in result.columns

    def test_missing_column_raises(self, sample_df):
        with pytest.raises(ValueError, match="nonexistent"):
            add_hour_features(sample_df, timestamp_col="nonexistent")


# ---------------------------------------------------------------------------
# add_day_features
# ---------------------------------------------------------------------------


class TestAddDayFeatures:
    def test_columns_added(self, sample_df):
        result = add_day_features(sample_df)
        assert {"day_of_week", "is_weekend", "is_holiday"}.issubset(result.columns)

    def test_monday_is_day_0(self, sample_df):
        # 2024-01-01 is a Monday
        result = add_day_features(sample_df)
        assert result["day_of_week"].iloc[0] == 0

    def test_saturday_is_weekend(self, sample_df):
        # 2024-01-06 is a Saturday
        result = add_day_features(sample_df)
        assert result["is_weekend"].iloc[2] == 1

    def test_monday_not_weekend(self, sample_df):
        result = add_day_features(sample_df)
        assert result["is_weekend"].iloc[0] == 0

    def test_new_years_day_is_french_holiday(self, sample_df):
        result = add_day_features(sample_df, country="FR")
        # rows 0 and 1 are both on 2024-01-01
        assert result["is_holiday"].iloc[0] == 1
        assert result["is_holiday"].iloc[1] == 1

    def test_regular_weekday_not_holiday(self, sample_df):
        result = add_day_features(sample_df, country="FR")
        # 2024-01-08 and 2024-01-09 are ordinary days
        assert result["is_holiday"].iloc[4] == 0
        assert result["is_holiday"].iloc[6] == 0

    def test_is_weekend_is_integer(self, sample_df):
        result = add_day_features(sample_df)
        assert np.issubdtype(result["is_weekend"].dtype, np.integer)

    def test_original_df_not_modified(self, sample_df):
        original_cols = list(sample_df.columns)
        add_day_features(sample_df)
        assert list(sample_df.columns) == original_cols


# ---------------------------------------------------------------------------
# add_rush_hour_features
# ---------------------------------------------------------------------------


class TestAddRushHourFeatures:
    def test_columns_added(self, sample_df):
        result = add_rush_hour_features(sample_df)
        assert {"is_morning_rush", "is_evening_rush", "is_rush_hour"}.issubset(result.columns)

    def test_morning_rush(self, sample_df):
        # 07:30 → morning rush
        result = add_rush_hour_features(sample_df)
        assert result["is_morning_rush"].iloc[1] == 1
        assert result["is_evening_rush"].iloc[1] == 0
        assert result["is_rush_hour"].iloc[1] == 1

    def test_evening_rush(self, sample_df):
        # 17:30 → evening rush
        result = add_rush_hour_features(sample_df)
        assert result["is_evening_rush"].iloc[3] == 1
        assert result["is_morning_rush"].iloc[3] == 0
        assert result["is_rush_hour"].iloc[3] == 1

    def test_18h_is_evening_rush(self, sample_df):
        # 18:00 is within the evening window
        result = add_rush_hour_features(sample_df)
        idx = sample_df[sample_df["timestamp"].dt.hour == 18].index[0]
        assert result.loc[idx, "is_evening_rush"] == 1

    def test_midnight_not_rush(self, sample_df):
        result = add_rush_hour_features(sample_df)
        assert result["is_morning_rush"].iloc[0] == 0
        assert result["is_evening_rush"].iloc[0] == 0
        assert result["is_rush_hour"].iloc[0] == 0

    def test_noon_not_rush(self, sample_df):
        result = add_rush_hour_features(sample_df)
        idx = sample_df[sample_df["timestamp"].dt.hour == 12].index[0]
        assert result.loc[idx, "is_rush_hour"] == 0

    def test_rush_hour_boundaries(self):
        """Boundary hours must be included; adjacent hours must not."""
        timestamps = [f"2024-01-08 {h:02d}:00:00" for h in range(24)]
        df = pd.DataFrame({"timestamp": pd.to_datetime(timestamps)})
        result = add_rush_hour_features(df)

        for h in range(MORNING_RUSH_START, MORNING_RUSH_END + 1):
            assert result.loc[h, "is_morning_rush"] == 1, f"hour {h} should be morning rush"
        assert result.loc[MORNING_RUSH_START - 1, "is_morning_rush"] == 0
        assert result.loc[MORNING_RUSH_END + 1, "is_morning_rush"] == 0

        for h in range(EVENING_RUSH_START, EVENING_RUSH_END + 1):
            assert result.loc[h, "is_evening_rush"] == 1, f"hour {h} should be evening rush"
        assert result.loc[EVENING_RUSH_START - 1, "is_evening_rush"] == 0
        assert result.loc[EVENING_RUSH_END + 1, "is_evening_rush"] == 0

    def test_original_df_not_modified(self, sample_df):
        original_cols = list(sample_df.columns)
        add_rush_hour_features(sample_df)
        assert list(sample_df.columns) == original_cols


# ---------------------------------------------------------------------------
# add_time_since_last_observation
# ---------------------------------------------------------------------------


class TestAddTimeSinceLastObservation:
    def test_column_added(self, sample_df):
        result = add_time_since_last_observation(sample_df)
        assert "seconds_since_last_obs" in result.columns

    def test_first_row_is_nan(self, sample_df):
        result = add_time_since_last_observation(sample_df)
        assert pd.isna(result["seconds_since_last_obs"].iloc[0])

    def test_delta_values(self):
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    [
                        "2024-01-08 10:00:00",
                        "2024-01-08 10:05:00",
                        "2024-01-08 10:15:00",
                    ]
                )
            }
        )
        result = add_time_since_last_observation(df)
        assert result["seconds_since_last_obs"].iloc[1] == pytest.approx(300.0)
        assert result["seconds_since_last_obs"].iloc[2] == pytest.approx(600.0)

    def test_grouped_first_rows_are_nan(self, multi_station_df):
        df_sorted = multi_station_df.sort_values(["station_id", "timestamp"]).reset_index(drop=True)
        result = add_time_since_last_observation(df_sorted, group_col="station_id")
        for station in ["A", "B"]:
            first_idx = result[result["station_id"] == station].index[0]
            assert pd.isna(result.loc[first_idx, "seconds_since_last_obs"])

    def test_grouped_delta_values(self, multi_station_df):
        df_sorted = multi_station_df.sort_values(["station_id", "timestamp"]).reset_index(drop=True)
        result = add_time_since_last_observation(df_sorted, group_col="station_id")
        station_a = result[result["station_id"] == "A"].reset_index(drop=True)
        assert station_a["seconds_since_last_obs"].iloc[1] == pytest.approx(300.0)  # 5 min
        assert station_a["seconds_since_last_obs"].iloc[2] == pytest.approx(300.0)  # 5 min

    def test_custom_output_column_name(self, sample_df):
        result = add_time_since_last_observation(sample_df, output_col="time_delta_s")
        assert "time_delta_s" in result.columns

    def test_original_df_not_modified(self, sample_df):
        original_cols = list(sample_df.columns)
        add_time_since_last_observation(sample_df)
        assert list(sample_df.columns) == original_cols

    def test_missing_column_raises(self, sample_df):
        with pytest.raises(ValueError):
            add_time_since_last_observation(sample_df, timestamp_col="nonexistent")


# ---------------------------------------------------------------------------
# add_all_temporal_features
# ---------------------------------------------------------------------------


class TestAddAllTemporalFeatures:
    EXPECTED_COLS = [
        "hour",
        "hour_sin",
        "hour_cos",
        "day_of_week",
        "is_weekend",
        "is_holiday",
        "is_morning_rush",
        "is_evening_rush",
        "is_rush_hour",
        "seconds_since_last_obs",
    ]

    def test_all_columns_present(self, sample_df):
        result = add_all_temporal_features(sample_df)
        for col in self.EXPECTED_COLS:
            assert col in result.columns, f"Missing column: {col}"

    def test_original_columns_preserved(self, sample_df):
        result = add_all_temporal_features(sample_df)
        assert "timestamp" in result.columns
        assert "value" in result.columns

    def test_row_count_unchanged(self, sample_df):
        result = add_all_temporal_features(sample_df)
        assert len(result) == len(sample_df)

    def test_with_group_col(self, multi_station_df):
        df_sorted = multi_station_df.sort_values(["station_id", "timestamp"])
        result = add_all_temporal_features(df_sorted, group_col="station_id")
        for col in self.EXPECTED_COLS:
            assert col in result.columns, f"Missing column: {col}"

    def test_original_df_not_modified(self, sample_df):
        original_cols = list(sample_df.columns)
        add_all_temporal_features(sample_df)
        assert list(sample_df.columns) == original_cols
