"""Unit tests for lagged feature engineering module."""

import numpy as np
import pandas as pd
import pytest

from src.data.features.lagged import (
    add_all_lagged_features,
    add_capacity_gap,
    add_lag_features,
    add_rate_of_change,
    add_rolling_means,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def single_station_df() -> pd.DataFrame:
    """One station, 8 observations at 15-min cadence."""
    times = pd.date_range("2024-01-08 08:00", periods=8, freq="15min")
    return pd.DataFrame(
        {
            "time": times,
            "station_id": 1,
            "num_bikes_available": [10, 12, 8, 6, 9, 11, 7, 5],
            "num_docks_available": [10, 8, 12, 14, 11, 9, 13, 15],
        }
    )


@pytest.fixture
def two_station_df() -> pd.DataFrame:
    """Two stations, 6 observations each at 15-min cadence."""
    times = pd.date_range("2024-01-08 08:00", periods=6, freq="15min")
    return pd.DataFrame(
        {
            "time": list(times) * 2,
            "station_id": [1] * 6 + [2] * 6,
            "num_bikes_available": [10, 12, 8, 6, 9, 11, 20, 18, 15, 14, 16, 13],
            "num_docks_available": [10, 8, 12, 14, 11, 9, 5, 7, 10, 11, 9, 12],
        }
    )


@pytest.fixture
def capacity_df() -> pd.DataFrame:
    """DataFrame with explicit capacity column."""
    times = pd.date_range("2024-01-08 08:00", periods=4, freq="15min")
    return pd.DataFrame(
        {
            "time": times,
            "station_id": 1,
            "num_bikes_available": [10, 12, 8, 6],
            "num_docks_available": [10, 8, 12, 14],
            "capacity": [25, 25, 25, 25],
        }
    )


# ---------------------------------------------------------------------------
# add_lag_features
# ---------------------------------------------------------------------------


class TestAddLagFeatures:
    def test_columns_added(self, single_station_df):
        result = add_lag_features(single_station_df)
        for lag in (1, 2, 3, 4):
            assert f"num_bikes_available_lag_{lag}" in result.columns

    def test_lag1_values(self, single_station_df):
        result = add_lag_features(single_station_df, lags=(1,))
        values = result["num_bikes_available_lag_1"].tolist()
        expected = [np.nan, 10, 12, 8, 6, 9, 11, 7]
        for got, exp in zip(values, expected):
            if np.isnan(exp):
                assert pd.isna(got)
            else:
                assert got == exp

    def test_lag2_first_two_are_nan(self, single_station_df):
        result = add_lag_features(single_station_df, lags=(2,))
        assert pd.isna(result["num_bikes_available_lag_2"].iloc[0])
        assert pd.isna(result["num_bikes_available_lag_2"].iloc[1])
        assert not pd.isna(result["num_bikes_available_lag_2"].iloc[2])

    def test_lag_does_not_cross_station_boundary(self, two_station_df):
        result = add_lag_features(two_station_df, lags=(1,))
        # First observation of each station must be NaN
        for station in [1, 2]:
            first_idx = result[result["station_id"] == station].index[0]
            assert pd.isna(result.loc[first_idx, "num_bikes_available_lag_1"])

    def test_custom_lags(self, single_station_df):
        result = add_lag_features(single_station_df, lags=(1, 6))
        assert "num_bikes_available_lag_1" in result.columns
        assert "num_bikes_available_lag_6" in result.columns
        assert "num_bikes_available_lag_2" not in result.columns

    def test_row_count_unchanged(self, single_station_df):
        result = add_lag_features(single_station_df)
        assert len(result) == len(single_station_df)

    def test_original_df_not_modified(self, single_station_df):
        cols = list(single_station_df.columns)
        add_lag_features(single_station_df)
        assert list(single_station_df.columns) == cols

    def test_missing_column_raises(self, single_station_df):
        with pytest.raises(ValueError, match="missing_col"):
            add_lag_features(single_station_df, value_col="missing_col")

    def test_row_order_preserved(self, two_station_df):
        """Output index order must match input, not sorting order."""
        result = add_lag_features(two_station_df)
        pd.testing.assert_index_equal(result.index, two_station_df.index)


# ---------------------------------------------------------------------------
# add_rolling_means
# ---------------------------------------------------------------------------


class TestAddRollingMeans:
    def test_columns_added(self, single_station_df):
        result = add_rolling_means(single_station_df)
        for w in ("1h", "2h", "6h", "24h"):
            assert f"num_bikes_available_rolling_mean_{w}" in result.columns

    def test_custom_windows(self, single_station_df):
        result = add_rolling_means(single_station_df, windows=("30min",))
        assert "num_bikes_available_rolling_mean_30min" in result.columns

    def test_first_row_is_nan_with_closed_left(self, single_station_df):
        """With closed='left', no prior window → first row must be NaN."""
        result = add_rolling_means(single_station_df, windows=("1h",))
        assert pd.isna(result["num_bikes_available_rolling_mean_1h"].iloc[0])

    def test_rolling_mean_is_historical_only(self, single_station_df):
        """Mean at t should not include the value at t (closed='left')."""
        result = add_rolling_means(single_station_df, windows=("1h",))
        # Row 1 (t=08:15): only row 0 (t=08:00) is within the past 1 h
        assert result["num_bikes_available_rolling_mean_1h"].iloc[1] == pytest.approx(
            single_station_df["num_bikes_available"].iloc[0]
        )

    def test_does_not_cross_station_boundary(self, two_station_df):
        result = add_rolling_means(two_station_df, windows=("1h",))
        # First observation of each station must be NaN
        for station in [1, 2]:
            first_idx = result[result["station_id"] == station].index[0]
            assert pd.isna(result.loc[first_idx, "num_bikes_available_rolling_mean_1h"])

    def test_mean_within_value_range(self, single_station_df):
        result = add_rolling_means(single_station_df, windows=("1h",))
        col = "num_bikes_available_rolling_mean_1h"
        lo = single_station_df["num_bikes_available"].min()
        hi = single_station_df["num_bikes_available"].max()
        assert result[col].dropna().between(lo, hi).all()

    def test_row_count_unchanged(self, single_station_df):
        assert len(add_rolling_means(single_station_df)) == len(single_station_df)

    def test_original_df_not_modified(self, single_station_df):
        cols = list(single_station_df.columns)
        add_rolling_means(single_station_df)
        assert list(single_station_df.columns) == cols

    def test_row_order_preserved(self, two_station_df):
        result = add_rolling_means(two_station_df)
        pd.testing.assert_index_equal(result.index, two_station_df.index)


# ---------------------------------------------------------------------------
# add_capacity_gap
# ---------------------------------------------------------------------------


class TestAddCapacityGap:
    def test_columns_added(self, single_station_df):
        result = add_capacity_gap(single_station_df)
        assert "capacity_gap" in result.columns
        assert "fill_rate" in result.columns

    def test_gap_derived_from_docks(self, single_station_df):
        result = add_capacity_gap(single_station_df)
        # capacity = bikes + docks → gap = docks
        expected = single_station_df["num_docks_available"].astype(float)
        pd.testing.assert_series_equal(
            result["capacity_gap"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_fill_rate_derived_from_docks(self, single_station_df):
        result = add_capacity_gap(single_station_df)
        bikes = single_station_df["num_bikes_available"].astype(float)
        capacity = bikes + single_station_df["num_docks_available"].astype(float)
        expected = bikes / capacity
        pd.testing.assert_series_equal(
            result["fill_rate"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_explicit_capacity_col(self, capacity_df):
        result = add_capacity_gap(capacity_df, capacity_col="capacity")
        # gap = 25 - bikes
        expected = (25.0 - capacity_df["num_bikes_available"]).astype(float)
        pd.testing.assert_series_equal(
            result["capacity_gap"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_fill_rate_between_0_and_1(self, single_station_df):
        result = add_capacity_gap(single_station_df)
        assert result["fill_rate"].dropna().between(0, 1).all()

    def test_zero_capacity_gives_nan_fill_rate(self):
        df = pd.DataFrame(
            {
                "time": pd.to_datetime(["2024-01-08 08:00"]),
                "station_id": [1],
                "num_bikes_available": [0],
                "num_docks_available": [0],
            }
        )
        result = add_capacity_gap(df)
        assert pd.isna(result["fill_rate"].iloc[0])

    def test_custom_output_cols(self, single_station_df):
        result = add_capacity_gap(single_station_df, gap_col="gap", fill_rate_col="rate")
        assert "gap" in result.columns
        assert "rate" in result.columns

    def test_missing_docks_col_raises(self, single_station_df):
        with pytest.raises(ValueError, match="nonexistent"):
            add_capacity_gap(single_station_df, docks_col="nonexistent")

    def test_original_df_not_modified(self, single_station_df):
        cols = list(single_station_df.columns)
        add_capacity_gap(single_station_df)
        assert list(single_station_df.columns) == cols


# ---------------------------------------------------------------------------
# add_rate_of_change
# ---------------------------------------------------------------------------


class TestAddRateOfChange:
    def test_columns_added(self, single_station_df):
        result = add_rate_of_change(single_station_df)
        assert "num_bikes_available_diff" in result.columns
        assert "num_bikes_available_pct_change" in result.columns

    def test_first_row_is_nan(self, single_station_df):
        result = add_rate_of_change(single_station_df)
        assert pd.isna(result["num_bikes_available_diff"].iloc[0])
        assert pd.isna(result["num_bikes_available_pct_change"].iloc[0])

    def test_diff_values(self, single_station_df):
        result = add_rate_of_change(single_station_df)
        # Row 1: 12 - 10 = 2
        assert result["num_bikes_available_diff"].iloc[1] == pytest.approx(2.0)
        # Row 2: 8 - 12 = -4
        assert result["num_bikes_available_diff"].iloc[2] == pytest.approx(-4.0)

    def test_pct_change_values(self, single_station_df):
        result = add_rate_of_change(single_station_df)
        # Row 1: (12 - 10) / 10 = 0.2
        assert result["num_bikes_available_pct_change"].iloc[1] == pytest.approx(0.2)

    def test_does_not_cross_station_boundary(self, two_station_df):
        result = add_rate_of_change(two_station_df)
        for station in [1, 2]:
            first_idx = result[result["station_id"] == station].index[0]
            assert pd.isna(result.loc[first_idx, "num_bikes_available_diff"])

    def test_pct_change_nan_when_prev_zero(self):
        df = pd.DataFrame(
            {
                "time": pd.date_range("2024-01-08 08:00", periods=2, freq="15min"),
                "station_id": [1, 1],
                "num_bikes_available": [0, 5],
                "num_docks_available": [20, 15],
            }
        )
        result = add_rate_of_change(df)
        # prev=0 → pct_change should be NaN
        assert pd.isna(result["num_bikes_available_pct_change"].iloc[1])

    def test_custom_output_col_names(self, single_station_df):
        result = add_rate_of_change(single_station_df, diff_col="delta", pct_change_col="pct")
        assert "delta" in result.columns
        assert "pct" in result.columns

    def test_row_count_unchanged(self, single_station_df):
        assert len(add_rate_of_change(single_station_df)) == len(single_station_df)

    def test_original_df_not_modified(self, single_station_df):
        cols = list(single_station_df.columns)
        add_rate_of_change(single_station_df)
        assert list(single_station_df.columns) == cols

    def test_row_order_preserved(self, two_station_df):
        result = add_rate_of_change(two_station_df)
        pd.testing.assert_index_equal(result.index, two_station_df.index)


# ---------------------------------------------------------------------------
# add_all_lagged_features
# ---------------------------------------------------------------------------


class TestAddAllLaggedFeatures:
    EXPECTED_COLS = [
        "num_bikes_available_lag_1",
        "num_bikes_available_lag_2",
        "num_bikes_available_lag_3",
        "num_bikes_available_lag_4",
        "num_bikes_available_rolling_mean_1h",
        "num_bikes_available_rolling_mean_2h",
        "num_bikes_available_rolling_mean_6h",
        "num_bikes_available_rolling_mean_24h",
        "capacity_gap",
        "fill_rate",
        "num_bikes_available_diff",
        "num_bikes_available_pct_change",
    ]

    def test_all_columns_present(self, two_station_df):
        result = add_all_lagged_features(two_station_df)
        for col in self.EXPECTED_COLS:
            assert col in result.columns, f"Missing column: {col}"

    def test_original_columns_preserved(self, two_station_df):
        result = add_all_lagged_features(two_station_df)
        for col in two_station_df.columns:
            assert col in result.columns

    def test_row_count_unchanged(self, two_station_df):
        assert len(add_all_lagged_features(two_station_df)) == len(two_station_df)

    def test_original_df_not_modified(self, two_station_df):
        cols = list(two_station_df.columns)
        add_all_lagged_features(two_station_df)
        assert list(two_station_df.columns) == cols

    def test_with_explicit_capacity(self, capacity_df):
        result = add_all_lagged_features(capacity_df, capacity_col="capacity")
        assert "capacity_gap" in result.columns
        assert (result["capacity_gap"] == 25.0 - capacity_df["num_bikes_available"]).all()

    def test_row_order_preserved(self, two_station_df):
        result = add_all_lagged_features(two_station_df)
        pd.testing.assert_index_equal(result.index, two_station_df.index)
