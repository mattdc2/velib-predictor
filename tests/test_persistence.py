"""Unit tests for persistence and historical-average baseline models."""

import math

import numpy as np
import pandas as pd
import pytest

from src.models.baseline.persistence import (
    HistoricalAverageModel,
    PersistenceModel,
    compute_metrics,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def two_station_df() -> pd.DataFrame:
    """Two stations, each with 6 hourly observations."""
    times = pd.date_range("2024-01-08 08:00", periods=6, freq="1h")
    rows = []
    for sid, base in [(1, 10), (2, 20)]:
        for i, t in enumerate(times):
            rows.append(
                {
                    "time": t,
                    "station_id": sid,
                    "num_bikes_available": base + i,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def single_station_df() -> pd.DataFrame:
    """One station, 4 hourly observations."""
    return pd.DataFrame(
        {
            "time": pd.date_range("2024-01-08 10:00", periods=4, freq="1h"),
            "station_id": [42] * 4,
            "num_bikes_available": [5, 8, 6, 9],
        }
    )


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------


class TestComputeMetrics:
    def test_perfect_forecast_gives_zero_errors(self):
        y = pd.Series([10.0, 20.0, 30.0])
        m = compute_metrics(y, y)
        assert m["mae"] == pytest.approx(0.0)
        assert m["rmse"] == pytest.approx(0.0)
        assert m["mape"] == pytest.approx(0.0)

    def test_mae_value(self):
        y_true = pd.Series([10.0, 20.0, 30.0])
        y_pred = pd.Series([12.0, 18.0, 33.0])
        m = compute_metrics(y_true, y_pred)
        assert m["mae"] == pytest.approx((2 + 2 + 3) / 3)

    def test_rmse_value(self):
        y_true = pd.Series([0.0, 0.0])
        y_pred = pd.Series([3.0, 4.0])
        m = compute_metrics(y_true, y_pred)
        # errors: 3, 4 → rmse = sqrt((9+16)/2) = sqrt(12.5)
        assert m["rmse"] == pytest.approx(math.sqrt(12.5))

    def test_mape_avoids_division_by_zero(self):
        # y_true contains zero — should not raise
        y_true = pd.Series([0.0, 10.0])
        y_pred = pd.Series([2.0, 10.0])
        m = compute_metrics(y_true, y_pred)
        assert not math.isnan(m["mape"])

    def test_mape_percentage_scale(self):
        y_true = pd.Series([10.0])
        y_pred = pd.Series([11.0])
        m = compute_metrics(y_true, y_pred)
        # |1| / max(10, 1) * 100 = 10%
        assert m["mape"] == pytest.approx(10.0)

    def test_nan_rows_are_ignored(self):
        y_true = pd.Series([np.nan, 10.0, 20.0])
        y_pred = pd.Series([5.0, 10.0, 20.0])
        m = compute_metrics(y_true, y_pred)
        assert m["mae"] == pytest.approx(0.0)

    def test_all_nan_returns_nan_metrics(self):
        y = pd.Series([np.nan, np.nan])
        m = compute_metrics(y, y)
        assert math.isnan(m["mae"])
        assert math.isnan(m["rmse"])
        assert math.isnan(m["mape"])

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="same length"):
            compute_metrics(pd.Series([1, 2]), pd.Series([1]))

    def test_returns_all_three_keys(self):
        m = compute_metrics(pd.Series([1.0]), pd.Series([1.0]))
        assert set(m.keys()) == {"mae", "rmse", "mape"}


# ---------------------------------------------------------------------------
# PersistenceModel
# ---------------------------------------------------------------------------


class TestPersistenceModel:
    def test_fit_returns_self(self, two_station_df):
        model = PersistenceModel()
        assert model.fit(two_station_df) is model

    def test_fit_stores_last_known(self, two_station_df):
        model = PersistenceModel()
        model.fit(two_station_df)
        # Station 1: base=10, last index=5 → 10+5=15
        assert model._last_known[1] == pytest.approx(15.0)
        assert model._last_known[2] == pytest.approx(25.0)

    def test_predict_uses_lag_col_when_present(self, two_station_df):
        model = PersistenceModel()
        model.fit(two_station_df)
        df = two_station_df.copy()
        df["num_bikes_available_lag_1"] = 99.0
        preds = model.predict(df)
        assert (preds == 99.0).all()

    def test_predict_derives_lag_in_place(self, two_station_df):
        model = PersistenceModel()
        model.fit(two_station_df)
        preds = model.predict(two_station_df)
        # For station 1, second row prediction should equal first row value (10)
        s1 = two_station_df[two_station_df["station_id"] == 1].sort_values("time")
        assert preds.loc[s1.index[1]] == pytest.approx(10.0)

    def test_predict_first_row_filled_from_fit(self, single_station_df):
        train = single_station_df.iloc[:2]
        model = PersistenceModel()
        model.fit(train)
        # Inject a fresh df where station 42's first row has no prior
        single_row = pd.DataFrame(
            {
                "time": [pd.Timestamp("2024-01-08 20:00")],
                "station_id": [42],
                "num_bikes_available": [7],
            }
        )
        preds = model.predict(single_row)
        # last known for station 42 after training on first 2 rows = 8
        assert preds.iloc[0] == pytest.approx(8.0)

    def test_predict_series_named_prediction(self, two_station_df):
        model = PersistenceModel().fit(two_station_df)
        assert model.predict(two_station_df).name == "prediction"

    def test_predict_aligned_with_input_index(self, two_station_df):
        model = PersistenceModel().fit(two_station_df)
        preds = model.predict(two_station_df)
        pd.testing.assert_index_equal(preds.index, two_station_df.index)

    def test_predict_length_matches_input(self, two_station_df):
        model = PersistenceModel().fit(two_station_df)
        assert len(model.predict(two_station_df)) == len(two_station_df)

    def test_evaluate_returns_metric_dict(self, two_station_df):
        model = PersistenceModel().fit(two_station_df)
        metrics = model.evaluate(two_station_df)
        assert set(metrics.keys()) == {"mae", "rmse", "mape"}

    def test_evaluate_mae_is_nonnegative(self, two_station_df):
        model = PersistenceModel().fit(two_station_df)
        assert model.evaluate(two_station_df)["mae"] >= 0


# ---------------------------------------------------------------------------
# HistoricalAverageModel
# ---------------------------------------------------------------------------


class TestHistoricalAverageModel:
    def test_fit_returns_self(self, two_station_df):
        model = HistoricalAverageModel()
        assert model.fit(two_station_df) is model

    def test_fit_populates_lookup(self, two_station_df):
        model = HistoricalAverageModel().fit(two_station_df)
        assert model._lookup is not None
        assert len(model._lookup) > 0

    def test_fit_sets_global_mean(self, two_station_df):
        model = HistoricalAverageModel().fit(two_station_df)
        expected = two_station_df["num_bikes_available"].mean()
        assert model._global_mean == pytest.approx(expected)

    def test_predict_correct_station_hour_average(self):
        # Controlled dataset: station 1 always has 10 bikes at 09:00
        df = pd.DataFrame(
            {
                "time": pd.date_range("2024-01-08 09:00", periods=3, freq="1D"),
                "station_id": [1, 1, 1],
                "num_bikes_available": [10.0, 10.0, 10.0],
            }
        )
        model = HistoricalAverageModel().fit(df)
        query = pd.DataFrame(
            {
                "time": [pd.Timestamp("2024-01-11 09:00")],
                "station_id": [1],
                "num_bikes_available": [0],  # ignored
            }
        )
        preds = model.predict(query)
        assert preds.iloc[0] == pytest.approx(10.0)

    def test_predict_falls_back_to_station_mean_unknown_hour(self):
        df = pd.DataFrame(
            {
                "time": pd.date_range("2024-01-08 09:00", periods=3, freq="1D"),
                "station_id": [1, 1, 1],
                "num_bikes_available": [10.0, 12.0, 8.0],
            }
        )
        model = HistoricalAverageModel().fit(df)
        # Ask for 03:00 which is not in the training data
        query = pd.DataFrame(
            {
                "time": [pd.Timestamp("2024-01-11 03:00")],
                "station_id": [1],
                "num_bikes_available": [0],
            }
        )
        preds = model.predict(query)
        assert preds.iloc[0] == pytest.approx(10.0)  # station mean

    def test_predict_falls_back_to_global_mean_unknown_station(self):
        df = pd.DataFrame(
            {
                "time": pd.date_range("2024-01-08 09:00", periods=3, freq="1D"),
                "station_id": [1, 1, 1],
                "num_bikes_available": [10.0, 10.0, 10.0],
            }
        )
        model = HistoricalAverageModel().fit(df)
        query = pd.DataFrame(
            {
                "time": [pd.Timestamp("2024-01-11 09:00")],
                "station_id": [999],  # unseen station
                "num_bikes_available": [0],
            }
        )
        preds = model.predict(query)
        assert preds.iloc[0] == pytest.approx(10.0)  # global mean

    def test_predict_with_day_of_week(self):
        # Monday (dow=0) always has 5; Tuesday (dow=1) always has 15
        dates_mon = pd.date_range("2024-01-08 09:00", periods=2, freq="7D")  # Mondays
        dates_tue = pd.date_range("2024-01-09 09:00", periods=2, freq="7D")  # Tuesdays
        df = pd.DataFrame(
            {
                "time": list(dates_mon) + list(dates_tue),
                "station_id": [1] * 4,
                "num_bikes_available": [5.0, 5.0, 15.0, 15.0],
            }
        )
        model = HistoricalAverageModel(use_day_of_week=True).fit(df)

        monday_query = pd.DataFrame(
            {
                "time": [pd.Timestamp("2024-01-22 09:00")],
                "station_id": [1],
                "num_bikes_available": [0],
            }
        )
        tuesday_query = pd.DataFrame(
            {
                "time": [pd.Timestamp("2024-01-23 09:00")],
                "station_id": [1],
                "num_bikes_available": [0],
            }
        )
        assert model.predict(monday_query).iloc[0] == pytest.approx(5.0)
        assert model.predict(tuesday_query).iloc[0] == pytest.approx(15.0)

    def test_predict_series_named_prediction(self, two_station_df):
        model = HistoricalAverageModel().fit(two_station_df)
        assert model.predict(two_station_df).name == "prediction"

    def test_predict_aligned_with_input_index(self, two_station_df):
        model = HistoricalAverageModel().fit(two_station_df)
        preds = model.predict(two_station_df)
        pd.testing.assert_index_equal(preds.index, two_station_df.index)

    def test_predict_no_nans(self, two_station_df):
        model = HistoricalAverageModel().fit(two_station_df)
        assert not model.predict(two_station_df).isna().any()

    def test_predict_before_fit_raises(self, two_station_df):
        model = HistoricalAverageModel()
        with pytest.raises(RuntimeError, match="fit\\(\\)"):
            model.predict(two_station_df)

    def test_evaluate_returns_metric_dict(self, two_station_df):
        model = HistoricalAverageModel().fit(two_station_df)
        metrics = model.evaluate(two_station_df)
        assert set(metrics.keys()) == {"mae", "rmse", "mape"}
