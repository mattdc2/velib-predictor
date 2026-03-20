"""Unit tests for the evaluation framework (evaluator + visualizer)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.evaluation.evaluator import (
    EvaluationResult,
    ModelEvaluator,
    TemporalSplit,
    compute_metrics,
    temporal_split,
)
from src.models.baseline.linear import RidgeModel
from src.models.baseline.persistence import PersistenceModel
from src.models.baseline.tree_models import XGBoostModel

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FEATURES = ["f1", "f2"]
TARGET = "num_bikes_available"


@pytest.fixture
def simple_df() -> pd.DataFrame:
    """300-row synthetic dataset with a single station, sorted by time."""
    rng = np.random.default_rng(7)
    n = 300
    f1 = rng.uniform(0, 10, n)
    f2 = rng.uniform(0, 5, n)
    target = 2 * f1 - f2 + rng.normal(0, 0.5, n)
    times = pd.date_range("2024-01-01", periods=n, freq="1h")
    return pd.DataFrame({"time": times, "station_id": 0, "f1": f1, "f2": f2, TARGET: target})


@pytest.fixture
def split(simple_df) -> TemporalSplit:
    return temporal_split(simple_df, train_frac=0.7, val_frac=0.15)


@pytest.fixture
def evaluator_with_results(split) -> ModelEvaluator:
    ev = ModelEvaluator(target_col=TARGET)
    ev.register("persistence", PersistenceModel(value_col=TARGET, timestamp_col="time"))
    ev.register("ridge", RidgeModel(feature_cols=FEATURES, target_col=TARGET))
    ev.register("xgb", XGBoostModel(feature_cols=FEATURES, target_col=TARGET, n_estimators=10))
    ev.fit_evaluate(split)
    return ev


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------


class TestComputeMetrics:
    def test_perfect_predictions(self):
        y = pd.Series([1.0, 2.0, 3.0])
        m = compute_metrics(y, y)
        assert m["mae"] == pytest.approx(0.0)
        assert m["rmse"] == pytest.approx(0.0)
        assert m["r2"] == pytest.approx(1.0)

    def test_mae_rmse_known_values(self):
        y_true = pd.Series([0.0, 0.0, 0.0])
        y_pred = pd.Series([1.0, 2.0, 3.0])
        m = compute_metrics(y_true, y_pred)
        assert m["mae"] == pytest.approx(2.0)
        assert m["rmse"] == pytest.approx(math.sqrt((1 + 4 + 9) / 3))

    def test_r2_zero_for_mean_predictor(self):
        y_true = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = pd.Series([3.0] * 5)  # predicting the mean
        m = compute_metrics(y_true, y_pred)
        assert m["r2"] == pytest.approx(0.0)

    def test_r2_negative_for_bad_predictor(self):
        y_true = pd.Series([1.0, 2.0, 3.0])
        y_pred = pd.Series([10.0, 10.0, 10.0])
        assert compute_metrics(y_true, y_pred)["r2"] < 0

    def test_mape_avoids_div_zero(self):
        y_true = pd.Series([0.0, 0.0, 0.0])
        y_pred = pd.Series([1.0, 1.0, 1.0])
        m = compute_metrics(y_true, y_pred)
        assert np.isfinite(m["mape"])

    def test_all_nan_returns_nan_dict(self):
        y_true = pd.Series([np.nan, np.nan])
        y_pred = pd.Series([1.0, 2.0])
        m = compute_metrics(y_true, y_pred)
        assert all(np.isnan(v) for v in m.values())

    def test_partial_nan_ignored(self):
        y_true = pd.Series([np.nan, 2.0, 3.0])
        y_pred = pd.Series([1.0, 2.0, 3.0])
        m = compute_metrics(y_true, y_pred)
        assert m["mae"] == pytest.approx(0.0)

    def test_returns_four_keys(self):
        y = pd.Series([1.0, 2.0])
        assert set(compute_metrics(y, y).keys()) == {"mae", "rmse", "mape", "r2"}

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            compute_metrics(pd.Series([1.0, 2.0]), pd.Series([1.0]))

    def test_constant_y_true_r2_nan(self):
        """R² is undefined (NaN) when y_true has zero variance."""
        y_true = pd.Series([5.0, 5.0, 5.0])
        y_pred = pd.Series([5.0, 5.0, 5.0])
        m = compute_metrics(y_true, y_pred)
        assert np.isnan(m["r2"])


# ---------------------------------------------------------------------------
# temporal_split
# ---------------------------------------------------------------------------


class TestTemporalSplit:
    def test_sizes_sum_to_total(self, simple_df):
        s = temporal_split(simple_df, 0.7, 0.15)
        assert len(s.train) + len(s.val) + len(s.test) == len(simple_df)

    def test_train_size_approx(self, simple_df):
        s = temporal_split(simple_df, 0.7, 0.15)
        assert len(s.train) == int(len(simple_df) * 0.7)

    def test_val_size_approx(self, simple_df):
        s = temporal_split(simple_df, 0.7, 0.15)
        assert len(s.val) == int(len(simple_df) * 0.15)

    def test_no_overlap(self, simple_df):
        s = temporal_split(simple_df, 0.7, 0.15)
        # DataFrames are reset-indexed — verify via values not index
        train_times = set(s.train["time"].astype(str))
        val_times = set(s.val["time"].astype(str))
        test_times = set(s.test["time"].astype(str))
        assert len(train_times & val_times) == 0
        assert len(train_times & test_times) == 0
        assert len(val_times & test_times) == 0

    def test_chronological_order(self, simple_df):
        s = temporal_split(simple_df, 0.7, 0.15)
        assert s.train["time"].max() <= s.val["time"].min()
        assert s.val["time"].max() <= s.test["time"].min()

    def test_fractions_dont_sum_to_one_raises(self, simple_df):
        with pytest.raises(ValueError, match="sum to 1.0"):
            temporal_split(simple_df, 0.5, 0.5, test_frac=0.5)

    def test_negative_fraction_raises(self, simple_df):
        with pytest.raises(ValueError, match="in \\[0, 1\\]"):
            temporal_split(simple_df, -0.1, 0.5, test_frac=0.6)

    def test_zero_val_frac(self, simple_df):
        s = temporal_split(simple_df, 0.8, 0.0, test_frac=0.2)
        assert len(s.val) == 0

    def test_train_val_property(self, simple_df):
        s = temporal_split(simple_df, 0.7, 0.15)
        assert len(s.train_val) == len(s.train) + len(s.val)

    def test_stored_fracs(self, simple_df):
        s = temporal_split(simple_df, 0.6, 0.2, test_frac=0.2)
        assert s.train_frac == pytest.approx(0.6)
        assert s.val_frac == pytest.approx(0.2)
        assert s.test_frac == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# EvaluationResult
# ---------------------------------------------------------------------------


class TestEvaluationResult:
    @pytest.fixture
    def result(self) -> EvaluationResult:
        y_true = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = pd.Series([1.1, 1.9, 3.1, 3.9, 5.0])
        metrics = compute_metrics(y_true, y_pred)
        return EvaluationResult(
            model_name="test_model",
            metrics=metrics,
            predictions=y_pred,
            y_true=y_true,
            train_size=100,
            test_size=5,
        )

    def test_residuals_sign(self, result):
        expected = result.predictions - result.y_true
        pd.testing.assert_series_equal(result.residuals, expected.rename("residual"))

    def test_residuals_name(self, result):
        assert result.residuals.name == "residual"

    def test_metrics_series_name(self, result):
        assert result.metrics_series().name == "test_model"

    def test_metrics_series_keys(self, result):
        assert set(result.metrics_series().index) == {"mae", "rmse", "mape", "r2"}


# ---------------------------------------------------------------------------
# ModelEvaluator
# ---------------------------------------------------------------------------


class TestModelEvaluator:
    def test_register_returns_self(self, split):
        ev = ModelEvaluator(target_col=TARGET)
        assert ev.register("p", PersistenceModel()) is ev

    def test_duplicate_name_raises(self, split):
        ev = ModelEvaluator(target_col=TARGET)
        ev.register("p", PersistenceModel())
        with pytest.raises(ValueError, match="already registered"):
            ev.register("p", PersistenceModel())

    def test_fit_evaluate_no_models_raises(self, split):
        ev = ModelEvaluator(target_col=TARGET)
        with pytest.raises(RuntimeError, match="No models registered"):
            ev.fit_evaluate(split)

    def test_fit_evaluate_returns_dict(self, evaluator_with_results):
        assert isinstance(evaluator_with_results.results, dict)

    def test_fit_evaluate_keys_match_registered(self, evaluator_with_results):
        assert set(evaluator_with_results.results.keys()) == {"persistence", "ridge", "xgb"}

    def test_each_result_is_evaluation_result(self, evaluator_with_results):
        for r in evaluator_with_results.results.values():
            assert isinstance(r, EvaluationResult)

    def test_predictions_length_matches_test(self, evaluator_with_results, split):
        for r in evaluator_with_results.results.values():
            assert len(r.predictions) == len(split.test)

    def test_metrics_keys_present(self, evaluator_with_results):
        for r in evaluator_with_results.results.values():
            assert {"mae", "rmse", "mape", "r2"} <= set(r.metrics.keys())

    def test_compare_returns_dataframe(self, evaluator_with_results):
        assert isinstance(evaluator_with_results.compare(), pd.DataFrame)

    def test_compare_columns(self, evaluator_with_results):
        df = evaluator_with_results.compare()
        assert {"model", "mae", "rmse", "mape", "r2", "train_size", "test_size"} <= set(df.columns)

    def test_compare_row_count(self, evaluator_with_results):
        assert len(evaluator_with_results.compare()) == 3

    def test_compare_sorted_by_mae_ascending(self, evaluator_with_results):
        df = evaluator_with_results.compare(sort_by="mae", ascending=True)
        assert df["mae"].is_monotonic_increasing

    def test_compare_before_fit_raises(self, split):
        ev = ModelEvaluator(target_col=TARGET)
        ev.register("p", PersistenceModel())
        with pytest.raises(RuntimeError, match="No results"):
            ev.compare()

    def test_best_model_returns_tuple(self, evaluator_with_results):
        name, result = evaluator_with_results.best_model(metric="mae")
        assert isinstance(name, str)
        assert isinstance(result, EvaluationResult)

    def test_best_model_mae_is_minimum(self, evaluator_with_results):
        best_name, _ = evaluator_with_results.best_model(metric="mae")
        comparison = evaluator_with_results.compare(sort_by="mae")
        assert best_name == comparison.iloc[0]["model"]

    def test_best_model_r2_is_maximum(self, evaluator_with_results):
        best_name, _ = evaluator_with_results.best_model(metric="r2")
        comparison = evaluator_with_results.compare(sort_by="r2", ascending=False)
        assert best_name == comparison.iloc[0]["model"]

    def test_best_model_before_fit_raises(self, split):
        ev = ModelEvaluator(target_col=TARGET)
        ev.register("p", PersistenceModel())
        with pytest.raises(RuntimeError, match="No results"):
            ev.best_model()

    def test_best_model_unknown_metric_raises(self, evaluator_with_results):
        with pytest.raises(ValueError, match="Unknown metric"):
            evaluator_with_results.best_model(metric="unknown")

    def test_fit_on_train_val_uses_larger_set(self, simple_df):
        split_ = temporal_split(simple_df, 0.7, 0.15)
        ev_train = ModelEvaluator(target_col=TARGET, fit_on_train_val=False)
        ev_trainval = ModelEvaluator(target_col=TARGET, fit_on_train_val=True)
        for ev in (ev_train, ev_trainval):
            ev.register("ridge", RidgeModel(feature_cols=FEATURES, target_col=TARGET))
        ev_train.fit_evaluate(split_)
        ev_trainval.fit_evaluate(split_)
        assert ev_trainval.results["ridge"].train_size > ev_train.results["ridge"].train_size

    def test_train_size_stored_correctly(self, evaluator_with_results, split):
        for r in evaluator_with_results.results.values():
            assert r.train_size == len(split.train)

    def test_test_size_stored_correctly(self, evaluator_with_results, split):
        for r in evaluator_with_results.results.values():
            assert r.test_size == len(split.test)


# ---------------------------------------------------------------------------
# Visualizer (smoke tests — verify figures are created without errors)
# ---------------------------------------------------------------------------


class TestVisualizer:
    """Smoke tests: verify plots are created without raising exceptions."""

    @pytest.fixture
    def ev(self, evaluator_with_results) -> ModelEvaluator:
        return evaluator_with_results

    def test_plot_predictions_returns_figure(self, ev):
        import matplotlib.figure

        from src.evaluation.visualizer import plot_predictions

        result = next(iter(ev.results.values()))
        fig = plot_predictions(result)
        assert isinstance(fig, matplotlib.figure.Figure)
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_residuals_returns_figure(self, ev):
        import matplotlib.figure

        from src.evaluation.visualizer import plot_residuals

        result = next(iter(ev.results.values()))
        fig = plot_residuals(result)
        assert isinstance(fig, matplotlib.figure.Figure)
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_metric_comparison_returns_figure(self, ev):
        import matplotlib.figure

        from src.evaluation.visualizer import plot_metric_comparison

        fig = plot_metric_comparison(ev)
        assert isinstance(fig, matplotlib.figure.Figure)
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_metric_comparison_custom_metrics(self, ev):
        import matplotlib.figure

        from src.evaluation.visualizer import plot_metric_comparison

        fig = plot_metric_comparison(ev, metrics=["mae", "r2"])
        assert isinstance(fig, matplotlib.figure.Figure)
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_dashboard_returns_figure(self, ev):
        import matplotlib.figure

        from src.evaluation.visualizer import plot_dashboard

        fig = plot_dashboard(ev)
        assert isinstance(fig, matplotlib.figure.Figure)
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_dashboard_explicit_model(self, ev):
        import matplotlib.figure

        from src.evaluation.visualizer import plot_dashboard

        fig = plot_dashboard(ev, best_model_name="ridge")
        assert isinstance(fig, matplotlib.figure.Figure)
        import matplotlib.pyplot as plt

        plt.close(fig)
