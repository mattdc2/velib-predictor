"""Unit tests for tree-based baseline models (XGBoost and LightGBM)."""

import numpy as np
import pandas as pd
import pytest

from src.models.baseline.tree_models import (
    LightGBMModel,
    PerStationTreeModel,
    XGBoostModel,
    compare_per_station_vs_global,
    cross_validate_tree_model,
    tune_hyperparams,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FEATURES = ["f1", "f2", "f3"]
TARGET = "num_bikes_available"
STATION_COL = "station_id"


@pytest.fixture
def tree_df() -> pd.DataFrame:
    """Synthetic dataset with a non-linear relationship: target = f1² - f2 + noise."""
    rng = np.random.default_rng(0)
    n = 200
    f1 = rng.uniform(0, 5, n)
    f2 = rng.uniform(0, 3, n)
    f3 = rng.uniform(-1, 1, n)
    noise = rng.normal(0, 0.2, n)
    target = f1**2 - f2 + noise
    times = pd.date_range("2024-01-01", periods=n, freq="1h")
    return pd.DataFrame({"time": times, "f1": f1, "f2": f2, "f3": f3, TARGET: target})


@pytest.fixture
def multi_station_df() -> pd.DataFrame:
    """Dataset with 5 stations, 60 time steps, sorted by time so all stations appear
    in both train and test when doing a temporal split."""
    rng = np.random.default_rng(1)
    times = pd.date_range("2024-01-01", periods=60, freq="1h")
    dfs = []
    for sid in range(5):
        f1 = rng.uniform(0, 5, 60)
        f2 = rng.uniform(0, 3, 60)
        f3 = rng.uniform(-1, 1, 60)
        target = f1**2 - f2 + rng.normal(0, 0.3, 60)
        dfs.append(
            pd.DataFrame(
                {
                    "time": times,
                    STATION_COL: sid,
                    "f1": f1,
                    "f2": f2,
                    "f3": f3,
                    TARGET: target,
                }
            )
        )
    # Interleave by sorting on time: each timestamp has one row per station,
    # so any temporal split covers all stations in both halves.
    return pd.concat(dfs, ignore_index=True).sort_values("time").reset_index(drop=True)


@pytest.fixture
def df_with_nans(tree_df) -> pd.DataFrame:
    df = tree_df.copy()
    df.loc[[3, 7, 15], "f1"] = np.nan
    return df


# ---------------------------------------------------------------------------
# Shared behaviour across XGBoostModel and LightGBMModel
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ModelClass", [XGBoostModel, LightGBMModel])
class TestSharedTreeBehaviour:
    def test_fit_returns_self(self, tree_df, ModelClass):
        model = ModelClass(feature_cols=FEATURES)
        assert model.fit(tree_df) is model

    def test_predict_series_named_prediction(self, tree_df, ModelClass):
        model = ModelClass(feature_cols=FEATURES).fit(tree_df)
        assert model.predict(tree_df).name == "prediction"

    def test_predict_length_matches_input(self, tree_df, ModelClass):
        model = ModelClass(feature_cols=FEATURES).fit(tree_df)
        assert len(model.predict(tree_df)) == len(tree_df)

    def test_predict_index_aligned(self, tree_df, ModelClass):
        model = ModelClass(feature_cols=FEATURES).fit(tree_df)
        pd.testing.assert_index_equal(model.predict(tree_df).index, tree_df.index)

    def test_predict_before_fit_raises(self, tree_df, ModelClass):
        with pytest.raises(RuntimeError, match="fit\\(\\)"):
            ModelClass(feature_cols=FEATURES).predict(tree_df)

    def test_predict_nan_rows_give_nan(self, df_with_nans, ModelClass):
        model = ModelClass(feature_cols=FEATURES).fit(df_with_nans)
        preds = model.predict(df_with_nans)
        assert preds.loc[[3, 7, 15]].isna().all()

    def test_predict_non_nan_rows_have_values(self, df_with_nans, ModelClass):
        model = ModelClass(feature_cols=FEATURES).fit(df_with_nans)
        preds = model.predict(df_with_nans)
        valid_idx = df_with_nans.index.difference([3, 7, 15])
        assert preds.loc[valid_idx].notna().all()

    def test_auto_feature_detection(self, tree_df, ModelClass):
        model = ModelClass().fit(tree_df)
        assert "f1" in model._fitted_feature_cols
        assert TARGET not in model._fitted_feature_cols

    def test_evaluate_returns_metric_keys(self, tree_df, ModelClass):
        model = ModelClass(feature_cols=FEATURES).fit(tree_df)
        assert {"mae", "rmse", "mape"} == set(model.evaluate(tree_df).keys())

    def test_evaluate_mae_finite(self, tree_df, ModelClass):
        model = ModelClass(feature_cols=FEATURES).fit(tree_df)
        assert np.isfinite(model.evaluate(tree_df)["mae"])

    def test_feature_importance_before_fit_raises(self, ModelClass):
        with pytest.raises(RuntimeError, match="fit\\(\\)"):
            ModelClass(feature_cols=FEATURES).feature_importance()

    def test_feature_importance_columns(self, tree_df, ModelClass):
        model = ModelClass(feature_cols=FEATURES).fit(tree_df)
        fi = model.feature_importance()
        assert list(fi.columns) == ["feature", "importance"]

    def test_feature_importance_length(self, tree_df, ModelClass):
        model = ModelClass(feature_cols=FEATURES).fit(tree_df)
        assert len(model.feature_importance()) == len(FEATURES)

    def test_feature_importance_sorted_descending(self, tree_df, ModelClass):
        model = ModelClass(feature_cols=FEATURES).fit(tree_df)
        fi = model.feature_importance()
        assert fi["importance"].is_monotonic_decreasing

    def test_feature_importance_covers_all_features(self, tree_df, ModelClass):
        model = ModelClass(feature_cols=FEATURES).fit(tree_df)
        assert set(model.feature_importance()["feature"]) == set(FEATURES)

    def test_feature_importance_nonnegative(self, tree_df, ModelClass):
        model = ModelClass(feature_cols=FEATURES).fit(tree_df)
        assert (model.feature_importance()["importance"] >= 0).all()

    def test_mae_lower_than_naive(self, tree_df, ModelClass):
        """Tree model should outperform a constant (mean) predictor."""
        model = ModelClass(feature_cols=FEATURES).fit(tree_df)
        mae = model.evaluate(tree_df)["mae"]
        naive_mae = float((tree_df[TARGET] - tree_df[TARGET].mean()).abs().mean())
        assert mae < naive_mae

    def test_captures_nonlinearity(self, tree_df, ModelClass):
        """The non-linear target (f1²) should yield lower MAE than a linear baseline."""
        from src.models.baseline.linear import LinearModel

        tree_model = ModelClass(feature_cols=FEATURES).fit(tree_df)
        linear_model = LinearModel(feature_cols=FEATURES).fit(tree_df)
        assert tree_model.evaluate(tree_df)["mae"] < linear_model.evaluate(tree_df)["mae"]


# ---------------------------------------------------------------------------
# XGBoostModel specifics
# ---------------------------------------------------------------------------


class TestXGBoostModel:
    def test_default_hyperparams_stored(self):
        model = XGBoostModel()
        assert model._estimator_kwargs["n_estimators"] == 100
        assert model._estimator_kwargs["max_depth"] == 6
        assert model._estimator_kwargs["learning_rate"] == pytest.approx(0.1)

    def test_custom_hyperparams_forwarded(self, tree_df):
        model = XGBoostModel(n_estimators=10, max_depth=3, feature_cols=FEATURES).fit(tree_df)
        assert model._estimator is not None
        assert model.predict(tree_df).notna().any()

    def test_more_estimators_reduces_train_mae(self, tree_df):
        low = XGBoostModel(n_estimators=5, feature_cols=FEATURES).fit(tree_df)
        high = XGBoostModel(n_estimators=200, feature_cols=FEATURES).fit(tree_df)
        assert high.evaluate(tree_df)["mae"] <= low.evaluate(tree_df)["mae"]


# ---------------------------------------------------------------------------
# LightGBMModel specifics
# ---------------------------------------------------------------------------


class TestLightGBMModel:
    def test_default_hyperparams_stored(self):
        model = LightGBMModel()
        assert model._estimator_kwargs["n_estimators"] == 100
        assert model._estimator_kwargs["num_leaves"] == 31

    def test_custom_hyperparams_forwarded(self, tree_df):
        model = LightGBMModel(n_estimators=10, num_leaves=8, feature_cols=FEATURES).fit(tree_df)
        assert model._estimator is not None
        assert model.predict(tree_df).notna().any()

    def test_more_estimators_reduces_train_mae(self, tree_df):
        low = LightGBMModel(n_estimators=5, feature_cols=FEATURES).fit(tree_df)
        high = LightGBMModel(n_estimators=200, feature_cols=FEATURES).fit(tree_df)
        assert high.evaluate(tree_df)["mae"] <= low.evaluate(tree_df)["mae"]


# ---------------------------------------------------------------------------
# PerStationTreeModel
# ---------------------------------------------------------------------------


class TestPerStationTreeModel:
    @pytest.mark.parametrize("ModelClass", [XGBoostModel, LightGBMModel])
    def test_fit_returns_self(self, multi_station_df, ModelClass):
        model = PerStationTreeModel(ModelClass, feature_cols=FEATURES)
        assert model.fit(multi_station_df) is model

    @pytest.mark.parametrize("ModelClass", [XGBoostModel, LightGBMModel])
    def test_predict_series_named_prediction(self, multi_station_df, ModelClass):
        model = PerStationTreeModel(ModelClass, feature_cols=FEATURES).fit(multi_station_df)
        assert model.predict(multi_station_df).name == "prediction"

    @pytest.mark.parametrize("ModelClass", [XGBoostModel, LightGBMModel])
    def test_predict_length_matches_input(self, multi_station_df, ModelClass):
        model = PerStationTreeModel(ModelClass, feature_cols=FEATURES).fit(multi_station_df)
        assert len(model.predict(multi_station_df)) == len(multi_station_df)

    @pytest.mark.parametrize("ModelClass", [XGBoostModel, LightGBMModel])
    def test_predict_index_aligned(self, multi_station_df, ModelClass):
        model = PerStationTreeModel(ModelClass, feature_cols=FEATURES).fit(multi_station_df)
        pd.testing.assert_index_equal(model.predict(multi_station_df).index, multi_station_df.index)

    @pytest.mark.parametrize("ModelClass", [XGBoostModel, LightGBMModel])
    def test_predict_before_fit_raises(self, ModelClass):
        with pytest.raises(RuntimeError, match="fit\\(\\)"):
            PerStationTreeModel(ModelClass, feature_cols=FEATURES).predict(pd.DataFrame())

    def test_trains_per_station_models(self, multi_station_df):
        model = PerStationTreeModel(XGBoostModel, feature_cols=FEATURES, min_train_samples=30).fit(
            multi_station_df
        )
        # 5 stations × 60 rows each → all above threshold
        assert len(model._station_models) == 5

    def test_sparse_station_uses_global_fallback(self, multi_station_df):
        """Stations below min_train_samples should not get a per-station model."""
        model = PerStationTreeModel(XGBoostModel, feature_cols=FEATURES, min_train_samples=100).fit(
            multi_station_df
        )
        # 60 rows < 100 → no per-station models, all fall back to global
        assert len(model._station_models) == 0
        assert model._global_model is not None

    def test_unseen_station_uses_global_model(self, multi_station_df, tree_df):
        """Predictions for an unseen station should use the global fallback."""
        model = PerStationTreeModel(XGBoostModel, feature_cols=FEATURES, min_train_samples=1).fit(
            multi_station_df
        )

        unseen_df = tree_df.copy()
        unseen_df[STATION_COL] = 999  # station not in training set
        preds = model.predict(unseen_df)
        assert preds.notna().any()

    def test_feature_importance_returns_expected_columns(self, multi_station_df):
        model = PerStationTreeModel(XGBoostModel, feature_cols=FEATURES).fit(multi_station_df)
        fi = model.feature_importance()
        assert list(fi.columns) == ["feature", "importance"]
        assert set(fi["feature"]) == set(FEATURES)

    def test_feature_importance_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="fit\\(\\)"):
            PerStationTreeModel(XGBoostModel, feature_cols=FEATURES).feature_importance()

    def test_feature_importance_sorted_descending(self, multi_station_df):
        model = PerStationTreeModel(XGBoostModel, feature_cols=FEATURES).fit(multi_station_df)
        fi = model.feature_importance()
        assert fi["importance"].is_monotonic_decreasing


# ---------------------------------------------------------------------------
# tune_hyperparams
# ---------------------------------------------------------------------------


class TestTuneHyperparams:
    def test_returns_expected_keys(self, tree_df):
        result = tune_hyperparams(
            XGBoostModel,
            tree_df,
            FEATURES,
            param_grid={"n_estimators": [10, 20], "max_depth": [3]},
            n_splits=2,
        )
        assert {"best_params", "best_score", "all_results"} == set(result.keys())

    def test_best_params_is_dict(self, tree_df):
        result = tune_hyperparams(
            XGBoostModel,
            tree_df,
            FEATURES,
            param_grid={"n_estimators": [10]},
            n_splits=2,
        )
        assert isinstance(result["best_params"], dict)
        assert "n_estimators" in result["best_params"]

    def test_all_results_length(self, tree_df):
        result = tune_hyperparams(
            XGBoostModel,
            tree_df,
            FEATURES,
            param_grid={"n_estimators": [5, 10], "max_depth": [3, 6]},
            n_splits=2,
        )
        # 2 × 2 = 4 combinations
        assert len(result["all_results"]) == 4

    def test_best_score_is_finite(self, tree_df):
        result = tune_hyperparams(
            XGBoostModel,
            tree_df,
            FEATURES,
            param_grid={"n_estimators": [10]},
            n_splits=2,
        )
        assert np.isfinite(result["best_score"])

    def test_best_score_matches_min_mean_mae(self, tree_df):
        result = tune_hyperparams(
            XGBoostModel,
            tree_df,
            FEATURES,
            param_grid={"n_estimators": [5, 50]},
            n_splits=2,
        )
        min_mean_mae = min(r["mean_mae"] for r in result["all_results"])
        assert result["best_score"] == pytest.approx(min_mean_mae)

    def test_works_with_lightgbm(self, tree_df):
        result = tune_hyperparams(
            LightGBMModel,
            tree_df,
            FEATURES,
            param_grid={"n_estimators": [10], "num_leaves": [8]},
            n_splits=2,
        )
        assert np.isfinite(result["best_score"])


# ---------------------------------------------------------------------------
# cross_validate_tree_model
# ---------------------------------------------------------------------------


class TestCrossValidateTreeModel:
    def test_returns_expected_keys(self, tree_df):
        result = cross_validate_tree_model(XGBoostModel, tree_df, FEATURES, n_splits=3)
        expected = {
            "fold_metrics",
            "mean_mae",
            "std_mae",
            "mean_rmse",
            "std_rmse",
            "mean_mape",
            "std_mape",
        }
        assert expected == set(result.keys())

    def test_fold_count(self, tree_df):
        result = cross_validate_tree_model(XGBoostModel, tree_df, FEATURES, n_splits=3)
        assert len(result["fold_metrics"]) == 3

    def test_each_fold_has_metric_keys(self, tree_df):
        result = cross_validate_tree_model(XGBoostModel, tree_df, FEATURES, n_splits=3)
        for fold in result["fold_metrics"]:
            assert {"mae", "rmse", "mape"} == set(fold.keys())

    def test_mean_mae_finite(self, tree_df):
        result = cross_validate_tree_model(XGBoostModel, tree_df, FEATURES, n_splits=3)
        assert np.isfinite(result["mean_mae"])

    def test_std_nonnegative(self, tree_df):
        result = cross_validate_tree_model(XGBoostModel, tree_df, FEATURES, n_splits=3)
        assert result["std_mae"] >= 0
        assert result["std_rmse"] >= 0

    def test_works_with_lightgbm(self, tree_df):
        result = cross_validate_tree_model(LightGBMModel, tree_df, FEATURES, n_splits=3)
        assert np.isfinite(result["mean_rmse"])

    def test_model_kwargs_forwarded(self, tree_df):
        result = cross_validate_tree_model(
            XGBoostModel,
            tree_df,
            FEATURES,
            n_splits=3,
            model_kwargs={"n_estimators": 10, "max_depth": 3},
        )
        assert np.isfinite(result["mean_mae"])

    def test_mean_mae_lower_than_naive(self, tree_df):
        result = cross_validate_tree_model(XGBoostModel, tree_df, FEATURES, n_splits=3)
        target_std = tree_df[TARGET].std()
        assert result["mean_mae"] < target_std


# ---------------------------------------------------------------------------
# compare_per_station_vs_global
# ---------------------------------------------------------------------------


class TestComparePerStationVsGlobal:
    def test_returns_expected_keys(self, multi_station_df):
        result = compare_per_station_vs_global(
            XGBoostModel,
            multi_station_df,
            FEATURES,
            station_col=STATION_COL,
            model_kwargs={"n_estimators": 10},
        )
        assert {"global_metrics", "per_station_metrics", "station_comparison", "winner"} == set(
            result.keys()
        )

    def test_global_metrics_keys(self, multi_station_df):
        result = compare_per_station_vs_global(
            XGBoostModel,
            multi_station_df,
            FEATURES,
            station_col=STATION_COL,
            model_kwargs={"n_estimators": 10},
        )
        assert {"mae", "rmse", "mape"} == set(result["global_metrics"].keys())

    def test_station_comparison_columns(self, multi_station_df):
        result = compare_per_station_vs_global(
            XGBoostModel,
            multi_station_df,
            FEATURES,
            station_col=STATION_COL,
            model_kwargs={"n_estimators": 10},
        )
        df = result["station_comparison"]
        assert {"station_id", "global_mae", "per_station_mae", "mae_delta", "n_test_rows"} <= set(
            df.columns
        )

    def test_station_comparison_row_count(self, multi_station_df):
        result = compare_per_station_vs_global(
            XGBoostModel,
            multi_station_df,
            FEATURES,
            station_col=STATION_COL,
            model_kwargs={"n_estimators": 10},
        )
        # One row per station in the test set
        assert len(result["station_comparison"]) == 5

    def test_winner_is_valid_string(self, multi_station_df):
        result = compare_per_station_vs_global(
            XGBoostModel,
            multi_station_df,
            FEATURES,
            station_col=STATION_COL,
            model_kwargs={"n_estimators": 10},
        )
        assert result["winner"] in {"global", "per_station"}

    def test_winner_consistent_with_metrics(self, multi_station_df):
        result = compare_per_station_vs_global(
            XGBoostModel,
            multi_station_df,
            FEATURES,
            station_col=STATION_COL,
            model_kwargs={"n_estimators": 10},
        )
        g_mae = result["global_metrics"]["mae"]
        ps_mae = result["per_station_metrics"]["mae"]
        expected_winner = "per_station" if ps_mae < g_mae else "global"
        assert result["winner"] == expected_winner

    def test_mae_delta_consistent(self, multi_station_df):
        """mae_delta = per_station_mae - global_mae for each station."""
        result = compare_per_station_vs_global(
            XGBoostModel,
            multi_station_df,
            FEATURES,
            station_col=STATION_COL,
            model_kwargs={"n_estimators": 10},
        )
        df = result["station_comparison"]
        expected_delta = df["per_station_mae"] - df["global_mae"]
        pd.testing.assert_series_equal(
            df["mae_delta"].reset_index(drop=True),
            expected_delta.reset_index(drop=True),
            check_names=False,
        )

    def test_works_with_lightgbm(self, multi_station_df):
        result = compare_per_station_vs_global(
            LightGBMModel,
            multi_station_df,
            FEATURES,
            station_col=STATION_COL,
            model_kwargs={"n_estimators": 10},
        )
        assert np.isfinite(result["global_metrics"]["mae"])
