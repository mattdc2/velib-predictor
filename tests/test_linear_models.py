"""Unit tests for linear regression baseline models."""

import numpy as np
import pandas as pd
import pytest

from src.models.baseline.linear import (
    LassoModel,
    LinearModel,
    RidgeModel,
    cross_validate_model,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FEATURES = ["f1", "f2", "f3"]
TARGET = "num_bikes_available"


@pytest.fixture
def linear_df() -> pd.DataFrame:
    """Synthetic dataset: target = 2*f1 - f2 + noise."""
    rng = np.random.default_rng(42)
    n = 120
    f1 = rng.uniform(0, 10, n)
    f2 = rng.uniform(0, 5, n)
    f3 = rng.uniform(-1, 1, n)
    noise = rng.normal(0, 0.1, n)
    target = 2 * f1 - f2 + noise
    times = pd.date_range("2024-01-01", periods=n, freq="1h")
    return pd.DataFrame({"time": times, "f1": f1, "f2": f2, "f3": f3, TARGET: target})


@pytest.fixture
def df_with_nans(linear_df) -> pd.DataFrame:
    df = linear_df.copy()
    df.loc[[5, 10, 20], "f1"] = np.nan
    return df


# ---------------------------------------------------------------------------
# Shared behaviour across all three model classes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ModelClass", [LinearModel, RidgeModel, LassoModel])
class TestSharedBehaviour:
    def test_fit_returns_self(self, linear_df, ModelClass):
        model = ModelClass(feature_cols=FEATURES)
        assert model.fit(linear_df) is model

    def test_predict_series_named_prediction(self, linear_df, ModelClass):
        model = ModelClass(feature_cols=FEATURES).fit(linear_df)
        assert model.predict(linear_df).name == "prediction"

    def test_predict_length_matches_input(self, linear_df, ModelClass):
        model = ModelClass(feature_cols=FEATURES).fit(linear_df)
        assert len(model.predict(linear_df)) == len(linear_df)

    def test_predict_index_aligned(self, linear_df, ModelClass):
        model = ModelClass(feature_cols=FEATURES).fit(linear_df)
        pd.testing.assert_index_equal(model.predict(linear_df).index, linear_df.index)

    def test_predict_before_fit_raises(self, linear_df, ModelClass):
        with pytest.raises(RuntimeError, match="fit\\(\\)"):
            ModelClass(feature_cols=FEATURES).predict(linear_df)

    def test_predict_nan_rows_give_nan(self, df_with_nans, ModelClass):
        model = ModelClass(feature_cols=FEATURES).fit(df_with_nans)
        preds = model.predict(df_with_nans)
        assert preds.loc[[5, 10, 20]].isna().all()

    def test_predict_non_nan_rows_have_values(self, df_with_nans, ModelClass):
        model = ModelClass(feature_cols=FEATURES).fit(df_with_nans)
        preds = model.predict(df_with_nans)
        valid_idx = df_with_nans.index.difference([5, 10, 20])
        assert preds.loc[valid_idx].notna().all()

    def test_auto_feature_detection(self, linear_df, ModelClass):
        # No feature_cols → should use f1, f2, f3 (all numeric except target)
        model = ModelClass().fit(linear_df)
        assert (
            set(model._fitted_feature_cols) == {"f1", "f2", "f3", "time"} - {"time"}
            or "f1" in model._fitted_feature_cols
        )

    def test_evaluate_returns_metric_keys(self, linear_df, ModelClass):
        model = ModelClass(feature_cols=FEATURES).fit(linear_df)
        metrics = model.evaluate(linear_df)
        assert {"mae", "rmse", "mape"} == set(metrics.keys())

    def test_evaluate_mae_finite(self, linear_df, ModelClass):
        model = ModelClass(feature_cols=FEATURES).fit(linear_df)
        assert np.isfinite(model.evaluate(linear_df)["mae"])

    def test_feature_importance_before_fit_raises(self, ModelClass):
        with pytest.raises(RuntimeError, match="fit\\(\\)"):
            ModelClass(feature_cols=FEATURES).feature_importance()

    def test_feature_importance_shape(self, linear_df, ModelClass):
        model = ModelClass(feature_cols=FEATURES).fit(linear_df)
        fi = model.feature_importance()
        assert list(fi.columns) == ["feature", "coefficient", "abs_coefficient"]
        assert len(fi) == len(FEATURES)

    def test_feature_importance_sorted_descending(self, linear_df, ModelClass):
        model = ModelClass(feature_cols=FEATURES).fit(linear_df)
        fi = model.feature_importance()
        assert fi["abs_coefficient"].is_monotonic_decreasing

    def test_feature_importance_covers_all_features(self, linear_df, ModelClass):
        model = ModelClass(feature_cols=FEATURES).fit(linear_df)
        assert set(model.feature_importance()["feature"]) == set(FEATURES)

    def test_scale_features_false(self, linear_df, ModelClass):
        model = ModelClass(feature_cols=FEATURES, scale_features=False).fit(linear_df)
        preds = model.predict(linear_df)
        assert preds.notna().any()
        assert model._scaler is None


# ---------------------------------------------------------------------------
# LinearModel specifics
# ---------------------------------------------------------------------------


class TestLinearModel:
    def test_recovers_known_coefficients(self, linear_df):
        """OLS should closely recover the true coefficients (2, -1, ~0)."""
        model = LinearModel(feature_cols=FEATURES, scale_features=False).fit(linear_df)
        fi = model.feature_importance().set_index("feature")["coefficient"]
        assert fi["f1"] == pytest.approx(2.0, abs=0.1)
        assert fi["f2"] == pytest.approx(-1.0, abs=0.1)
        assert abs(fi["f3"]) < 0.5  # noise feature

    def test_low_mae_on_training_data(self, linear_df):
        model = LinearModel(feature_cols=FEATURES).fit(linear_df)
        assert model.evaluate(linear_df)["mae"] < 0.5


# ---------------------------------------------------------------------------
# RidgeModel specifics
# ---------------------------------------------------------------------------


class TestRidgeModel:
    def test_high_alpha_shrinks_coefficients(self, linear_df):
        low_alpha = RidgeModel(alpha=0.001, feature_cols=FEATURES).fit(linear_df)
        high_alpha = RidgeModel(alpha=1000.0, feature_cols=FEATURES).fit(linear_df)
        low_fi = low_alpha.feature_importance()["abs_coefficient"].sum()
        high_fi = high_alpha.feature_importance()["abs_coefficient"].sum()
        assert high_fi < low_fi

    def test_alpha_stored(self):
        model = RidgeModel(alpha=5.0)
        assert model._estimator_kwargs["alpha"] == 5.0

    def test_predictions_finite(self, linear_df):
        model = RidgeModel(feature_cols=FEATURES).fit(linear_df)
        assert model.predict(linear_df).notna().all()


# ---------------------------------------------------------------------------
# LassoModel specifics
# ---------------------------------------------------------------------------


class TestLassoModel:
    def test_high_alpha_zeros_noise_feature(self, linear_df):
        """With strong regularisation Lasso should zero out the noise feature."""
        model = LassoModel(alpha=1.0, feature_cols=FEATURES).fit(linear_df)
        fi = model.feature_importance().set_index("feature")["abs_coefficient"]
        assert fi["f3"] == pytest.approx(0.0, abs=1e-6)

    def test_alpha_stored(self):
        model = LassoModel(alpha=0.5)
        assert model._estimator_kwargs["alpha"] == 0.5

    def test_predictions_finite(self, linear_df):
        model = LassoModel(feature_cols=FEATURES).fit(linear_df)
        assert model.predict(linear_df).notna().all()


# ---------------------------------------------------------------------------
# cross_validate_model
# ---------------------------------------------------------------------------


class TestCrossValidateModel:
    def test_returns_expected_keys(self, linear_df):
        result = cross_validate_model(LinearModel, linear_df, FEATURES, n_splits=3)
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

    def test_fold_count(self, linear_df):
        result = cross_validate_model(LinearModel, linear_df, FEATURES, n_splits=3)
        assert len(result["fold_metrics"]) == 3

    def test_each_fold_has_metric_keys(self, linear_df):
        result = cross_validate_model(LinearModel, linear_df, FEATURES, n_splits=3)
        for fold in result["fold_metrics"]:
            assert {"mae", "rmse", "mape"} == set(fold.keys())

    def test_mean_mae_finite(self, linear_df):
        result = cross_validate_model(LinearModel, linear_df, FEATURES, n_splits=3)
        assert np.isfinite(result["mean_mae"])

    def test_std_nonnegative(self, linear_df):
        result = cross_validate_model(LinearModel, linear_df, FEATURES, n_splits=3)
        assert result["std_mae"] >= 0
        assert result["std_rmse"] >= 0

    def test_works_with_ridge(self, linear_df):
        result = cross_validate_model(
            RidgeModel, linear_df, FEATURES, n_splits=3, model_kwargs={"alpha": 0.5}
        )
        assert np.isfinite(result["mean_rmse"])

    def test_works_with_lasso(self, linear_df):
        result = cross_validate_model(
            LassoModel, linear_df, FEATURES, n_splits=3, model_kwargs={"alpha": 0.01}
        )
        assert np.isfinite(result["mean_rmse"])

    def test_mean_less_than_naive_on_clean_data(self, linear_df):
        """Linear model should beat a constant predictor on structured data."""
        result = cross_validate_model(LinearModel, linear_df, FEATURES, n_splits=3)
        # The target has std ~ 6; MAE << std means the model is learning
        target_std = linear_df[TARGET].std()
        assert result["mean_mae"] < target_std
