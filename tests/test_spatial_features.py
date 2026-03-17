"""Unit tests for spatial feature engineering module."""

import numpy as np
import pandas as pd
import pytest

from src.data.features.spatial import (PARIS_LAT, PARIS_LON,
                                       add_all_spatial_features,
                                       add_cluster_labels,
                                       add_distance_to_center,
                                       add_knn_features,
                                       add_neighborhood_averages,
                                       compute_distance_matrix, haversine_km)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def paris_stations() -> pd.DataFrame:
    """Twelve realistic Velib stations spread across Paris arrondissements."""
    return pd.DataFrame(
        {
            "station_id": list(range(1, 13)),
            "lat": [
                48.8566,
                48.8606,
                48.8650,
                48.8530,
                48.8490,
                48.8720,
                48.8460,
                48.8800,
                48.8400,
                48.8700,
                48.8350,
                48.8850,
            ],
            "lon": [
                2.3522,
                2.3477,
                2.3400,
                2.3600,
                2.3700,
                2.3300,
                2.3800,
                2.3200,
                2.3900,
                2.3100,
                2.4000,
                2.3000,
            ],
            "capacity": [20, 25, 30, 15, 20, 25, 20, 30, 15, 25, 20, 30],
            "bikes_available": [10, 12, 20, 5, 8, 15, 10, 25, 7, 18, 9, 22],
        }
    )


@pytest.fixture
def two_points() -> pd.DataFrame:
    """Two stations whose distance can be verified against a known value."""
    # Paris centre (48.8566, 2.3522) and a point ~1 km north-east
    return pd.DataFrame(
        {
            "station_id": [1, 2],
            "lat": [48.8566, 48.8656],
            "lon": [2.3522, 2.3522],
        }
    )


# ---------------------------------------------------------------------------
# haversine_km
# ---------------------------------------------------------------------------


class TestHaversineKm:
    def test_same_point_is_zero(self):
        assert haversine_km(48.8566, 2.3522, 48.8566, 2.3522) == pytest.approx(0.0)

    def test_known_distance(self):
        # Roughly 1 degree of latitude ≈ 111 km
        d = haversine_km(0.0, 0.0, 1.0, 0.0)
        assert d == pytest.approx(111.195, abs=0.1)

    def test_symmetry(self):
        d1 = haversine_km(48.8566, 2.3522, 48.9000, 2.4000)
        d2 = haversine_km(48.9000, 2.4000, 48.8566, 2.3522)
        assert d1 == pytest.approx(d2)

    def test_vectorised(self):
        lats = np.array([48.8566, 48.9000])
        lons = np.array([2.3522, 2.4000])
        result = haversine_km(lats, lons, PARIS_LAT, PARIS_LON)
        assert result.shape == (2,)
        assert result[0] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# compute_distance_matrix
# ---------------------------------------------------------------------------


class TestComputeDistanceMatrix:
    def test_shape(self, paris_stations):
        n = len(paris_stations)
        D = compute_distance_matrix(paris_stations)
        assert D.shape == (n, n)

    def test_diagonal_is_zero(self, paris_stations):
        D = compute_distance_matrix(paris_stations)
        np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-9)

    def test_symmetric(self, paris_stations):
        D = compute_distance_matrix(paris_stations)
        np.testing.assert_allclose(D, D.T, atol=1e-9)

    def test_non_negative(self, paris_stations):
        D = compute_distance_matrix(paris_stations)
        assert (D >= 0).all()


# ---------------------------------------------------------------------------
# add_distance_to_center
# ---------------------------------------------------------------------------


class TestAddDistanceToCentre:
    def test_column_added(self, paris_stations):
        result = add_distance_to_center(paris_stations)
        assert "dist_to_center_km" in result.columns

    def test_paris_centre_station_is_near_zero(self, paris_stations):
        # Station 1 sits exactly at PARIS_LAT / PARIS_LON
        result = add_distance_to_center(paris_stations)
        assert result.loc[0, "dist_to_center_km"] == pytest.approx(0.0, abs=1e-6)

    def test_all_distances_non_negative(self, paris_stations):
        result = add_distance_to_center(paris_stations)
        assert (result["dist_to_center_km"] >= 0).all()

    def test_custom_reference_point(self, paris_stations):
        result = add_distance_to_center(paris_stations, center_lat=0.0, center_lon=0.0)
        # All Paris stations should be > 4000 km from the equator/prime meridian
        assert (result["dist_to_center_km"] > 4000).all()

    def test_custom_output_col(self, paris_stations):
        result = add_distance_to_center(paris_stations, output_col="d_centre")
        assert "d_centre" in result.columns

    def test_original_df_not_modified(self, paris_stations):
        original_cols = list(paris_stations.columns)
        add_distance_to_center(paris_stations)
        assert list(paris_stations.columns) == original_cols

    def test_two_points_north_south(self, two_points):
        # ~1 degree latitude difference ≈ ~1 km
        result = add_distance_to_center(
            two_points,
            center_lat=two_points["lat"].iloc[0],
            center_lon=two_points["lon"].iloc[0],
        )
        assert result["dist_to_center_km"].iloc[0] == pytest.approx(0.0, abs=1e-6)
        assert result["dist_to_center_km"].iloc[1] == pytest.approx(1.0, abs=0.1)


# ---------------------------------------------------------------------------
# add_knn_features
# ---------------------------------------------------------------------------


class TestAddKnnFeatures:
    def test_columns_added_default_ks(self, paris_stations):
        result = add_knn_features(paris_stations)
        for k in (5, 10):
            assert f"knn_{k}_ids" in result.columns
            assert f"knn_{k}_mean_dist" in result.columns

    def test_neighbor_list_length(self, paris_stations):
        result = add_knn_features(paris_stations, ks=(3,))
        assert all(len(ids) == 3 for ids in result["knn_3_ids"])

    def test_self_not_in_neighbors(self, paris_stations):
        result = add_knn_features(paris_stations, ks=(5,))
        for i, row in result.iterrows():
            assert row["station_id"] not in row["knn_5_ids"]

    def test_mean_dist_positive(self, paris_stations):
        result = add_knn_features(paris_stations, ks=(5,))
        assert (result["knn_5_mean_dist"] > 0).all()

    def test_larger_k_larger_mean_dist(self, paris_stations):
        result = add_knn_features(paris_stations, ks=(3, 8))
        # Averaging over more (farther) neighbours should increase mean distance
        assert (result["knn_8_mean_dist"] >= result["knn_3_mean_dist"]).all()

    def test_k_too_large_raises(self, paris_stations):
        with pytest.raises(ValueError, match="k="):
            add_knn_features(paris_stations, ks=(len(paris_stations),))

    def test_original_df_not_modified(self, paris_stations):
        original_cols = list(paris_stations.columns)
        add_knn_features(paris_stations)
        assert list(paris_stations.columns) == original_cols

    def test_neighbor_ids_are_valid_station_ids(self, paris_stations):
        valid_ids = set(paris_stations["station_id"])
        result = add_knn_features(paris_stations, ks=(5,))
        for ids in result["knn_5_ids"]:
            assert set(ids).issubset(valid_ids)


# ---------------------------------------------------------------------------
# add_cluster_labels
# ---------------------------------------------------------------------------


class TestAddClusterLabels:
    def test_column_added(self, paris_stations):
        result = add_cluster_labels(paris_stations, n_clusters=4)
        assert "cluster_id" in result.columns

    def test_label_range(self, paris_stations):
        n_clusters = 4
        result = add_cluster_labels(paris_stations, n_clusters=n_clusters)
        assert result["cluster_id"].between(0, n_clusters - 1).all()

    def test_all_clusters_assigned(self, paris_stations):
        n_clusters = 4
        result = add_cluster_labels(paris_stations, n_clusters=n_clusters)
        assert len(result) == len(paris_stations)
        assert result["cluster_id"].notna().all()

    def test_reproducible_with_same_seed(self, paris_stations):
        r1 = add_cluster_labels(paris_stations, n_clusters=4, random_state=0)
        r2 = add_cluster_labels(paris_stations, n_clusters=4, random_state=0)
        pd.testing.assert_series_equal(r1["cluster_id"], r2["cluster_id"])

    def test_custom_output_col(self, paris_stations):
        result = add_cluster_labels(paris_stations, n_clusters=3, output_col="zone")
        assert "zone" in result.columns

    def test_too_many_clusters_raises(self, paris_stations):
        with pytest.raises(ValueError, match="n_clusters"):
            add_cluster_labels(paris_stations, n_clusters=len(paris_stations) + 1)

    def test_original_df_not_modified(self, paris_stations):
        original_cols = list(paris_stations.columns)
        add_cluster_labels(paris_stations)
        assert list(paris_stations.columns) == original_cols

    def test_geographically_close_stations_same_cluster(self, paris_stations):
        """Stations very close together should end up in the same cluster."""
        close = pd.DataFrame(
            {
                "station_id": [1, 2, 3, 4, 5, 6],
                "lat": [48.8566, 48.8567, 48.8568, 48.9000, 48.9001, 48.9002],
                "lon": [2.3522, 2.3523, 2.3524, 2.4000, 2.4001, 2.4002],
            }
        )
        result = add_cluster_labels(close, n_clusters=2, random_state=0)
        # The two tight groups should each form a distinct cluster
        group_a = set(result.loc[result["station_id"].isin([1, 2, 3]), "cluster_id"])
        group_b = set(result.loc[result["station_id"].isin([4, 5, 6]), "cluster_id"])
        assert len(group_a) == 1
        assert len(group_b) == 1
        assert group_a != group_b


# ---------------------------------------------------------------------------
# add_neighborhood_averages
# ---------------------------------------------------------------------------


class TestAddNeighborhoodAverages:
    def test_columns_added(self, paris_stations):
        result = add_neighborhood_averages(paris_stations, feature_cols=["bikes_available"])
        assert "bikes_available_nb_mean" in result.columns

    def test_multiple_feature_cols(self, paris_stations):
        result = add_neighborhood_averages(
            paris_stations, feature_cols=["bikes_available", "capacity"]
        )
        assert "bikes_available_nb_mean" in result.columns
        assert "capacity_nb_mean" in result.columns

    def test_averages_are_finite(self, paris_stations):
        result = add_neighborhood_averages(paris_stations, feature_cols=["bikes_available"])
        assert result["bikes_available_nb_mean"].notna().all()
        assert np.isfinite(result["bikes_available_nb_mean"]).all()

    def test_average_within_data_range(self, paris_stations):
        result = add_neighborhood_averages(paris_stations, feature_cols=["bikes_available"])
        lo = paris_stations["bikes_available"].min()
        hi = paris_stations["bikes_available"].max()
        assert result["bikes_available_nb_mean"].between(lo, hi).all()

    def test_k_too_large_raises(self, paris_stations):
        with pytest.raises(ValueError, match="k="):
            add_neighborhood_averages(
                paris_stations,
                feature_cols=["bikes_available"],
                k=len(paris_stations),
            )

    def test_missing_feature_col_raises(self, paris_stations):
        with pytest.raises(ValueError, match="nonexistent"):
            add_neighborhood_averages(paris_stations, feature_cols=["nonexistent"])

    def test_original_df_not_modified(self, paris_stations):
        original_cols = list(paris_stations.columns)
        add_neighborhood_averages(paris_stations, feature_cols=["bikes_available"])
        assert list(paris_stations.columns) == original_cols

    def test_uniform_values_give_same_average(self, paris_stations):
        """When all stations have the same value, the neighbour mean equals that value."""
        df = paris_stations.copy()
        df["uniform"] = 42.0
        result = add_neighborhood_averages(df, feature_cols=["uniform"])
        np.testing.assert_allclose(result["uniform_nb_mean"].to_numpy(), 42.0)


# ---------------------------------------------------------------------------
# add_all_spatial_features
# ---------------------------------------------------------------------------


class TestAddAllSpatialFeatures:
    EXPECTED_COLS = [
        "dist_to_center_km",
        "knn_5_ids",
        "knn_5_mean_dist",
        "knn_10_ids",
        "knn_10_mean_dist",
        "cluster_id",
    ]

    def test_all_base_columns_present(self, paris_stations):
        result = add_all_spatial_features(paris_stations, n_clusters=4)
        for col in self.EXPECTED_COLS:
            assert col in result.columns, f"Missing column: {col}"

    def test_neighbourhood_cols_when_feature_cols_given(self, paris_stations):
        result = add_all_spatial_features(
            paris_stations, feature_cols=["bikes_available"], n_clusters=4
        )
        assert "bikes_available_nb_mean" in result.columns

    def test_no_neighbourhood_cols_without_feature_cols(self, paris_stations):
        result = add_all_spatial_features(paris_stations, n_clusters=4)
        assert "bikes_available_nb_mean" not in result.columns

    def test_original_columns_preserved(self, paris_stations):
        result = add_all_spatial_features(paris_stations, n_clusters=4)
        for col in paris_stations.columns:
            assert col in result.columns

    def test_row_count_unchanged(self, paris_stations):
        result = add_all_spatial_features(paris_stations, n_clusters=4)
        assert len(result) == len(paris_stations)

    def test_original_df_not_modified(self, paris_stations):
        original_cols = list(paris_stations.columns)
        add_all_spatial_features(paris_stations, n_clusters=4)
        assert list(paris_stations.columns) == original_cols
