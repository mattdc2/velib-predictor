"""Spatial feature engineering for Velib station data.

Provides functions to enrich a station DataFrame with location-based features:
  - Distance to Paris city centre (Haversine)
  - K-nearest neighbours (k=5, 10) — IDs and mean distance
  - Station clustering via k-means on (lat, lon)
  - Neighbourhood averages for arbitrary status columns

All computations are pure NumPy/pandas — no optional GIS dependencies required.
"""

from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

# Reference point: geometric centre of Paris
PARIS_LAT: float = 48.8566
PARIS_LON: float = 2.3522

# Earth radius used for Haversine calculations
_EARTH_RADIUS_KM: float = 6371.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def haversine_km(
    lat1: float | np.ndarray,
    lon1: float | np.ndarray,
    lat2: float | np.ndarray,
    lon2: float | np.ndarray,
) -> float | np.ndarray:
    """Compute great-circle distance(s) in kilometres using the Haversine formula.

    Accepts scalars or broadcastable NumPy arrays.

    Args:
        lat1: Latitude of the first point(s) in decimal degrees.
        lon1: Longitude of the first point(s) in decimal degrees.
        lat2: Latitude of the second point(s) in decimal degrees.
        lon2: Longitude of the second point(s) in decimal degrees.

    Returns:
        Distance in kilometres (same shape as the broadcasted inputs).
    """
    lat1, lon1, lat2, lon2 = (np.radians(x) for x in (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * _EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def compute_distance_matrix(
    df: pd.DataFrame,
    lat_col: str = "lat",
    lon_col: str = "lon",
) -> np.ndarray:
    """Compute the pairwise Haversine distance matrix for all stations.

    Args:
        df: DataFrame with one row per station.
        lat_col: Name of the latitude column.
        lon_col: Name of the longitude column.

    Returns:
        A symmetric (n, n) array of distances in kilometres, with zeros on the
        diagonal.
    """
    lats = df[lat_col].to_numpy()
    lons = df[lon_col].to_numpy()
    # Broadcast to (n, n)
    dist = haversine_km(lats[:, None], lons[:, None], lats[None, :], lons[None, :])
    return np.asarray(dist)


def _kmeans_numpy(
    X: np.ndarray,
    n_clusters: int,
    max_iter: int = 300,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Minimal Lloyd's k-means on a (n, d) feature matrix.

    Uses random initialisation seeded by *random_state*.

    Args:
        X: Feature matrix of shape (n_samples, n_features).
        n_clusters: Number of clusters.
        max_iter: Maximum number of Lloyd iterations.
        random_state: Seed for the random number generator.

    Returns:
        Tuple of (labels, centres) where *labels* is shape (n,) and
        *centres* is shape (n_clusters, n_features).
    """
    rng = np.random.default_rng(random_state)
    init_idx = rng.choice(len(X), size=n_clusters, replace=False)
    centres = X[init_idx].copy().astype(float)
    labels = np.zeros(len(X), dtype=int)

    for _ in range(max_iter):
        # Assignment step: squared Euclidean distance to each centre
        diffs = X[:, None, :] - centres[None, :, :]  # (n, k, d)
        sq_dists = (diffs**2).sum(axis=2)  # (n, k)
        new_labels = sq_dists.argmin(axis=1)

        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

        # Update step
        for j in range(n_clusters):
            mask = labels == j
            if mask.any():
                centres[j] = X[mask].mean(axis=0)

    return labels, centres


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def add_distance_to_center(
    df: pd.DataFrame,
    lat_col: str = "lat",
    lon_col: str = "lon",
    center_lat: float = PARIS_LAT,
    center_lon: float = PARIS_LON,
    output_col: str = "dist_to_center_km",
) -> pd.DataFrame:
    """Add the Haversine distance from each station to a reference point.

    Args:
        df: Input DataFrame with one row per station. Not modified in place.
        lat_col: Name of the latitude column.
        lon_col: Name of the longitude column.
        center_lat: Latitude of the reference point (default: Paris centre).
        center_lon: Longitude of the reference point (default: Paris centre).
        output_col: Name of the output column.

    Returns:
        A copy of *df* with *output_col* appended (distances in km).
    """
    result = df.copy()
    result[output_col] = haversine_km(
        df[lat_col].to_numpy(),
        df[lon_col].to_numpy(),
        center_lat,
        center_lon,
    )
    logger.debug("Added distance-to-centre feature: %s", output_col)
    return result


def add_knn_features(
    df: pd.DataFrame,
    ks: tuple[int, ...] = (5, 10),
    lat_col: str = "lat",
    lon_col: str = "lon",
    id_col: str = "station_id",
) -> pd.DataFrame:
    """Add k-nearest-neighbour IDs and mean distances for each station.

    The diagonal (self-distance = 0) is excluded from neighbour lists.

    For each value *k* in *ks* the following columns are added:

    - ``knn_{k}_ids``       — list of the *k* nearest station IDs
    - ``knn_{k}_mean_dist`` — mean Haversine distance to those neighbours (km)

    Args:
        df: Input DataFrame with one row per station. Not modified in place.
        ks: Tuple of neighbourhood sizes to compute (default: ``(5, 10)``).
        lat_col: Name of the latitude column.
        lon_col: Name of the longitude column.
        id_col: Name of the station identifier column.

    Returns:
        A copy of *df* with 2 × len(ks) new columns appended.

    Raises:
        ValueError: If any *k* is ≥ the number of stations in *df*.
    """
    n = len(df)
    for k in ks:
        if k >= n:
            raise ValueError(f"k={k} must be less than the number of stations ({n}).")

    dist_matrix = compute_distance_matrix(df, lat_col, lon_col)  # (n, n)
    station_ids = df[id_col].to_numpy()
    result = df.copy()

    for k in ks:
        neighbor_ids: list[list] = []
        mean_dists: list[float] = []

        for i in range(n):
            row = dist_matrix[i].copy()
            row[i] = np.inf  # exclude self
            nn_idx = np.argpartition(row, k)[:k]  # unsorted k nearest
            nn_idx = nn_idx[np.argsort(row[nn_idx])]  # sort by distance
            neighbor_ids.append(station_ids[nn_idx].tolist())
            mean_dists.append(float(row[nn_idx].mean()))

        result[f"knn_{k}_ids"] = neighbor_ids
        result[f"knn_{k}_mean_dist"] = mean_dists
        logger.debug("Added KNN features for k=%d", k)

    return result


def add_cluster_labels(
    df: pd.DataFrame,
    n_clusters: int = 8,
    lat_col: str = "lat",
    lon_col: str = "lon",
    output_col: str = "cluster_id",
    random_state: int = 42,
) -> pd.DataFrame:
    """Cluster stations by geographic location using k-means.

    Runs Lloyd's algorithm on normalised (lat, lon) coordinates.  Cluster IDs
    are integers in ``[0, n_clusters)``.

    Args:
        df: Input DataFrame with one row per station. Not modified in place.
        n_clusters: Number of k-means clusters (default: 8).
        lat_col: Name of the latitude column.
        lon_col: Name of the longitude column.
        output_col: Name of the cluster label column.
        random_state: Seed for reproducible initialisation.

    Returns:
        A copy of *df* with *output_col* appended.

    Raises:
        ValueError: If *n_clusters* > number of stations.
    """
    n = len(df)
    if n_clusters > n:
        raise ValueError(f"n_clusters={n_clusters} cannot exceed the number of stations ({n}).")

    X = df[[lat_col, lon_col]].to_numpy(dtype=float)
    # Normalise so lat and lon contribute equally in Euclidean space
    X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    labels, _ = _kmeans_numpy(X_norm, n_clusters, random_state=random_state)

    result = df.copy()
    result[output_col] = labels
    logger.debug("Added cluster labels: %d clusters, column '%s'", n_clusters, output_col)
    return result


def add_neighborhood_averages(
    df: pd.DataFrame,
    feature_cols: list[str],
    k: int = 5,
    lat_col: str = "lat",
    lon_col: str = "lon",
    id_col: str = "station_id",
) -> pd.DataFrame:
    """For each station, compute the mean of *feature_cols* across its *k* nearest neighbours.

    Expects one row per station (a snapshot, not a time series).  For time-series
    use, apply this function per timestamp after a ``groupby``.

    Added columns: ``{col}_nb_mean`` for each column in *feature_cols*.

    Args:
        df: Input DataFrame with one row per station. Not modified in place.
        feature_cols: Columns whose neighbourhood average should be computed.
        k: Number of nearest neighbours to average over.
        lat_col: Name of the latitude column.
        lon_col: Name of the longitude column.
        id_col: Name of the station identifier column.

    Returns:
        A copy of *df* with one new column per entry in *feature_cols*.

    Raises:
        ValueError: If *k* ≥ number of stations, or a column in *feature_cols*
            is not present in *df*.
    """
    n = len(df)
    if k >= n:
        raise ValueError(f"k={k} must be less than the number of stations ({n}).")
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in df: {missing}")

    dist_matrix = compute_distance_matrix(df, lat_col, lon_col)  # (n, n)
    feature_matrix = df[feature_cols].to_numpy(dtype=float)  # (n, F)
    result = df.copy()

    nb_means = np.empty((n, len(feature_cols)), dtype=float)

    for i in range(n):
        row = dist_matrix[i].copy()
        row[i] = np.inf
        nn_idx = np.argpartition(row, k)[:k]
        nb_means[i] = feature_matrix[nn_idx].mean(axis=0)

    for j, col in enumerate(feature_cols):
        result[f"{col}_nb_mean"] = nb_means[:, j]

    logger.debug("Added neighbourhood averages (k=%d) for columns: %s", k, feature_cols)
    return result


def add_all_spatial_features(
    df: pd.DataFrame,
    feature_cols: Optional[list[str]] = None,
    ks: tuple[int, ...] = (5, 10),
    n_clusters: int = 8,
    lat_col: str = "lat",
    lon_col: str = "lon",
    id_col: str = "station_id",
    center_lat: float = PARIS_LAT,
    center_lon: float = PARIS_LON,
    random_state: int = 42,
) -> pd.DataFrame:
    """Apply all spatial feature transformations in a single call.

    Convenience wrapper that sequentially calls:
        1. :func:`add_distance_to_center`
        2. :func:`add_knn_features`
        3. :func:`add_cluster_labels`
        4. :func:`add_neighborhood_averages` (only when *feature_cols* is given)

    Args:
        df: Input DataFrame with one row per station. Not modified in place.
        feature_cols: Status columns to average across neighbours. Skipped when
            ``None``.
        ks: Neighbourhood sizes for KNN features.
        n_clusters: Number of k-means clusters.
        lat_col: Name of the latitude column.
        lon_col: Name of the longitude column.
        id_col: Name of the station identifier column.
        center_lat: Latitude of the reference point.
        center_lon: Longitude of the reference point.
        random_state: Seed for k-means initialisation.

    Returns:
        A copy of *df* with all spatial features appended.
    """
    result = add_distance_to_center(df, lat_col, lon_col, center_lat, center_lon)
    result = add_knn_features(result, ks=ks, lat_col=lat_col, lon_col=lon_col, id_col=id_col)
    result = add_cluster_labels(
        result,
        n_clusters=n_clusters,
        lat_col=lat_col,
        lon_col=lon_col,
        random_state=random_state,
    )
    if feature_cols:
        result = add_neighborhood_averages(
            result,
            feature_cols=feature_cols,
            k=min(ks),
            lat_col=lat_col,
            lon_col=lon_col,
            id_col=id_col,
        )
    logger.info("All spatial features added successfully")
    return result
