"""Abstract base class for all Velib forecasting models."""

from abc import ABC, abstractmethod

import pandas as pd


class VelibBaseModel(ABC):
    """Common interface every Velib model must implement.

    Subclasses only need to define :meth:`fit` and :meth:`predict`.
    :meth:`evaluate` is provided for free once those two are implemented.
    """

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> "VelibBaseModel":
        """Learn from historical station-status data.

        Args:
            df: Training DataFrame with at minimum a station identifier column,
                a timestamp column, and the target value column.

        Returns:
            *self* — enables method chaining.
        """

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Generate one-step-ahead forecasts for every row in *df*.

        Args:
            df: DataFrame in the same schema as the training data.

        Returns:
            A :class:`pandas.Series` of predicted values aligned with
            ``df.index``, named ``"prediction"``.
        """

    def evaluate(
        self,
        df: pd.DataFrame,
        target_col: str = "num_bikes_available",
    ) -> dict[str, float]:
        """Compute MAE, RMSE and MAPE on *df*.

        Delegates to :func:`~src.models.baseline.persistence.compute_metrics`
        after calling :meth:`predict`.

        Args:
            df: DataFrame that contains the ground-truth *target_col*.
            target_col: Name of the column holding true observations.

        Returns:
            ``{"mae": …, "rmse": …, "mape": …}``
        """
        from src.models.baseline.persistence import compute_metrics

        y_true = df[target_col]
        y_pred = self.predict(df)
        return compute_metrics(y_true, y_pred)
