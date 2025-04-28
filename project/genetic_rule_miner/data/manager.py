"""Data management layer with optimized database operations."""

from typing import Tuple

import numpy as np
import pandas as pd

from genetic_rule_miner.config import DBConfig
from genetic_rule_miner.data.database import DatabaseManager
from genetic_rule_miner.data.preprocessing import (
    clean_string_columns,
    preprocess_data,
)
from genetic_rule_miner.utils.exceptions import DataValidationError
from genetic_rule_miner.utils.logging import log_execution


class DataManager:
    """High-performance data loading and transformation manager."""

    def __init__(self, db_config: DBConfig) -> None:
        """Initialize with database configuration."""
        self.db_manager = DatabaseManager(db_config)

    @log_execution
    def load_and_preprocess_data(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load core datasets and preprocess each one separately."""
        tables = ["user_details", "anime_dataset", "user_score"]
        try:
            with self.db_manager.connection() as conn:
                # Load data
                user_details = pd.read_sql(f"SELECT * FROM {tables[0]}", conn)
                anime_data = pd.read_sql(f"SELECT * FROM {tables[1]}", conn)
                user_scores = pd.read_sql(
                    f"SELECT * FROM {tables[2]}",
                    conn,
                    parse_dates=["timestamp"],
                )

                # Clean and preprocess data
                user_details_cleaned = preprocess_data(
                    clean_string_columns(user_details)
                )
                anime_data_cleaned = preprocess_data(
                    clean_string_columns(anime_data)
                )
                user_scores_cleaned = preprocess_data(
                    clean_string_columns(user_scores)
                )

                return (
                    user_details_cleaned,
                    anime_data_cleaned,
                    user_scores_cleaned,
                )
        except Exception as e:
            raise DataValidationError(f"Data loading failed: {str(e)}")

    @staticmethod
    @log_execution
    def merge_data(
        user_scores: pd.DataFrame,
        user_details: pd.DataFrame,
        anime_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Merge datasets with validation and optimized operations."""
        if any(df.empty for df in [user_scores, user_details, anime_data]):
            raise DataValidationError("Cannot merge empty DataFrames")

        # First merge with type conversion optimization
        user_scores["user_id"] = user_scores["user_id"].astype(np.int64)
        user_details["mal_id"] = user_details["mal_id"].astype(np.int64)

        merged = pd.merge(
            user_scores,
            user_details,
            left_on="user_id",
            right_on="mal_id",
            how="inner",
            validate="many_to_one",
            suffixes=("_score", ""),
        )

        # Second merge with index optimization
        anime_data.set_index("anime_id", inplace=True)
        result = pd.merge(
            merged,
            anime_data,
            left_on="anime_id",
            right_index=True,
            how="inner",
            validate="many_to_one",
        )

        # Cleanup
        return result.drop(
            columns=["user_id", "mal_id", "anime_id"], errors="ignore"
        ).reset_index(drop=True)
