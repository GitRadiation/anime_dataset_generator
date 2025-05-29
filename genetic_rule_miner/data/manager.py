"""Data management layer with optimized database operations."""

from typing import Tuple

import numpy as np
import pandas as pd
from sqlalchemy.engine.base import Connection

from genetic_rule_miner.config import DBConfig
from genetic_rule_miner.data.database import DatabaseManager
from genetic_rule_miner.data.preprocessing import clean_string_columns
from genetic_rule_miner.utils.exceptions import DataValidationError
from genetic_rule_miner.utils.logging import log_execution


class DataManager:
    """High-performance data loading and transformation manager."""

    _instance = None

    def __new__(cls, db_config: DBConfig) -> "DataManager":
        if cls._instance is None:
            cls._instance = super(DataManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, db_config: DBConfig) -> None:
        if self._initialized:
            return
        self.db_manager = DatabaseManager(db_config)
        self._initialized = True

    def _load_and_clean_data(
        self, table_name: str, conn: Connection
    ) -> pd.DataFrame:
        """Load data from a table and preprocess it."""
        data = pd.read_sql(f"SELECT * FROM {table_name}", conn)
        return clean_string_columns(data)

    @log_execution
    def load_and_preprocess_data(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load core datasets and preprocess each one separately."""
        tables = ["user_details", "anime_dataset", "user_score"]
        try:
            with self.db_manager.connection() as conn:
                # Load and clean data for each table using the helper function
                user_details_cleaned = self._load_and_clean_data(
                    tables[0], conn
                )

                anime_data_cleaned = self._load_and_clean_data(tables[1], conn)
                user_scores_cleaned = self._load_and_clean_data(
                    tables[2], conn
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
        if "rating_x" in result.columns:
            result.rename(columns={"rating_x": "rating"}, inplace=True)
        # Cleanup
        return result.drop(
            columns=["user_id", "mal_id"], errors="ignore"
        ).reset_index(drop=True)
