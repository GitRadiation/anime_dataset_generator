"""Data preprocessing functions."""

import numpy as np
import pandas as pd

from genetic_rule_miner.utils.decorators import validate_dataframe
from genetic_rule_miner.utils.logging import LogManager, log_execution

logger = LogManager.get_logger(__name__)


@log_execution
def clean_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Clean string columns by stripping whitespace and replacing placeholders with NA.

    Args:
        df: Input DataFrame with columns to clean

    Returns:
        DataFrame with cleaned string columns
    """
    # Make a copy to avoid SettingWithCopyWarning
    df = df.copy()

    # Identify string columns (object dtype or explicit string dtype)
    str_cols = df.select_dtypes(include=["object", "string"]).columns

    for col in str_cols:
        # Ensure we're working with strings (convert if necessary)
        df[col] = df[col].astype("string")

        # Apply cleaning operations
        df[col] = (
            df[col]
            .str.strip()  # Remove whitespace
            .replace(
                ["\\N", "nan", "null", "None"], pd.NA
            )  # Replace placeholders
        )

    return df


@log_execution
@validate_dataframe("duration", "episodes")
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess data with robust handling of edge cases."""
    df = df.copy()

    try:
        # 1. Enhanced duration parsing
        if "duration" in df.columns:
            df["duration"] = (
                df["duration"]
                .astype(str)
                .str.extract(r"(\d+)")[0]  # Extract numeric part
                .replace(["", "nan", "None", "Unknown", "N/A"], pd.NA)
                .astype(float)
            )

            # Fallback to alternative duration column if available
            if df["duration"].isna().all() and "episode_length" in df.columns:
                df["duration"] = df["episode_length"]

        # 2. Enhanced episodes handling
        if "episodes" in df.columns:
            df["episodes"] = (
                df["episodes"]
                .astype(str)
                .replace(["", "nan", "None", "Unknown", "N/A"], pd.NA)
                .astype(float)
            )

        # 3. Handle completely missing data with defaults
        if "duration" in df.columns and df["duration"].isna().all():
            logger.warning("All duration values missing - using default 23.0")
            df["duration"] = 23.0

        if "episodes" in df.columns and df["episodes"].isna().all():
            logger.warning("All episodes values missing - using default 12.0")
            df["episodes"] = 12.0

        # 4. Fill NA values with sensible defaults
        duration_fill = (
            df["duration"].median() if df["duration"].notna().any() else 20
        )
        episodes_fill = (
            df["episodes"].median() if df["episodes"].notna().any() else 12
        )

        df["duration"] = df["duration"].fillna(duration_fill)
        df["episodes"] = df["episodes"].fillna(episodes_fill)

        # 5. Validate finite values
        if not np.isfinite(df[["duration", "episodes"]].values).all():
            invalid_duration = df.loc[~np.isfinite(df["duration"]), "duration"]
            invalid_episodes = df.loc[~np.isfinite(df["episodes"]), "episodes"]
            logger.warning(
                f"Found non-finite values - Duration: {invalid_duration}, Episodes: {invalid_episodes}"
            )
            raise ValueError("Non-finite values detected after cleaning")

        # 6. Create bins with safe bounds
        duration_bins = [0, 20, 25, max(30, df["duration"].max() + 1)]
        episodes_bins = [0, 12, 24, max(26, df["episodes"].max() + 1)]

        df["duration_class"] = pd.cut(
            df["duration"],
            bins=duration_bins,
            labels=["short", "standard", "long"],
            right=False,
        )

        df["episodes_class"] = pd.cut(
            df["episodes"],
            bins=episodes_bins,
            labels=["short", "medium", "long"],
            right=False,
        )

        # 7. Handle rating if present
        if "rating_x" in df.columns:
            df["rating"] = (
                pd.cut(
                    df["rating_x"],
                    bins=[0, 6.9, 7.9, 10],
                    labels=["low", "medium", "high"],
                    right=False,
                )
                .astype(object)
                .fillna("unknown")
            )

        return df.drop(columns=["rating_x"], errors="ignore")

    except Exception as e:
        logger.error(
            f"Preprocessing failed. Current data state:\n{df.head().to_string()}"
        )
        raise ValueError(f"Data preprocessing failed: {str(e)}") from e
