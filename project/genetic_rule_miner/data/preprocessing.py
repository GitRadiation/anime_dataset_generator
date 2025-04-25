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
    """Preprocess data with robust handling of edge cases and domain-specific enhancements."""
    df = df.copy()

    try:
        # Duration cleaning
        if "duration" in df.columns:
            df["duration"] = (
                df["duration"]
                .astype(str)
                .str.extract(r"(\d+)")[0]
                .replace(["", "nan", "None", "Unknown", "N/A"], np.nan)
                .astype(float)
            )

            if df["duration"].isna().all() and "episode_length" in df.columns:
                df["duration"] = df["episode_length"]

        # Episodes cleaning
        if "episodes" in df.columns:
            df["episodes"] = (
                df["episodes"]
                .astype(str)
                .replace(["", "nan", "None", "Unknown", "N/A"], np.nan)
                .astype(float)
            )

        # Fallback defaults
        if df["duration"].isna().all():
            logger.warning("All duration values missing - using default 23.0")
            df["duration"] = 23.0

        if df["episodes"].isna().all():
            logger.warning("All episodes values missing - using default 12.0")
            df["episodes"] = 12.0

        # Fill NA with medians
        df["duration"] = df["duration"].fillna(
            df["duration"].median() if df["duration"].notna().any() else 20
        )
        df["episodes"] = df["episodes"].fillna(
            df["episodes"].median() if df["episodes"].notna().any() else 12
        )

        # Validity check
        if not np.isfinite(df[["duration", "episodes"]].values).all():
            raise ValueError("Non-finite values detected after cleaning")

        # Binning duration and episodes
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

        # Binning rating_x
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

        # Parse birthday into age groups
        if "birthday" in df.columns:
            today = pd.Timestamp("now").normalize()
            df["birthday"] = pd.to_datetime(df["birthday"], errors="coerce")
            df["age"] = (today - df["birthday"]).dt.days // 365

            df["age_group"] = pd.cut(
                df["age"],
                bins=[0, 25, 40, 100],
                labels=["young", "adult", "senior"],
                include_lowest=True,
            )

        # Extract year from aired
        if "aired" in df.columns:
            df["aired_start_year"] = (
                df["aired"]
                .astype(str)
                .str.extract(r"(\d{4})")[0]
                .astype(float)
                .fillna(0)
                .astype(int)
            )

        # Convert producers, genres, keywords to lists
        for col in ["producers", "genres", "keywords"]:
            if col in df.columns:
                df[col] = (
                    df[col]
                    .fillna("")
                    .astype(str)
                    .str.split(r",\s*")
                    .apply(
                        lambda x: (
                            [i.strip() for i in x if i.strip()]
                            if isinstance(x, list)
                            else []
                        )
                    )
                )

        # Drop rows with unknown rating
        if "rating" in df.columns:
            df = df[df["rating"] != "unknown"]

        return df.drop(columns=["rating_x", "age"], errors="ignore")

    except Exception as e:
        logger.error(
            f"Preprocessing failed. Current data state:\n{df.head().to_string()}"
        )
        raise ValueError(f"Data preprocessing failed: {str(e)}") from e
