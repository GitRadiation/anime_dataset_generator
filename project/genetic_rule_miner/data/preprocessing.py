import numpy as np
import pandas as pd

from genetic_rule_miner.utils.logging import LogManager, log_execution

logger = LogManager.get_logger(__name__)


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


def clean_and_bin_column(
    df: pd.DataFrame,
    column: str,
    bins: list,
    labels: list,
    fill_value: float = 20,
) -> pd.DataFrame:
    """Clean, fill, and bin a numeric column based on specified bins and labels.

    Args:
        df: Input DataFrame.
        column: The column to clean and bin.
        bins: Bins for categorizing the values.
        labels: Labels for the bin categories.
        fill_value: Default fill value for missing data (default 20).

    Returns:
        DataFrame with the cleaned and binned column.
    """
    if column in df.columns:
        # Cleaning and conversion
        df[column] = (
            df[column]
            .astype(str)
            .replace(["", "nan", "None", "Unknown", "N/A"], np.nan)
        )
        df[column] = df[column].astype(float)

        # Fill missing values
        df[column] = df[column].fillna(
            df[column].median() if df[column].notna().any() else fill_value
        )

        # Binning the column
        df[f"{column}_class"] = pd.cut(
            df[column], bins=bins, labels=labels, right=False
        )
    return df


@log_execution
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess data with robust handling of edge cases and domain-specific enhancements."""
    df = df.copy()

    try:
        # Clean string columns
        df = clean_string_columns(df)

        if "duration" in df.columns:
            # Clean and bin 'duration' column
            df = clean_and_bin_column(
                df,
                "duration",
                [0, 20, 25, max(30, df["duration"].max() + 1)],
                ["short", "standard", "long"],
            )

        # Clean and bin 'episodes' column
        if "episodes" in df.columns:
            df = clean_and_bin_column(
                df,
                "episodes",
                [0, 12, 24, max(26, df["episodes"].max() + 1)],
                ["short", "medium", "long"],
            )

        # Clean 'rating_x' and create 'rating' column
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

        # Process 'birthday' column and create 'age' and 'age_group'
        if "birthday" in df.columns:
            today = pd.Timestamp("now").normalize()
            df["birthday"] = pd.to_datetime(df["birthday"], errors="coerce")
            df["age"] = (today - df["birthday"]).dt.days // 365

            df["age_group"] = pd.cut(
                df["age"],
                bins=[0, 25, 40, 200],
                labels=["young", "adult", "senior"],
                include_lowest=True,
            )

        # Process 'aired' column and extract start year
        if "aired" in df.columns:
            df["aired_start_year"] = (
                df["aired"]
                .astype(str)
                .str.extract(r"(\d{4})")[0]
                .astype(float)
                .fillna(0)
                .astype(int)
            )

        # Convert 'producers', 'genres', 'keywords' to lists if they exist
        for col in ["producers", "genres", "keywords"]:
            if col in df.columns:
                df[col] = (
                    df[col]
                    .fillna(" ")
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

        # Drop rows with unknown rating (if the 'rating' column exists)
        if "rating" in df.columns:
            df = df[df["rating"] != "unknown"]

        # Drop unnecessary columns: 'rating_x' and 'age' if they exist
        df = df.drop(columns=["rating_x", "age"], errors="ignore")

        return df

    except Exception as e:
        logger.error(
            f"Preprocessing failed. Current data state:\n{df.head().to_string()}"
        )
        raise ValueError(f"Data preprocessing failed: {str(e)}") from e
