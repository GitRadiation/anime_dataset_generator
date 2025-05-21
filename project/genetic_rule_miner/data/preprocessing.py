import numpy as np
import pandas as pd

from genetic_rule_miner.utils.logging import LogManager, log_execution

logger = LogManager.get_logger(__name__)


def safe_astype(column, dtype):
    """Convert a column to a specified dtype if it's not empty."""
    if column is not None and not column.isna().all():
        return column.astype(dtype)
    return column


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
            .replace(["\\N", "nan", "null", "None", "<NA>"], pd.NA)
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
            .replace(["", "nan", "None", "Unknown", "N/A", "<NA>"], np.nan)
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
    pd.set_option("future.no_silent_downcasting", True)
    df = df.copy()
    try:
        df = clean_string_columns(df)

        if "duration" in df.columns:
            df["duration"] = pd.to_numeric(df["duration"], errors="coerce")
            max_duration = df["duration"].max(skipna=True)
            if pd.isna(max_duration):
                max_duration = 30
            else:
                max_duration = max(30, int(max_duration) + 1)

            df = clean_and_bin_column(
                df,
                "duration",
                [0, 20, 25, max_duration],
                ["short", "standard", "long"],
            )

        if "episodes" in df.columns:
            df["episodes"] = pd.to_numeric(df["episodes"], errors="coerce")
            df = clean_and_bin_column(
                df,
                "episodes",
                [0, 12, 24, max(26, df["episodes"].max() + 1)],
                ["short", "medium", "long"],
            )

        if "rating_x" in df.columns:
            df["rating_x"] = pd.to_numeric(df["rating_x"], errors="coerce")
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

        if "birthday" in df.columns:
            # Ensure both today and birthday are tz-naive
            today = pd.Timestamp("now").tz_localize(
                None
            )  # This is already tz-naive

            df["birthday"] = pd.to_datetime(df["birthday"], errors="coerce")

            # Ensure birthday is tz-naive as well
            df["birthday"] = df["birthday"].dt.tz_localize(None)

            df["age"] = (today - pd.to_datetime(df["birthday"])).dt.days // 365

            df["age_group"] = pd.cut(
                df["age"],
                bins=[0, 25, 40, 200],
                labels=["young", "adult", "senior"],
                include_lowest=True,
            )

            df.drop(columns=["birthday"], errors="ignore", inplace=True)

        if "aired" in df.columns:
            df["aired"] = (
                df["aired"]
                .astype(str)
                .str.extract(r"(\d{4})")[0]
                .pipe(safe_astype, float)
                .fillna(0)
                .pipe(safe_astype, int)
            )

        for col in ["producers", "genres", "keywords"]:
            if col in df.columns:
                df[col] = (
                    df[col]
                    .fillna(" ")  # ✅ FIXED: no inplace
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

        if "rating" in df.columns:
            df = df[df["rating"] != "unknown"]

        df.drop(
            columns=["rating_x", "age", "duration", "birthday", "age"],
            errors="ignore",
            inplace=True,
        )

        return df

    except Exception as e:
        logger.error(
            f"Preprocessing failed. Current data state:\n{df.head().to_string()}"
        )
        raise ValueError(f"Data preprocessing failed: {str(e)}") from e
