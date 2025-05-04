import logging
import re
import time
from io import BytesIO, StringIO

import nltk
import pandas as pd
from genetic_rule_miner.bbdd_maker.anime_service import AnimeService
from genetic_rule_miner.bbdd_maker.details_service import DetailsService
from genetic_rule_miner.bbdd_maker.score_service import ScoreService
from genetic_rule_miner.bbdd_maker.user_service import UserService
from genetic_rule_miner.config import APIConfig, DBConfig
from genetic_rule_miner.data.database import DatabaseManager
from genetic_rule_miner.utils.logging import LogManager

# Configure logging
LogManager.configure()
logger = logging.getLogger(__name__)


def download_nltk_resources():
    try:
        # Check if 'stopwords' and 'punkt' are downloaded
        nltk.data.find("corpora/stopwords.zip")
    except LookupError:
        print("Downloading stopwords...")
        nltk.download("stopwords")

    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        print("Downloading punkt...")
        nltk.download("punkt")


# Call the function only on the first execution
download_nltk_resources()


def clean_premiered(value):
    if pd.isna(value) or value.strip().lower() == "none none":
        return None
    match = re.match(r"(spring|summer|fall|winter)", value.strip().lower())
    return match.group(1) if match else None


def clean_string_columns(df):
    """Removes whitespace from object-type columns"""
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].str.strip()
    return df


def convert_duration_to_minutes(duration):
    """Converts duration in format '1 hr 31 min' or '24 min' to minutes"""
    duration_pattern = re.match(
        r"(?:(\d+)\s*hr)?(?:\s*(\d+)\s*min)?", duration.strip()
    )

    if not duration_pattern:
        return None  # If it doesn't match the pattern, return None

    hours = duration_pattern.group(1)
    minutes = duration_pattern.group(2)

    total_minutes = 0
    if hours:
        total_minutes += int(hours) * 60
    if minutes:
        total_minutes += int(minutes)

    return total_minutes


def preprocess_to_memory(
    df, columns_to_keep, integer_columns, float_columns=None
):
    """Preprocesses and converts data types according to the database schema"""
    df = df.copy()
    df.columns = df.columns.str.strip()
    df = clean_string_columns(df)

    missing_columns = [col for col in columns_to_keep if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"The following columns were not found: {missing_columns}"
        )

    if "duration" in df.columns:
        df["duration"] = (
            df["duration"].astype(str).apply(convert_duration_to_minutes)
        )

    df = df[columns_to_keep]

    for col in integer_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    if float_columns:
        for col in float_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(how="all", inplace=True)
    df = df.where(pd.notnull(df), None)
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False, header=True, na_rep="\\N")
    csv_buffer.seek(0)
    return csv_buffer


def preprocess_user_score(
    df, columns_to_keep, integer_columns, valid_anime_ids
):
    """Preprocesses the user_score DataFrame and removes rows with invalid Anime IDs"""
    df = df.copy()
    df.columns = df.columns.str.strip()
    df = clean_string_columns(df)

    missing_columns = [col for col in columns_to_keep if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"The following columns were not found: {missing_columns}"
        )

    df = df[columns_to_keep]

    df["anime_id"] = pd.to_numeric(df["anime_id"], errors="coerce")
    valid_set = set(valid_anime_ids)
    df = df[df["anime_id"].isin(valid_set)]

    for col in integer_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    df.dropna(how="all", inplace=True)
    df = df.where(pd.notnull(df), None)
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False, header=True, na_rep="\\N")
    csv_buffer.seek(0)
    return csv_buffer


def main():
    # ==========================
    # Phase 1: Initialization
    # ==========================
    logger.info("🔧 Phase 1: Initialization")
    db_config = DBConfig()
    api_config = APIConfig()

    # Retry parameters
    max_retries = 3
    retry_delay = 5  # in seconds

    # ==========================
    # Phase 2: Data Retrieval
    # ==========================
    logger.info("📡 Phase 2: Data Retrieval")
    for attempt in range(1, max_retries + 1):
        logger.info(f"🤔 Attempt {attempt} to retrieve base data...")

        # 1. Retrieve anime data
        logger.info("📡 Retrieving anime data...")
        anime_buffer = AnimeService(api_config).get_anime_data(1, 100)

        # 2. Generate user list
        logger.info("📡 Generating user list...")
        user_service = UserService(api_config)
        userlist_buffer = user_service.generate_userlist(
            start_id=1, end_id=100
        )

        # 3. Prepare data for ScoreService
        logger.info("📡 Preparing user data...")
        userlist_df = pd.read_csv(userlist_buffer)
        userlist_df.rename(columns={"user_id": "mal_id"}, inplace=True)
        modified_userlist_buffer = BytesIO()
        userlist_df.to_csv(modified_userlist_buffer, index=False)
        modified_userlist_buffer.seek(0)

        # 4. Retrieve user details
        logger.info("📡 Retrieving user details...")
        usernames = userlist_df["username"].dropna().tolist()
        details_service = DetailsService(api_config)
        details_buffer = details_service.get_user_details(usernames)

        # 5. Retrieve scores
        logger.info("📡 Retrieving user scores...")
        score_service = ScoreService(api_config)
        scores_buffer = score_service.get_scores(modified_userlist_buffer)

        if all([anime_buffer, details_buffer, scores_buffer]):
            logger.info("✅ Successfully retrieved base data")
            break
        else:
            logger.warning(
                f"⚠️ Empty data on attempt {attempt}. Retrying in {retry_delay} seconds..."
            )
            time.sleep(retry_delay)
    else:
        logger.error(
            "🚨 Failed to retrieve base data after multiple attempts. Exiting program."
        )
        return

    # ==========================
    # Phase 3: Preprocessing and Data Loading
    # ==========================
    logger.info("🔧 Phase 3: Preprocessing and Data Loading")
    logger.info("🔗 Connecting to the database...")
    db = DatabaseManager(db_config)
    logger.info("✅ DatabaseManager loaded successfully")

    try:

        logger.info("📥 Starting data loading...")

        # Read and save original data
        anime_df = pd.read_csv(
            StringIO(anime_buffer.getvalue().decode("utf-8"))
        )
        details_df = pd.read_csv(
            StringIO(details_buffer.getvalue().decode("utf-8"))
        )
        scores_df = pd.read_csv(
            StringIO(scores_buffer.getvalue().decode("utf-8"))
        )
        # Processing
        anime_df["premiered"] = anime_df["premiered"].apply(clean_premiered)

        anime_buffer = preprocess_to_memory(
            anime_df,
            columns_to_keep=[
                "anime_id",
                "score",
                "type",
                "episodes",
                "status",
                "duration",
                "genres",
                "aired",
                "keywords",
                "rank",
                "popularity",
                "favorites",
                "scored_by",
                "members",
                "premiered",
                "producers",
                "studios",
                "source",
                "rating",
            ],
            integer_columns=[
                "anime_id",
                "episodes",
                "rank",
                "popularity",
                "favorites",
                "scored_by",
                "members",
                "duration",
            ],
            float_columns=["score"],
        )

        details_df.rename(
            columns={
                "Mal ID": "mal_id",
                "Days Watched": "days_watched",
                "Mean Score": "mean_score",
                "Total Entries": "total_entries",
                "Episodes Watched": "episodes_watched",
                "Gender": "gender",
                "Watching": "watching",
                "Completed": "completed",
                "On Hold": "on_hold",
                "Dropped": "dropped",
                "Plan to Watch": "plan_to_watch",
                "Rewatched": "rewatched",
                "Birthday": "birthday",
            },
            inplace=True,
        )

        details_buffer = preprocess_to_memory(
            details_df,
            columns_to_keep=[
                "mal_id",
                "gender",
                "days_watched",
                "mean_score",
                "birthday",
                "watching",
                "completed",
                "on_hold",
                "dropped",
                "plan_to_watch",
                "total_entries",
                "rewatched",
                "episodes_watched",
            ],
            integer_columns=[
                "mal_id",
                "watching",
                "completed",
                "on_hold",
                "dropped",
                "plan_to_watch",
                "total_entries",
                "rewatched",
                "episodes_watched",
            ],
            float_columns=["days_watched", "mean_score"],
        )

        scores_df.rename(
            columns={
                "User ID": "user_id",
                "Anime ID": "anime_id",
                "Score": "rating",
            },
            inplace=True,
        )

        valid_anime_ids = anime_df["anime_id"].dropna().unique().tolist()
        scores_buffer = preprocess_user_score(
            scores_df,
            columns_to_keep=["user_id", "anime_id", "rating"],
            integer_columns=["user_id", "anime_id", "rating"],
            valid_anime_ids=valid_anime_ids,
        )
        with db.connection() as conn:
            with conn.begin():
                db.copy_from_buffer(conn, anime_buffer, "anime_dataset")
                db.copy_from_buffer(conn, details_buffer, "user_details")
                db.copy_from_buffer(conn, scores_buffer, "user_score")

        logger.info("✅ Data loading completed successfully")
    except Exception as e:
        logger.error(f"🚨 Critical error: {e}")
        raise
    finally:
        if conn:
            conn.close()
            logger.info("🔗 Database connection closed")


if __name__ == "__main__":
    main()
