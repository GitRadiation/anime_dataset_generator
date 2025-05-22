import ast
import logging
import re
import threading
import time
from io import BytesIO, StringIO

import nltk
import numpy as np
import pandas as pd
from genetic_rule_miner.bbdd_maker.anime_service import AnimeService
from genetic_rule_miner.bbdd_maker.details_service import DetailsService
from genetic_rule_miner.bbdd_maker.score_service import ScoreService
from genetic_rule_miner.bbdd_maker.user_service import UserService
from genetic_rule_miner.config import APIConfig, DBConfig
from genetic_rule_miner.data.database import DatabaseManager
from genetic_rule_miner.data.preprocessing import preprocess_data
from genetic_rule_miner.models.genetic import GeneticRuleMiner
from genetic_rule_miner.utils.logging import LogManager
from genetic_rule_miner.utils.rule import Rule

# Configure logging
LogManager.configure()
logger = logging.getLogger(__name__)


def convert_text_to_list_column(df, column_name):
    """
    Convierte una columna que contiene strings representando listas
    en una lista real de Python. Si el valor no es una lista válida,
    la convierte en lista con un solo elemento.
    """

    def parse_cell(x):
        try:
            # Si es string y comienza con '[', intenta parsear
            if isinstance(x, str) and x.startswith("["):
                parsed = ast.literal_eval(x)
                if isinstance(parsed, list):
                    # Limpiar elementos vacíos o espacios
                    return [str(i).strip() for i in parsed if str(i).strip()]
                else:
                    return [str(parsed).strip()]
            elif isinstance(x, str):
                # No es lista, pero es string: lo convierte en lista de un elemento
                return [x.strip()] if x.strip() else []
            elif isinstance(x, list):
                # Ya es lista
                return [str(i).strip() for i in x if str(i).strip()]
            else:
                # Cualquier otro tipo, convertir a str y poner en lista
                return [str(x).strip()] if x else []
        except Exception as e:
            logger.warning(
                f"Error parsing column {column_name} value '{x}': {e}"
            )
            return []

    df[column_name] = df[column_name].fillna("[]").apply(parse_cell)


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
    df: pd.DataFrame, columns_to_keep, integer_columns, float_columns=None
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

    df = preprocess_data(df)
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


def validate_and_cleanup_rules(
    db,
    user_details,
    anime_df,
    user_scores,
    logger,
    fitness_threshold=0.95,
    confidence_threshold=0.95,
    skip=False,
):
    """
    Valida las reglas en la base de datos y elimina aquellas que no cumplen los umbrales.
    Si skip=True, no hace nada.
    """
    if skip:
        logger.info(
            "No changes detected in database. Skipping rule validation/cleanup."
        )
        return
    # Cargar reglas desde la base de datos
    from sqlalchemy import text

    with db.connection() as conn:
        # Elimina el límite de 1000 resultados (por defecto en algunos drivers)
        # Forzar el uso de un generador para evitar limitaciones de tamaño
        result = conn.execution_options(stream_results=True).execute(
            text("SELECT rule_id, conditions, target_value FROM rules")
        )
        rules = []
        rule_ids = []
        for row in result.mappings():
            try:
                conds = row["conditions"]
                if isinstance(conds, str):
                    import json

                    conds = json.loads(conds)

                rule = Rule(
                    columns=[],  # No se usa para la lógica
                    conditions=conds,
                    target=np.int64(row["target_value"]),
                )
                rules.append(rule)
                rule_ids.append(row["rule_id"])
                rule_ids.append(row["rule_id"])
            except Exception as e:
                logger.warning(f"Error parsing rule {row['rule_id']}: {e}")

    if not rules:
        logger.info("No rules found in the database for validation.")
        return

    # Preparar datos para evaluación usando DataManager
    from genetic_rule_miner.data.manager import DataManager

    db_config = (
        db.config
        if hasattr(db, "config") and db.config is not None
        else DBConfig()
    )
    data_manager = DataManager(db_config)
    user_details, anime_data, user_scores = (
        data_manager.load_and_preprocess_data()
    )
    merged_data = DataManager.merge_data(user_scores, user_details, anime_data)
    if "rating" in merged_data.columns:
        merged_data = merged_data[merged_data["rating"] != "unknown"]

    merged_data = merged_data.drop(
        columns=["username", "name", "mal_id", "user_id"], errors="ignore"
    )

    for col in ["producers", "genres", "keywords"]:
        if col in merged_data.columns:
            logger.info(f"Converting column '{col}' from text to list...")
            convert_text_to_list_column(merged_data, col)

    # Instanciar el evaluador
    user_details_columns = [col for col in user_details.columns]
    miner = GeneticRuleMiner(
        df=merged_data,
        target_column="anime_id",
        user_cols=user_details_columns,
        pop_size=10,
        generations=1,
    )

    # Evaluar reglas
    to_delete = []
    deleted_fitness_conf_pairs = set()
    for rule, rule_id in zip(rules, rule_ids):
        fitness = round(float(miner.fitness(rule)), 3)
        confidence = round(float(miner._vectorized_confidence(rule)), 3)
        if fitness < fitness_threshold or confidence < confidence_threshold:
            to_delete.append(rule_id)
            deleted_fitness_conf_pairs.add((fitness, confidence))

    # Eliminar reglas no válidas (borrado masivo)
    if to_delete:
        from sqlalchemy import text

        with db.connection() as conn:
            conn.execute(
                text("DELETE FROM rules WHERE rule_id = ANY(:rids)"),
                {"rids": to_delete},
            )
            conn.commit()
        logger.info(
            f"Deleted {len(to_delete)} rules not meeting thresholds. "
            f"Unique (fitness, confidence) pairs of deleted: {sorted(deleted_fitness_conf_pairs)}"
        )
    else:
        logger.info("All rules meet the required thresholds.")


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

    conn = None

    # ==========================
    # Phase 2: User Data Retrieval
    # ==========================
    logger.info("📡 Phase 2: User Data Retrieval")
    for attempt in range(1, max_retries + 1):
        logger.info(f"🤔 Attempt {attempt} to retrieve user data...")

        # 1. Generate user list
        logger.info("📡 Generating user list...")
        user_start_id = np.random.randint(1, 56500)
        user_end_id = user_start_id + 199
        user_service = UserService(api_config)
        userlist_buffer = user_service.generate_userlist(
            user_start_id, user_end_id
        )
        userlist_buffer.seek(0)

        if userlist_buffer:
            logger.info("✅ Successfully retrieved user list")
            break
        else:
            logger.warning(
                f"⚠️ Empty user list on attempt {attempt}. Retrying in {retry_delay} seconds..."
            )
            time.sleep(retry_delay)
    else:
        logger.error(
            "🚨 Failed to retrieve user list after multiple attempts. Exiting program."
        )
        return

    # 2. Prepare data for ScoreService
    logger.info("📡 Preparing user data...")
    userlist_df = pd.read_csv(userlist_buffer)
    userlist_df.rename(columns={"user_id": "mal_id"}, inplace=True)
    modified_userlist_buffer = BytesIO()
    userlist_df.to_csv(modified_userlist_buffer, index=False)
    modified_userlist_buffer.seek(0)

    # 3. Retrieve user details
    logger.info("📡 Retrieving user details...")
    usernames = userlist_df["username"].dropna().tolist()
    details_service = DetailsService(api_config)
    details_buffer = details_service.get_user_details(usernames)

    if not details_buffer:
        logger.error("🚨 Failed to retrieve user details. Exiting program.")
        return

    # 4. Start anime data retrieval in parallel
    anime_data_result = {}

    def fetch_anime_data():
        logger.info("📡 Retrieving anime data (in parallel)...")
        anime_start_id = np.random.randint(1, 99800)
        anime_end_id = anime_start_id + 199
        anime_buffer = AnimeService(api_config).get_anime_data(
            anime_start_id, anime_end_id
        )
        anime_data_result["anime_buffer"] = anime_buffer

    anime_thread = threading.Thread(target=fetch_anime_data)
    anime_thread.start()

    # ==========================
    # Phase 3: Preprocessing and User Data Loading
    # ==========================
    logger.info("🔧 Phase 3: Preprocessing and User Data Loading")
    logger.info("🔗 Connecting to the database...")
    db = DatabaseManager(db_config)
    logger.info("✅ DatabaseManager loaded successfully")

    try:
        logger.info("📥 Starting user data loading...")

        details_df = pd.read_csv(
            StringIO(details_buffer.getvalue().decode("utf-8"))
        )
        details_df.rename(
            columns={
                "Mal ID": "mal_id",
                "Username": "username",
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

        details_buffer_proc = preprocess_to_memory(
            details_df,
            columns_to_keep=[
                "mal_id",
                "gender",
                "days_watched",
                "mean_score",
                "username",
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

        with db.connection() as conn:
            with conn.begin():
                details_changed = db.copy_from_buffer(
                    conn, details_buffer_proc, "user_details"
                )

        # 5. Export users for scores retrieval (can be done after user upload)
        users_csv_buffer = db.export_users_to_csv_buffer()

        # 6. Retrieve scores (can be done after users are uploaded)
        for attempt in range(1, max_retries + 1):
            logger.info("📡 Retrieving user scores...")
            score_service = ScoreService(api_config)
            scores_buffer = score_service.get_scores(users_csv_buffer)
            if hasattr(scores_buffer, "getvalue") and scores_buffer.getvalue():
                logger.info("✅ Successfully retrieved scores data")
                break
            else:
                logger.warning(
                    f"⚠️ Empty scores data on attempt {attempt}. Retrying in {retry_delay} seconds..."
                )
                time.sleep(retry_delay)
        else:
            logger.error(
                "🚨 Failed to retrieve scores after multiple attempts. Exiting program."
            )
            return

        # 7. Esperar a que termine la obtención de animes
        anime_thread.join()
        anime_buffer = anime_data_result.get("anime_buffer")
        if not anime_buffer:
            logger.error("🚨 Failed to retrieve anime data. Exiting program.")
            return

        # 8. Procesar y subir datos de anime
        anime_df = pd.read_csv(
            StringIO(anime_buffer.getvalue().decode("utf-8"))
        )
        anime_df["premiered"] = anime_df["premiered"].apply(clean_premiered)

        anime_buffer_proc = preprocess_to_memory(
            anime_df,
            columns_to_keep=[
                "anime_id",
                "name",
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

        with db.connection() as conn:
            with conn.begin():
                anime_changed = db.copy_from_buffer(
                    conn, anime_buffer_proc, "anime_dataset"
                )

        # 9. Procesar y subir scores (solo después de usuarios y animes)
        scores_df = pd.read_csv(
            StringIO(scores_buffer.getvalue().decode("utf-8"))
        )
        scores_df.rename(
            columns={
                "User ID": "user_id",
                "Anime ID": "anime_id",
                "Score": "rating_x",
            },
            inplace=True,
        )

        scores_df = preprocess_data(scores_df)
        scores_df = scores_df[scores_df["rating"] == "high"]

        valid_anime_ids = anime_df["anime_id"].dropna().unique().tolist()
        scores_buffer_proc = preprocess_user_score(
            scores_df,
            columns_to_keep=["user_id", "anime_id", "rating"],
            integer_columns=["user_id", "anime_id"],
            valid_anime_ids=valid_anime_ids,
        )

        with db.connection() as conn:
            with conn.begin():
                score_changed = db.copy_from_buffer(
                    conn, scores_buffer_proc, "user_score"
                )

        db_changed = anime_changed or details_changed or score_changed

        # === VALIDAR Y LIMPIAR REGLAS DESPUÉS DE LA INSERCIÓN ===
        if db_changed:
            logger.info(
                "🔎 Validating rules in the database (fitness/confidence > 0.95)..."
            )
            # Consultar los datos actuales desde la base de datos
            with db.connection() as conn:
                user_details = pd.read_sql("SELECT * FROM user_details", conn)
                anime_df = pd.read_sql("SELECT * FROM anime_dataset", conn)
                user_scores = pd.read_sql("SELECT * FROM user_score", conn)

            validate_and_cleanup_rules(
                db, user_details, anime_df, user_scores, logger
            )
        else:
            logger.info(
                "No changes in database tables. Skipping rule validation/cleanup."
            )

    except Exception as e:
        logger.error(f"🚨 Critical error: {e}")
        raise
    finally:
        if conn:
            conn.close()
            logger.info("🔗 Database connection closed")


if __name__ == "__main__":
    main()
