"""Main execution pipeline for genetic rule mining."""

import ast
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

from genetic_rule_miner.config import DBConfig
from genetic_rule_miner.data.database import DatabaseManager
from genetic_rule_miner.data.manager import DataManager
from genetic_rule_miner.models.genetic import GeneticRuleMiner
from genetic_rule_miner.utils.logging import LogManager

LogManager.configure()
logger = LogManager.get_logger(__name__)


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

def main() -> None:
    """Execute the complete rule mining pipeline."""
    try:
        logger.info("Starting rule mining pipeline")

        # Initialize components
        db_config = DBConfig()
        data_manager = DataManager(db_config)

        logger.info("Loading and preprocessing data...")
        user_details, anime_data, user_scores = data_manager.load_and_preprocess_data()
        if "rating" in user_scores.columns:
            user_scores = user_scores.drop(columns=["rating"])

        logger.info("Merging preprocessed data...")
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

        logger.info("Preparing rule mining tasks...")
        db_manager = DatabaseManager(config=db_config)
        targets = db_manager.get_anime_ids_without_rules()

        total_targets = len(targets)
        logger.info(f"Total targets to process: {total_targets}")

        start_time = time.perf_counter()
        completed = 0
        submitted = 0
        max_workers = 4
        pending_targets = iter(targets)

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_tid = {}

            # Lanzamos los primeros hasta completar el pool
            while submitted < max_workers and submitted < total_targets:
                tid = next(pending_targets)
                future = executor.submit(
                    GeneticRuleMiner(
                        df=merged_data,
                        target_column="anime_id",
                        user_cols=user_details.columns.tolist(),
                        pop_size=720,
                        generations=10000,
                    ).evolve_per_target,
                    tid,
                )
                future_to_tid[future] = tid
                submitted += 1

            while future_to_tid:
                for future in as_completed(future_to_tid):
                    tid = future_to_tid.pop(future)
                    try:
                        result = future.result()
                        if result:
                            db_manager.save_rules(result)
                            logger.info(f"✅ Target {tid} finished and saved {len(result)} rules.")
                        else:
                            logger.info(f"⚠️ Target {tid} finished with no rules.")
                    except Exception as exc:
                        logger.error(f"❌ Target {tid} failed: {type(exc).__name__}: {exc}")
                        logger.error(traceback.format_exc())

                    completed += 1
                    if completed % 4 == 0:
                        logger.info(f"✅ {completed} targets completed.")

                    # Lanzar siguiente target si queda
                    try:
                        tid = next(pending_targets)
                        future = executor.submit(
                            GeneticRuleMiner(
                                df=merged_data,
                                target_column="anime_id",
                                user_cols=user_details.columns.tolist(),
                                pop_size=720,
                                generations=10000,
                            ).evolve_per_target,
                            tid,
                        )
                        future_to_tid[future] = tid
                        submitted += 1
                    except StopIteration:
                        continue

        duration = time.perf_counter() - start_time
        logger.info(f"🎉 Evolution process completed in {duration:.2f} seconds. Total targets: {completed}")

    except Exception as e:
        logger.error("Pipeline failed: %s", str(e), exc_info=True)
        raise



if __name__ == "__main__":
    main()
