"""Main execution pipeline for genetic rule mining."""

import argparse
import ast
import time

import pandas as pd
from joblib import Parallel, delayed
from sqlalchemy import text

from genetic_rule_miner.config import DBConfig
from genetic_rule_miner.data.database import DatabaseManager
from genetic_rule_miner.data.manager import DataManager
from genetic_rule_miner.models.genetic import GeneticRuleMiner
from genetic_rule_miner.utils.logging import LogManager

LogManager.configure()
logger = LogManager.get_logger(__name__)


def convert_text_to_list_column(df: pd.DataFrame, column_name: str) -> None:
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


def remove_obsolete_rules_for_target(
    target_id: int,
    merged_data: pd.DataFrame,
    user_details: pd.DataFrame,
    db_config: DBConfig,
) -> None:
    BATCH_SIZE = 500
    db_manager = DatabaseManager(config=db_config)
    to_delete = []

    try:
        with db_manager.connection() as conn:
            if target_id not in merged_data["anime_id"].values:
                conn.execute(
                    text("DELETE FROM rules WHERE target_value = :target_id"),
                    {"target_id": target_id},
                )
                logger.info(
                    f"Eliminadas todas las reglas del target_id {target_id} (no existe en dataset)"
                )
                return

            offset = 0
            while True:
                # Obtener reglas en lotes
                rules_with_id = db_manager.get_rules_by_target_value_paginated(
                    target_id, offset=offset, limit=BATCH_SIZE
                )
                if not rules_with_id:
                    break  # ya no quedan reglas

                filtered_data = merged_data[
                    merged_data["anime_id"] == target_id
                ].copy()
                miner = GeneticRuleMiner(
                    df=filtered_data,
                    target_column="anime_id",
                    user_cols=user_details.columns.tolist(),
                    pop_size=1,
                )

                rules = [r.rule_obj for r in rules_with_id]
                fitness_arr = miner.batch_vectorized_confidence(rules)
                support_arr = miner.batch_vectorized_support(rules)

                rule_id_list = [r.rule_id for r in rules_with_id]

                for idx, rule_id in enumerate(rule_id_list):
                    fitness = fitness_arr[idx]
                    support = support_arr[idx]
                    if fitness < 1 or support < 0.95:
                        to_delete.append(rule_id)
                        logger.info(
                            f"Eliminando regla {rule_id} (fitness: {fitness:.4f}, soporte: {support:.4f})"
                        )

                # Pasar al siguiente lote
                offset += BATCH_SIZE
            if to_delete:
                conn.execute(
                    text(
                        "DELETE FROM rules WHERE rule_id = ANY(:ids::uuid[])"
                    ),
                    {"ids": to_delete},
                )
                logger.info(
                    f"Eliminadas {len(to_delete)} reglas obsoletas para target {target_id} (offset {offset})"
                )
    except Exception as e:
        logger.error(
            f"Fallo al eliminar reglas para target {target_id}: {e}",
            exc_info=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="genetic_rule_miner",
        description="Pipeline para minería genética de reglas sobre datos de anime y usuarios.",
        epilog="Ejemplo de uso: python main.py --pop_size 256 --generations 500 --mutation_rate 0.05",
    )

    parser.add_argument(
        "--pop_size",
        type=int,
        default=512,
        help="Tamaño de la población inicial del algoritmo genético. Por defecto: 512.",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=720,
        help="Número de generaciones a ejecutar durante la evolución. Por defecto: 720.",
    )
    parser.add_argument(
        "--mutation_rate",
        type=float,
        default=0.10,
        help="Tasa de mutación (valor entre 0 y 1). Por defecto: 0.10.",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=None,
        help="Semilla aleatoria para reproducibilidad. Si no se especifica, se genera aleatoriamente.",
    )
    parser.add_argument(
        "--max_stagnation",
        type=int,
        default=100,
        help="Número máximo de generaciones sin mejora antes de terminar. Por defecto: 100.",
    )
    args = parser.parse_args()

    def process_target_with_params(
        tid,
        merged_data,
        user_details_columns,
        db_config,
        pop_size,
        generations,
        mutation_rate,
        random_seed,
        max_stagnation,
    ):
        db_manager = DatabaseManager(config=db_config)
        try:
            filtered_data = merged_data[merged_data["anime_id"] == tid].copy()
            if filtered_data.empty:
                return (tid, False, "No data for target")
            rules = GeneticRuleMiner(
                df=filtered_data,
                target_column="anime_id",
                user_cols=user_details_columns,
                db_manager=db_manager,
                pop_size=pop_size,
                generations=generations,
                mutation_rate=mutation_rate,
                random_seed=random_seed,
                max_stagnation=max_stagnation,
            ).evolve_per_target(tid)

            if rules:
                db_manager.save_rules(rules)
                logger.info(
                    f"✅ Target {tid} finished and saved {len(rules)} rules."
                )
            else:
                logger.info(f"⚠️ Target {tid} finished with no rules.")
            return (tid, True, None)
        except Exception as e:
            logger.error(f"❌ Target {tid} failed: {e}")
            return (tid, False, str(e))

    def main() -> None:
        """Execute the complete rule mining pipeline using sequential processing (no joblib)."""
        try:
            logger.info("Starting rule mining pipeline with custom parameters")

            db_config = DBConfig()
            data_manager = DataManager(db_config)

            user_details, anime_data, user_scores = (
                data_manager.load_and_preprocess_data()
            )
            if "rating" in user_scores.columns:
                user_scores = user_scores.drop(columns=["rating"])

            merged_data = DataManager.merge_data(
                user_scores, user_details, anime_data
            )

            if "rating" in merged_data.columns:
                merged_data = merged_data[merged_data["rating"] != "unknown"]

            merged_data = merged_data.drop(
                columns=["username", "name", "mal_id", "user_id"],
                errors="ignore",
            )

            for col in ["producers", "genres", "keywords"]:
                if col in merged_data.columns:
                    convert_text_to_list_column(merged_data, col)

            db_manager = DatabaseManager(config=db_config)

            Parallel(n_jobs=-1, prefer="threads", verbose=10)(
                delayed(remove_obsolete_rules_for_target)(
                    int(tid), merged_data, user_details, db_config
                )
                for tid in merged_data["anime_id"].unique()
            )

            targets = db_manager.get_anime_ids_without_rules() or []
            logger.info(f"Total targets to process: {len(targets)}")

            start_time = time.perf_counter()

            results = list(
                Parallel(n_jobs=-1, prefer="threads", verbose=10)(
                    delayed(
                        lambda tid: process_target_with_params(
                            tid,
                            merged_data,
                            user_details.columns.tolist(),
                            db_config,
                            args.pop_size,
                            args.generations,
                            args.mutation_rate,
                            args.random_seed,
                            args.max_stagnation,
                        )
                    )(tid)
                    for tid in targets
                )
            )

            logger.info(f"✅ {len(results)} targets completed.")
            logger.info(
                f"🎉 Completed in {time.perf_counter() - start_time:.2f} seconds."
            )
        except Exception as e:
            logger.error("Pipeline failed: %s", str(e), exc_info=True)
            raise

    main()
