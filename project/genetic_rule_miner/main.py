"""Main execution pipeline for genetic rule mining."""

import ast
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sqlalchemy import text

from genetic_rule_miner.config import DBConfig
from genetic_rule_miner.data.database import DatabaseManager
from genetic_rule_miner.data.manager import DataManager
from genetic_rule_miner.models.genetic import GeneticRuleMiner
from genetic_rule_miner.utils.logging import LogManager
from genetic_rule_miner.utils.rule import Rule

LogManager.configure()
logger = LogManager.get_logger(__name__)


def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Optimiza el DataFrame para acceso secuencial y reduce fragmentación de memoria."""
    # Ordenar columnas por tipo para mejorar localidad
    cols_ordered = [
        col for col in df.columns if df[col].dtype.kind in "biufc"
    ] + [  # Numéricos
        col for col in df.columns if df[col].dtype.kind in "O"
    ]  # Objetos
    df = df[cols_ordered].copy()

    # Convertir a tipos más eficientes
    for col in df.select_dtypes(include=["int64"]):
        df[col] = pd.to_numeric(df[col], downcast="integer")
    for col in df.select_dtypes(include=["float64"]):
        df[col] = pd.to_numeric(df[col], downcast="float")

    return df


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


def process_target(tid, merged_data, user_details_columns, db_config):
    """Procesa un target individual para minería genética de reglas."""
    try:
        rules = GeneticRuleMiner(
            df=merged_data,
            target_column="anime_id",
            user_cols=user_details_columns,
            pop_size=720,
            generations=10000,
        ).evolve_per_target(tid)

        if rules:
            db_manager = DatabaseManager(config=db_config)
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


def remove_obsolete_rules(db_manager, merged_data, user_details):
    """
    Elimina reglas obsoletas:
    - cuyo target_value ya no existe en anime_dataset
    - o cuyo fitness o confidence < 0.95
    Muestra el fitness y score de cada regla eliminada.
    Borra en lotes para evitar problemas de demasiados parámetros.
    Procesa las reglas en lotes de 10,000.
    """

    with db_manager.connection() as conn:
        # 1. Eliminar reglas cuyo target_value no existe
        obsolete_rules = conn.execute(
            text(
                """
                SELECT rule_id FROM rules
                WHERE NOT EXISTS (
                    SELECT 1 FROM anime_dataset
                    WHERE anime_id = rules.target_value::integer
                )
                """
            )
        ).fetchall()
        rule_ids = [row[0] for row in obsolete_rules]
        rule_infos = []

        # 2. Eliminar reglas con fitness o confidence < 0.95 en lotes de 10k
        offset = 0
        batch_size_rules = 10000

        miner = GeneticRuleMiner(
            df=merged_data,
            target_column="anime_id",
            user_cols=user_details.columns.tolist(),
            pop_size=10,
            generations=1,
        )

        while True:
            rules_rows = conn.execute(
                text(
                    "SELECT rule_id, conditions, target_value FROM rules ORDER BY rule_id OFFSET :offset LIMIT :limit"
                ),
                {"offset": offset, "limit": batch_size_rules},
            ).fetchall()
            if not rules_rows:
                break
            for row in rules_rows:
                rule_id, conditions, target_value = row
                try:
                    if isinstance(conditions, str):
                        conditions = ast.literal_eval(conditions)
                    rule = Rule(
                        columns=[],
                        conditions=conditions,
                        target=np.int64(target_value),
                    )
                    fitness = miner.fitness(rule)
                    confidence = miner._vectorized_confidence(rule)
                    if fitness < 0.95 or confidence < 0.95:
                        rule_ids.append(rule_id)
                        rule_infos.append((rule_id, fitness, confidence))
                except Exception as e:
                    logger.warning(
                        f"No se pudo evaluar la regla {rule_id}: {e}"
                    )
                    rule_ids.append(rule_id)
                    rule_infos.append((rule_id, "error", "error"))
            offset += batch_size_rules

        # Eliminar todas las reglas obsoletas en masa y mostrar info
        if rule_ids:
            logger.info(
                "Eliminando reglas obsoletas (rule_id, fitness, confidence):"
            )
            for info in rule_infos:
                logger.info(f"  {info}")
            # Borrado por lotes para evitar demasiados parámetros
            batch_size = 1000
            for i in range(0, len(rule_ids), batch_size):
                batch = rule_ids[i : i + batch_size]
                conn.execute(
                    text("DELETE FROM rules WHERE rule_id = ANY(:batch)"),
                    {"batch": batch},
                )
            logger.info(f"Eliminadas {len(rule_ids)} reglas obsoletas.")
        else:
            logger.info("No se encontraron reglas obsoletas para eliminar.")


def main() -> None:
    """Execute the complete rule mining pipeline using sequential processing (no joblib)."""
    try:
        logger.info("Starting rule mining pipeline")

        # Initialize components
        db_config = DBConfig()
        data_manager = DataManager(db_config)

        logger.info("Loading and preprocessing data...")
        user_details, anime_data, user_scores = (
            data_manager.load_and_preprocess_data()
        )
        if "rating" in user_scores.columns:
            user_scores = user_scores.drop(columns=["rating"])

        logger.info("Merging preprocessed data...")
        merged_data = DataManager.merge_data(
            user_scores, user_details, anime_data
        )

        if "rating" in merged_data.columns:
            merged_data = merged_data[merged_data["rating"] != "unknown"]

        merged_data = merged_data.drop(
            columns=["username", "name", "mal_id", "user_id"], errors="ignore"
        )

        for col in ["producers", "genres", "keywords"]:
            if col in merged_data.columns:
                logger.info(f"Converting column '{col}' from text to list...")
                convert_text_to_list_column(merged_data, col)
        merged_data = optimize_dataframe(merged_data.drop(columns=["rating"]))

        logger.info("Preparing rule mining tasks...")
        db_manager = DatabaseManager(config=db_config)
        # Eliminar reglas obsoletas antes de procesar
        remove_obsolete_rules(db_manager, merged_data, user_details)
        targets = db_manager.get_anime_ids_without_rules() or []
        total_targets = len(targets)
        logger.info(f"Total targets to process: {total_targets}")

        start_time = time.perf_counter()

        # Procesamiento paralelo con joblib
        results = list(
            Parallel(n_jobs=-1, prefer="processes", verbose=10)(
                delayed(process_target)(
                    tid, merged_data, user_details.columns.tolist(), db_config
                )
                for tid in targets
            )
        )

        completed = len(results)
        logger.info(f"✅ {completed} targets completed.")

        duration = time.perf_counter() - start_time
        logger.info(
            f"🎉 Evolution process completed in {duration:.2f} seconds. Total targets: {completed}"
        )

    except Exception as e:
        logger.error("Pipeline failed: %s", str(e), exc_info=True)
        raise


if __name__ == "__main__":
    main()
