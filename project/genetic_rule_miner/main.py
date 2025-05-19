"""Main execution pipeline for genetic rule mining."""

import ast

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
            logger.warning(f"Error parsing column {column_name} value '{x}': {e}")
            return []

    df[column_name] = df[column_name].fillna("[]").apply(parse_cell)

def main() -> None:
    """Execute the complete rule mining pipeline."""
    try:
        logger.info("Starting rule mining pipeline")

        # Initialize components
        db_config = DBConfig()
        data_manager = DataManager(db_config)

        # Data loading and preparation
        logger.info("Loading and preprocessing data...")
        user_details, anime_data, user_scores = (
            data_manager.load_and_preprocess_data()
        )
        # Remove 'rating' column from user_scores if it exists
        if "rating" in user_scores.columns:
            user_scores = user_scores.drop(columns=["rating"])

        logger.info("Merging preprocessed data...")
        merged_data = DataManager.merge_data(
            user_scores, user_details, anime_data
        )

        # Clean merged_data
        if "rating" in merged_data.columns:
            logger.info("Dropping rows with unknown ratings...")
            merged_data = merged_data[merged_data["rating"] != "unknown"]

        logger.info("Dropping unnecessary columns...")
        merged_data = merged_data.drop(
            columns=["username", "name", "mal_id", "user_id"], errors="ignore"
        )

        # Convert text columns to list columns
        for col in ["producers", "genres", "keywords"]:
            if col in merged_data.columns:
                logger.info(f"Converting column '{col}' from text to list...")
                convert_text_to_list_column(merged_data, col)

        # Initialize genetic rule miner
        miner = GeneticRuleMiner(
            df=merged_data,
            target="anime_id",
            user_cols=user_details.columns.tolist(),
            pop_size=720,
            generations=10000,
        )
        logger.info("Starting evolution process...")
        miner.evolve()

        # Output results
        high_fitness_rules = miner.get_high_fitness_rules(threshold=0.9)
        rules, ids = high_fitness_rules
        if high_fitness_rules:
            logger.info("\nRules with Fitness >= 0.9:")
            if isinstance(rules, list):
                for idx, rule in enumerate(rules, start=1):
                    fitness = miner.fitness(rule)
                    logger.info(f"Rule {idx}: {rule} (Fitness: {fitness:.4f})")
            else:
                fitness = miner.fitness(rules)
                logger.info(f"Rule 1: {rules} (Fitness: {fitness:.4f})")
        else:
            logger.info("No rules with Fitness >= 0.9 were found.")

        # Database config and save rules
        db_manager = DatabaseManager(config=db_config)
        db_manager.save_rules(rules)

    except Exception as e:
        logger.error("Pipeline failed: %s", str(e), exc_info=True)
        raise


if __name__ == "__main__":
    main()