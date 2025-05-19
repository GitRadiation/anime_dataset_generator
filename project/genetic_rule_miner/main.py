"""Main execution pipeline for genetic rule mining."""

from genetic_rule_miner.config import DBConfig
from genetic_rule_miner.data.database import DatabaseManager
from genetic_rule_miner.data.manager import DataManager
from genetic_rule_miner.models.genetic import GeneticRuleMiner
from genetic_rule_miner.utils.logging import LogManager

LogManager.configure()
logger = LogManager.get_logger(__name__)


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

        # Genetic algorithm execution
        logger.info("Initializing genetic algorithm...")
        # Drop rows with unknown rating (if the 'rating' column exists)
        if "rating" in merged_data.columns:
            logger.info("Dropping rows with unknown ratings...")
            merged_data = merged_data[merged_data["rating"] != "unknown"]

        # Drop unnecessary columns: 'rating_x' and 'age' if they exist
        logger.info("Dropping unnecessary columns...")
        import ast

        for col in ["producers", "genres", "keywords"]:
            if col in merged_data.columns:
                merged_data[col] = (
                    merged_data[col]
                    .fillna("[]")  # String de lista vacía
                    .astype(str)
                    .apply(
                        lambda x: (
                            ast.literal_eval(x) if x.startswith("[") else [x]
                        )
                    )
                    .apply(
                        lambda x: (
                            [i.strip() for i in x if i.strip()]
                            if isinstance(x, list)
                            else []
                        )
                    )
                )
        merged_data = merged_data.drop(
            columns=["username", "name", "mal_id", "user_id"], errors="ignore"
        )
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

        # Configuración de la base de datos
        db_manager = DatabaseManager(config=db_config)

        # Guardar las reglas en la tabla "rules"
        db_manager.save_rules(rules)

    except Exception as e:
        logger.error("Pipeline failed: %s", str(e), exc_info=True)
        raise


if __name__ == "__main__":
    main()
