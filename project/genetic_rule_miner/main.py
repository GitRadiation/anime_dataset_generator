"""Main execution pipeline for genetic rule mining."""


from genetic_rule_miner.config import DBConfig
from genetic_rule_miner.data.manager import DataManager
from genetic_rule_miner.data.preprocessing import clean_string_columns, preprocess_data
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
        logger.info("Loading data from database...")
        user_details, anime_data, user_scores = data_manager.load_data()
        
        logger.info("Merging and preprocessing data...")
        merged_data = DataManager.merge_data(user_scores, user_details, anime_data)
        processed_df = preprocess_data(clean_string_columns(merged_data))
        processed_df.to_csv("datos_limpios.csv", index=False)
        # Genetic algorithm execution
        logger.info("Initializing genetic algorithm...")
        miner = GeneticRuleMiner(
            df=processed_df,
            target='rating',
            user_cols=user_details.columns.tolist(),
            pop_size=50,
            generations=100,
            random_seed=42
        )
        
        logger.info("Starting evolution process...")
        results = miner.evolve()
        
        # Output results
        best_rule = results["best_rule"]
        logger.info("\nBest Rule Found:")
        logger.info(miner.format_rule(best_rule))
        logger.info(f"\nFitness: {results['best_fitness']:.4f}")
        logger.info(f"Support: {miner.calculate_support(best_rule):.4f}")
        logger.info(f"Confidence: {miner.calculate_confidence(best_rule):.4f}")
        
    except Exception as e:
        logger.error("Pipeline failed: %s", str(e), exc_info=True)
        raise


if __name__ == "__main__":
    main()
