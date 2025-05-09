"""Main execution pipeline for genetic rule mining."""

import pandas as pd

from genetic_rule_miner.config import DBConfig
from genetic_rule_miner.data.manager import DataManager
from genetic_rule_miner.data.preprocessing import (
    clean_string_columns,
    preprocess_data,
)
from genetic_rule_miner.models.genetic import GeneticRuleMiner
from genetic_rule_miner.utils.logging import LogManager, log_execution


@log_execution
def save_to_excel(
    df_dict: dict, output_path: str = "processed_data.xlsx"
) -> None:
    """Save the DataFrames to an Excel file with different sheets.

    Args:
        df_dict: Dictionary where keys are sheet names and values are DataFrames to save.
        output_path: Path to save the Excel file.
    """
    try:
        # Save the dictionary of DataFrames to an Excel file with each DataFrame in a separate sheet
        with pd.ExcelWriter(output_path) as writer:
            for sheet_name, df in df_dict.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        logger.info(f"Data successfully saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save data to Excel: {str(e)}")
        raise


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
        save_to_excel(
            {
                "user_details": user_details,
                "anime_data": anime_data,
                "user_scores": user_scores,
            },
            "raw_data.xlsx",
        )
        logger.info("Merging preprocessed data...")
        merged_data = DataManager.merge_data(
            user_scores, user_details, anime_data
        )
        save_to_excel({"merged_data": merged_data}, "processed_data.xlsx")

        # Genetic algorithm execution
        logger.info("Initializing genetic algorithm...")
        processed_df = preprocess_data(clean_string_columns(merged_data))
        save_to_excel({"processed_data": processed_df}, "datos_limpios.xlsx")

        miner = GeneticRuleMiner(
            df=processed_df,
            target="rating",
            user_cols=user_details.columns.tolist(),
            pop_size=250,
            generations=100,
        )
        logger.info("Starting evolution process...")
        results = miner.evolve()

        # Output results
        best_rule = results["best_rule"]
        logger.info("\nBest Rule Found:")
        logger.info(miner.format_rule(best_rule))
        logger.info(f"\nFitness: {results['best_fitness']:.4f}")
        logger.info(f"Support: {results['best_support']}")
        logger.info(f"Confidence: {results['best_confidence']}")

    except Exception as e:
        logger.error("Pipeline failed: %s", str(e), exc_info=True)
        raise


if __name__ == "__main__":
    main()
