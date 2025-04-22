#!/usr/bin/env python3
"""Setup script for creating the genetic rule mining project structure."""

from pathlib import Path

PROJECT_STRUCTURE = {
    "project": {
        "__init__.py": "",
        "main.py": """\"\"\"Main execution pipeline for genetic rule mining.\"\"\"

import logging
from config import DBConfig
from data.manager import DataManager
from data.preprocessing import clean_string_columns, preprocess_data
from models.genetic import GeneticRuleMiner
from utils.logging import setup_logging


def main() -> None:
    \"\"\"Execute the complete rule mining pipeline.\"\"\"
    setup_logging()
    logger = logging.getLogger(__name__)
    
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
        logger.info("\\nBest Rule Found:")
        logger.info(miner.format_rule(best_rule))
        logger.info(f"\\nFitness: {results['best_fitness']:.4f}")
        logger.info(f"Support: {miner.calculate_support(best_rule):.4f}")
        logger.info(f"Confidence: {miner.calculate_confidence(best_rule):.4f}")
        
    except Exception as e:
        logger.error("Pipeline failed: %s", str(e), exc_info=True)
        raise


if __name__ == "__main__":
    main()
""",
        "config.py": """\"\"\"Configuration module for the project.\"\"\"

from dataclasses import dataclass


@dataclass
class DBConfig:
    \"\"\"Database configuration settings.\"\"\"
    host: str = "localhost"
    port: int = 5432
    database: str = "anime_db"
    user: str = "user"
    password: str = "password"
""",
        "data": {
            "__init__.py": "",
            "manager.py": """\"\"\"Data loading and merging operations.\"\"\"

from collections.abc import Sequence
from typing import Tuple
import pandas as pd
from sqlalchemy import create_engine, Engine
from sqlalchemy.exc import SQLAlchemyError
from config import DBConfig
from utils.exceptions import DataValidationError


class DataManager:
    \"\"\"Handles database operations and data preparation.\"\"\"
    
    def __init__(self, db_config: DBConfig):
        \"\"\"Initialize with database configuration.\"\"\"
        self.db_config = db_config
        self.engine = self._create_engine()

    def _create_engine(self) -> Engine:
        \"\"\"Create SQLAlchemy engine instance.\"\"\"
        return create_engine(
            f"postgresql://{self.db_config.user}:{self.db_config.password}@"
            f"{self.db_config.host}:{self.db_config.port}/{self.db_config.database}"
        )

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        \"\"\"Load required datasets from database.\"\"\"
        tables = ["user_details", "anime_dataset", "user_score"]
        try:
            return tuple(pd.read_sql_table(table, self.engine) for table in tables)
        except SQLAlchemyError as e:
            raise RuntimeError("Database operation failed") from e

    @staticmethod
    def merge_data(
        user_scores: pd.DataFrame,
        user_details: pd.DataFrame,
        anime_data: pd.DataFrame
    ) -> pd.DataFrame:
        \"\"\"Merge datasets using relational joins.\"\"\"
        if any(df.empty for df in [user_scores, user_details, anime_data]):
            raise DataValidationError("Cannot merge empty DataFrames")

        # First merge: user scores with user details
        merged = user_scores.merge(
            user_details,
            left_on="user_id",
            right_on="mal_id",
            how="inner",
            validate="many_to_one",
            suffixes=("_score", "")
        )

        # Second merge with anime data
        return merged.merge(
            anime_data,
            on="anime_id",
            how="inner",
            validate="many_to_one"
        ).drop(columns=["user_id", "mal_id", "anime_id"], errors="ignore")
""",
            "preprocessing.py": """\"\"\"Data preprocessing functions.\"\"\"

from collections.abc import Sequence
from typing import Optional
import pandas as pd
from utils.decorators import validate_dataframe
from utils.logging import log_execution


@log_execution
def clean_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    \"\"\"Clean and standardize string columns.\"\"\"
    str_cols = df.select_dtypes(include=["object", "string"]).columns
    df[str_cols] = df[str_cols].transform(
        lambda col: col.str.strip().replace(['\\N', 'nan'], pd.NA)
    )
    return df.dropna(how='all', axis=1)


@log_execution
@validate_dataframe('duration', 'episodes')
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    \"\"\"Preprocess data with feature engineering.\"\"\"
    # Feature transformation
    df['duration'] = df['duration'].str.replace(' min per ep', '').astype(float)
    df['episodes'] = df['episodes'].astype(float)

    # Feature discretization
    df['duration_class'] = pd.cut(
        df['duration'],
        bins=pd.IntervalIndex.from_breaks([0, 20, 25, df['duration'].max() + 1]),
        labels=['short', 'standard', 'long'],
        right=False
    )

    df['episodes_class'] = pd.cut(
        df['episodes'],
        bins=pd.IntervalIndex.from_breaks([0, 12, 24, df['episodes'].max() + 1]),
        labels=['short', 'medium', 'long'],
        right=False
    )

    # Target encoding
    if 'rating_x' in df.columns:
        df['rating'] = pd.cut(
            df['rating_x'],
            bins=[0, 6.9, 7.9, 10],
            labels=['low', 'medium', 'high'],
            right=False
        )
    
    return df.drop(columns=['rating_x'], errors='ignore')
"""
        },
        "models": {
            "__init__.py": "",
            "genetic.py": """\"\"\"Genetic algorithm implementation with collections.abc and NumPy.\"\"\"

from collections.abc import Callable, Sequence, MutableSequence, Mapping
from functools import partial, lru_cache
from typing import Any, Optional, Union
import logging
import numpy as np
import pandas as pd
from utils.exceptions import GeneticAlgorithmError


logger = logging.getLogger(__name__)


class GeneticRuleMiner:
    \"\"\"Genetic algorithm for association rule mining with NumPy optimizations.\"\"\"
    
    OPERATOR_MAP = {
        '<': 'less than',
        '>': 'greater than',
        '==': 'equals'
    }

    def __init__(self, 
                 df: pd.DataFrame, 
                 target: str, 
                 user_cols: Sequence[str],
                 pop_size: int = 50,
                 generations: int = 100,
                 mutation_rate: float = 0.1,
                 random_seed: Optional[int] = None):
        \"\"\"Initialize the genetic algorithm.\"\"\"
        self.df = df.copy()
        self.target = target
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.user_cols = [col for col in user_cols if col in df.columns]
        
        # Initialize NumPy random generator
        self.rng = np.random.default_rng(random_seed)
        
        # Initialize collections
        self.population: MutableSequence[dict] = []
        self.best_fitness_history: MutableSequence[float] = []
        
        # Setup data structures
        self._initialize_data_structures()

    def _initialize_data_structures(self) -> None:
        \"\"\"Initialize all data structures with proper typing.\"\"\"
        self._numeric_cols = self._get_numeric_cols()
        self._categorical_cols = self._get_categorical_cols()
        
        # Pre-compute percentiles using NumPy
        self._percentiles: Mapping[str, np.ndarray] = {
            col: np.percentile(self.df[col].dropna().values, np.arange(25, 76))
            for col in self._numeric_cols
        }
        
        # Initialize population
        self.population = self._init_population()

    @lru_cache(maxsize=None)
    def _get_numeric_cols(self) -> Sequence[str]:
        \"\"\"Get numeric columns with caching.\"\"\"
        return [
            col for col in self.df.select_dtypes(include=np.number).columns
            if col != self.target
        ]

    @lru_cache(maxsize=None)
    def _get_categorical_cols(self) -> Sequence[str]:
        \"\"\"Get categorical columns with caching.\"\"\"
        return [
            col for col in self.df.select_dtypes(include=['object', 'category']).columns
            if col != self.target
        ]

    def _init_population(self) -> MutableSequence[dict]:
        \"\"\"Initialize population of rules.\"\"\"
        population: MutableSequence[dict] = []
        
        create_rule = partial(
            self._create_rule,
            min_user_conds=2,
            max_conditions=5
        )
        
        for _ in range(self.pop_size):
            population.append(create_rule())
            
        return population

    def _create_rule(self, 
                    min_user_conds: int = 2,
                    max_conditions: int = 5) -> dict:
        \"\"\"Create a rule with conditions.\"\"\"
        conditions: MutableSequence[tuple] = []
        
        # Add required user conditions
        user_conds = self.rng.choice(
            self.user_cols, 
            size=min_user_conds, 
            replace=False
        )
        conditions.extend(self._create_condition(col) for col in user_conds)
        
        # Add optional additional conditions
        other_cols = [c for c in self.df.columns 
                     if c not in self.user_cols + [self.target]]
        if other_cols:
            extra = min(max_conditions - min_user_conds, len(other_cols))
            conditions.extend(
                self._create_condition(col) 
                for col in self.rng.choice(other_cols, size=extra, replace=False)
            )
            
        return {
            "conditions": conditions,
            "target": (self.target, 'high')
        }

    def _create_condition(self, col: str) -> tuple[str, str, Union[float, str]]:
        \"\"\"Create a condition for a column.\"\"\"
        if col in self._numeric_cols:
            op = self.rng.choice(['<', '>'])
            value = float(self.rng.choice(self._percentiles[col]))
            return (col, op, value)
        else:
            unique_vals = self.df[col].dropna().unique()
            return (col, '==', str(self.rng.choice(unique_vals)))

    def fitness(self, rule: dict) -> float:
        \"\"\"Calculate rule fitness (support * confidence).\"\"\"
        support = self._vectorized_support(rule["conditions"])
        confidence = self._vectorized_confidence(rule)
        return support * confidence

    def _vectorized_support(self, conditions: Sequence[tuple]) -> float:
        \"\"\"NumPy-optimized support calculation.\"\"\"
        mask = np.ones(len(self.df), dtype=bool)
        for col, op, value in conditions:
            col_data = self.df[col].values
            if op == '<':
                mask &= (col_data < value)
            elif op == '>':
                mask &= (col_data > value)
            elif op == '==':
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    mask &= (col_data == float(value))
                else:
                    mask &= (self.df[col].astype(str).values == str(value))
        return np.mean(mask)

    def _vectorized_confidence(self, rule: dict) -> float:
        \"\"\"NumPy-optimized confidence calculation.\"\"\"
        conditions_mask = self._vectorized_support(rule["conditions"])
        if conditions_mask == 0:
            return 0
        target_mask = (self.df[self.target].values == rule["target"][1])
        return np.mean(target_mask[self._vectorized_support(rule["conditions"])])

    def calculate_support(self, rule: dict) -> float:
        \"\"\"Calculate rule support (coverage of dataset).\"\"\"
        return self._vectorized_support(rule["conditions"])

    def calculate_confidence(self, rule: dict) -> float:
        \"\"\"Calculate rule confidence (accuracy when conditions met).\"\"\"
        return self._vectorized_confidence(rule)

    def format_rule(self, rule: dict) -> str:
        \"\"\"Format rule for human-readable display.\"\"\"
        conditions = []
        for col, op, value in rule["conditions"]:
            readable_op = self.OPERATOR_MAP.get(op, op)
            if col in self._numeric_cols:
                conditions.append(f"{col} {readable_op} {value:.2f}")
            else:
                conditions.append(f"{col} {readable_op} '{value}'")
                
        target_col, target_val = rule["target"]
        return f"IF {' AND '.join(conditions)} THEN {target_col} = {target_val}"

    def mutate(self, rule: dict) -> dict:
        \"\"\"Apply mutation to a rule.\"\"\"
        if self.rng.random() >= self.mutation_rate:
            return rule
            
        idx = self.rng.integers(0, len(rule["conditions"]))
        col, op, value = rule["conditions"][idx]
        
        if col in self._numeric_cols:
            new_op = self.rng.choice(['<', '>'])
            new_value = value * self.rng.uniform(0.8, 1.2)
            rule["conditions"][idx] = (col, new_op, new_value)
        elif col in self._categorical_cols:
            unique_vals = self.df[col].dropna().unique()
            if len(unique_vals) > 1:
                new_value = self.rng.choice([v for v in unique_vals if v != value])
                rule["conditions"][idx] = (col, '==', new_value)
                
        return rule

    def crossover(self, parent1: dict, parent2: dict) -> tuple[dict, dict]:
        \"\"\"Perform crossover between two parent rules.\"\"\"
        min_len = min(len(parent1["conditions"]), len(parent2["conditions"]))
        if min_len < 2:
            return parent1.copy(), parent2.copy()
            
        split = self.rng.integers(1, min_len)
        
        child1 = {
            "conditions": parent1["conditions"][:split] + parent2["conditions"][split:],
            "target": parent1["target"]
        }
        child2 = {
            "conditions": parent2["conditions"][:split] + parent1["conditions"][split:],
            "target": parent2["target"]
        }
        
        # Ensure minimum user conditions
        validator = partial(
            self._validate_rule_conditions,
            min_user_conds=2,
            max_conditions=5
        )
        validator(child1)
        validator(child2)
        
        return child1, child2

    def _validate_rule_conditions(self, 
                                rule: dict,
                                min_user_conds: int,
                                max_conditions: int) -> None:
        \"\"\"Ensure rule meets condition requirements.\"\"\"
        user_conds = sum(1 for cond in rule["conditions"] if cond[0] in self.user_cols)
        while user_conds < min_user_conds and len(rule["conditions"]) < max_conditions:
            col = self.rng.choice(self.user_cols)
            rule["conditions"].append(self._create_condition(col))
            user_conds += 1

    def evolve(self) -> dict:
        \"\"\"Execute the evolution process.\"\"\"
        for generation in range(self.generations):
            # Selection
            parents = self._select_parents()
            
            # Reproduction
            new_population = self._create_new_generation(parents)
            
            # Elitism
            if generation > 0:
                new_population[0] = self._get_best_individual()
                
            self.population = new_population
            self._update_tracking(generation)
            
        return self._compile_results()

    def _select_parents(self) -> Sequence[dict]:
        \"\"\"Select parents using tournament selection.\"\"\"
        tournament_indices = self.rng.choice(
            len(self.population),
            size=(self.pop_size, 5),
            replace=True
        )
        fitness_scores = np.array([self.fitness(rule) for rule in self.population])
        winners = tournament_indices[
            np.arange(self.pop_size), 
            np.argmax(fitness_scores[tournament_indices], axis=1)
        ]
        return [self.population[i] for i in winners]

    def _create_new_generation(self, parents: Sequence[dict]) -> MutableSequence[dict]:
        \"\"\"Create new generation through crossover and mutation.\"\"\"
        new_population: MutableSequence[dict] = []
        for i in range(0, self.pop_size, 2):
            if i+1 < self.pop_size:
                child1, child2 = self.crossover(parents[i], parents[i+1])
                new_population.extend([self.mutate(child1), self.mutate(child2)])
            else:
                new_population.append(self.mutate(parents[i]))
        return new_population

    def _get_best_individual(self) -> dict:
        \"\"\"Get best individual from current population.\"\"\"
        fitness_scores = np.array([self.fitness(rule) for rule in self.population])
        return self.population[np.argmax(fitness_scores)]

    def _update_tracking(self, generation: int) -> None:
        \"\"\"Update tracking metrics.\"\"\"
        current_fitness = np.array([self.fitness(rule) for rule in self.population])
        self.best_fitness_history.append(np.max(current_fitness))
        
        logger.info(
            f"Generation {generation}: "
            f"Best Fitness={np.max(current_fitness):.4f}, "
            f"Avg Fitness={np.mean(current_fitness):.4f}"
        )

    def _compile_results(self) -> dict:
        \"\"\"Compile final results.\"\"\"
        best_idx = np.argmax(self.best_fitness_history)
        return {
            "best_rule": self.population[best_idx],
            "best_fitness": self.best_fitness_history[best_idx],
            "history": {
                "fitness": list(self.best_fitness_history),
                "generations": self.generations
            }
        }
"""
        },
        "utils": {
            "__init__.py": "",
            "exceptions.py": """\"\"\"Custom exceptions for the project.\"\"\"

class DataValidationError(Exception):
    \"\"\"Raised when data validation fails.\"\"\"

class GeneticAlgorithmError(Exception):
    \"\"\"Base exception for genetic algorithm errors.\"\"\"

class PopulationInitializationError(GeneticAlgorithmError):
    \"\"\"Raised when population initialization fails.\"\"\"

class EvolutionError(GeneticAlgorithmError):
    \"\"\"Raised when evolution process fails.\"\"\"
""",
            "decorators.py": """\"\"\"Custom decorators for data validation and logging.\"\"\"

from collections.abc import Callable
from functools import wraps
from typing import Any
import pandas as pd
import time
from utils.exceptions import DataValidationError


def validate_dataframe(*required_columns: str) -> Callable:
    \"\"\"Decorator to validate DataFrame columns.
    
    Args:
        required_columns: Column names to validate
        
    Returns:
        Decorator function
    \"\"\"
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(df: pd.DataFrame, *args, **kwargs) -> Any:
            missing = [col for col in required_columns if col not in df.columns]
            if missing:
                raise DataValidationError(f"Missing columns: {missing}")
            return func(df, *args, **kwargs)
        return wrapper
    return decorator


def log_execution(func: Callable) -> Callable:
    \"\"\"Decorator to log function execution time.\"\"\"
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        print(f"Executed {func.__name__} in {end_time - start_time:.4f} seconds")
        return result
    return wrapper
""",
            "logging.py": """\"\"\"Logging configuration for the project.\"\"\"

import logging
from typing import Optional


def setup_logging(level: int = logging.INFO,
                 format_str: Optional[str] = None) -> None:
    \"\"\"Configure logging for the project.
    
    Args:
        level: Logging level
        format_str: Custom format string
    \"\"\"
    if format_str is None:
        format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        
    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=[logging.StreamHandler()]
    )
"""
        }
    }
}


def create_project_structure(base_path: Path, structure: dict) -> None:
    """Recursively create the project structure."""
    for name, content in structure.items():
        path = base_path / name
        if isinstance(content, dict):
            path.mkdir(exist_ok=True)
            create_project_structure(path, content)
        else:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)


def main():
    """Main function to set up the project."""
    project_path = Path.cwd() / "genetic_rule_miner"
    project_path.mkdir(exist_ok=True)
    
    print(f"Creating project structure at {project_path}")
    create_project_structure(project_path, PROJECT_STRUCTURE["project"])
    print("Project setup completed successfully!")
    
    # Make the main script executable
    main_script = project_path / "main.py"
    main_script.chmod(main_script.stat().st_mode | 0o111)
    print(f"You can now run the project with: python {main_script}")


if __name__ == "__main__":
    main()