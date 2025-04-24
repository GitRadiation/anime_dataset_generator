"""Genetic algorithm implementation with collections.abc and NumPy."""

from collections.abc import Mapping, MutableSequence, Sequence
from functools import lru_cache, partial
from typing import Optional, Union

import numpy as np
import pandas as pd

from genetic_rule_miner.utils.logging import LogManager

logger = LogManager.get_logger(__name__)


class GeneticRuleMiner:
    OPERATOR_MAP = {"<": "less than", ">": "greater than", "==": "equals"}

    def __init__(
        self,
        df: pd.DataFrame,
        target: str,
        user_cols: Sequence[str],
        pop_size: int = 50,
        generations: int = 100,
        mutation_rate: float = 0.1,
        random_seed: Optional[int] = None,
    ):
        """Initialize the genetic algorithm."""
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
        """Initialize all data structures with proper typing."""
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
        """Get numeric columns with caching."""
        return [
            col
            for col in self.df.select_dtypes(include=np.number).columns
            if col != self.target
        ]

    @lru_cache(maxsize=None)
    def _get_categorical_cols(self) -> Sequence[str]:
        """Get categorical columns with caching."""
        return [
            col
            for col in self.df.select_dtypes(
                include=["object", "category"]
            ).columns
            if col != self.target
        ]

    def _init_population(self) -> MutableSequence[dict]:
        """Initialize population of rules."""
        population: MutableSequence[dict] = []

        create_rule = partial(
            self._create_rule, min_user_conds=2, max_conditions=5
        )

        for _ in range(self.pop_size):
            population.append(create_rule())

        return population

    def _create_rule(
        self, min_user_conds: int = 2, max_conditions: int = 5
    ) -> dict:
        """Create a rule with conditions."""
        conditions: MutableSequence[tuple] = []

        # Add required user conditions
        user_conds = self.rng.choice(
            self.user_cols, size=min_user_conds, replace=False
        )
        conditions.extend(self._create_condition(col) for col in user_conds)

        # Add optional additional conditions
        other_cols = [
            c
            for c in self.df.columns
            if c not in self.user_cols + [self.target]
        ]
        if other_cols:
            extra = min(max_conditions - min_user_conds, len(other_cols))
            conditions.extend(
                self._create_condition(col)
                for col in self.rng.choice(
                    other_cols, size=extra, replace=False
                )
            )

        return {"conditions": conditions, "target": (self.target, "high")}

    def _create_condition(
        self, col: str
    ) -> tuple[str, str, Union[float, str]]:
        """Create a condition for a column."""
        if col in self._numeric_cols:
            op = self.rng.choice(["<", ">"])
            value = float(self.rng.choice(self._percentiles[col]))
            return (col, op, value)
        else:
            # Obtener los valores únicos temporalmente
            unique_vals = [
                set(self.df[col].dropna())
            ]  # No modificamos el DataFrame original
            if len(unique_vals) == 0:
                logger.warning(f"No unique values found for column '{col}'")
                return (col, "==", "UNKNOWN")  # o bien omitir esta condición
            return (col, "==", str(self.rng.choice(unique_vals)))

    def fitness(self, rule: dict) -> float:
        """Calculate rule fitness (support * confidence)."""
        support = self._vectorized_support(rule["conditions"])
        confidence = self._vectorized_confidence(rule)
        return support * confidence

    def _vectorized_support(self, conditions: Sequence[tuple]) -> np.ndarray:
        mask = np.ones(
            len(self.df), dtype=bool
        )  # Inicializar la máscara booleana con 'True' para todas las filas
        for col, op, value in conditions:
            col_data = self.df[col].values  # Obtener los valores de la columna

            # Aplicar la condición en función del operador
            if op == "<":
                mask &= col_data < value
            elif op == ">":
                mask &= col_data > value
            elif op == "==":
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    mask &= col_data == float(value)
                else:
                    mask &= self.df[col].astype(str).values == str(value)
            else:
                raise ValueError(f"Operador no soportado: {op}")

        # Asegurarse de que la máscara tiene la longitud correcta
        assert len(mask) == len(
            self.df
        ), f"Longitud de la máscara no coincide con el DataFrame: {len(mask)} != {len(self.df)}"

        return mask  # Devolver la máscara booleana

    def _vectorized_confidence(self, rule: dict) -> float:
        # La función _vectorized_support debe retornar un array booleano de las filas que cumplen las condiciones
        conditions_mask = self._vectorized_support(rule["conditions"])

        # Verificar que conditions_mask sea un array booleano
        if np.isscalar(conditions_mask):
            conditions_mask = np.array([conditions_mask])

        if (
            np.sum(conditions_mask) == 0
        ):  # Si no hay filas que cumplan las condiciones, la confianza es 0
            return 0

        # Asegurarse de que target_mask sea un array, no un valor escalar
        target_mask = self.df[self.target].values == rule["target"][1]

        if np.isscalar(target_mask):
            target_mask = np.array([target_mask])

        # Asegurarse de que ambas máscaras tengan la misma longitud
        if len(conditions_mask) != len(target_mask):
            raise ValueError(
                f"Las longitudes de las máscaras no coinciden: {len(conditions_mask)} != {len(target_mask)}"
            )

        # Aplicamos la máscara booleana conditions_mask a target_mask
        return np.mean(target_mask[conditions_mask])

    def calculate_support(self, rule: dict) -> float:
        """Calculate rule support (coverage of dataset)."""
        return self._vectorized_support(rule["conditions"])

    def calculate_confidence(self, rule: dict) -> float:
        """Calculate rule confidence (accuracy when conditions met)."""
        return self._vectorized_confidence(rule)

    def format_rule(self, rule: dict) -> str:
        """Format rule for human-readable display."""
        conditions = []
        for col, op, value in rule["conditions"]:
            readable_op = self.OPERATOR_MAP.get(op, op)
            if col in self._numeric_cols:
                conditions.append(f"{col} {readable_op} {value:.2f}")
            else:
                conditions.append(f"{col} {readable_op} '{value}'")

        target_col, target_val = rule["target"]
        return (
            f"IF {' AND '.join(conditions)} THEN {target_col} = {target_val}"
        )

    def mutate(self, rule: dict) -> dict:
        """Apply mutation to a rule."""
        # TODO MUTATE
        pass

    def crossover(self, parent1: dict, parent2: dict) -> tuple[dict, dict]:
        """Perform crossover between two parent rules."""
        min_len = min(len(parent1["conditions"]), len(parent2["conditions"]))
        if min_len < 2:
            return parent1.copy(), parent2.copy()

        split = self.rng.integers(1, min_len)

        child1 = {
            "conditions": parent1["conditions"][:split]
            + parent2["conditions"][split:],
            "target": parent1["target"],
        }
        child2 = {
            "conditions": parent2["conditions"][:split]
            + parent1["conditions"][split:],
            "target": parent2["target"],
        }

        # Ensure minimum user conditions
        validator = partial(
            self._validate_rule_conditions, min_user_conds=2, max_conditions=5
        )
        validator(child1)
        validator(child2)

        return child1, child2

    def _validate_rule_conditions(
        self, rule: dict, min_user_conds: int, max_conditions: int
    ) -> None:
        """Ensure rule meets condition requirements."""
        user_conds = sum(
            1 for cond in rule["conditions"] if cond[0] in self.user_cols
        )
        while (
            user_conds < min_user_conds
            and len(rule["conditions"]) < max_conditions
        ):
            col = self.rng.choice(self.user_cols)
            rule["conditions"].append(self._create_condition(col))
            user_conds += 1

    def evolve(self) -> dict:
        """Execute the evolution process."""
        for generation in range(self.generations):
            # Selection
            """parents = self._select_parents()

            # Reproduction
            new_population = self._create_new_generation(parents)"""

            """# Elitism
            # TODO CORRECTO?
            if generation > 0:
                new_population[0] = self._get_best_individual()

            self.population = new_population"""
            self._update_tracking(generation)

        return self._compile_results()

    def _select_parents(self) -> Sequence[dict]:
        """Select parents using tournament selection."""
        # TODO Implement select_parents
        pass

    def _create_new_generation(
        self, parents: Sequence[dict]
    ) -> MutableSequence[dict]:
        """Create new generation through crossover and mutation."""
        new_population: MutableSequence[dict] = []
        for i in range(0, self.pop_size, 2):
            if i + 1 < self.pop_size:
                child1, child2 = self.crossover(parents[i], parents[i + 1])
                new_population.extend(
                    [self.mutate(child1), self.mutate(child2)]
                )
            else:
                new_population.append(self.mutate(parents[i]))
        return new_population

    def _get_best_individual(self) -> dict:
        """Get best individual from current population."""
        fitness_scores = np.array(
            [self.fitness(rule) for rule in self.population]
        )
        return self.population[np.argmax(fitness_scores)]

    def _update_tracking(self, generation: int) -> None:
        """Update tracking metrics."""
        current_fitness = np.array(
            [self.fitness(rule) for rule in self.population]
        )
        self.best_fitness_history.append(np.max(current_fitness))

        logger.info(
            f"Generation {generation}: "
            f"Best Fitness={np.max(current_fitness):.4f}, "
            f"Avg Fitness={np.mean(current_fitness):.4f}"
        )

    def _compile_results(self) -> dict:
        """Compile final results."""
        best_idx = np.argmax(self.best_fitness_history)
        best_rule = self.population[best_idx]
        best_support = self.calculate_support(best_rule)
        best_confidence = self.calculate_confidence(best_rule)
        return {
            "best_rule": best_rule,
            "best_fitness": self.best_fitness_history[best_idx],
            "best_support": best_support,
            "best_confidence": best_confidence,
            "generations": self.generations,
        }
