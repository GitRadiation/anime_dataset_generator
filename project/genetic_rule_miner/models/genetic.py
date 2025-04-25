"""Genetic algorithm implementation with collections.abc and NumPy."""

import itertools
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
        self.df = df.copy()
        self.target = target
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.user_cols = [col for col in user_cols if col in df.columns]
        self.rng = np.random.default_rng(random_seed)

        self.population: MutableSequence[dict] = []
        self.best_fitness_history: MutableSequence[float] = []

        self._initialize_data_structures()

    def _initialize_data_structures(self) -> None:
        self._numeric_cols = self._get_numeric_cols()
        self._categorical_cols = self._get_categorical_cols()

        self._percentiles: Mapping[str, np.ndarray] = {
            col: np.percentile(self.df[col].dropna().values, np.arange(25, 76))
            for col in self._numeric_cols
        }

        self.population = self._init_population()

    @lru_cache(maxsize=None)
    def _get_numeric_cols(self) -> Sequence[str]:
        return [
            col
            for col in self.df.select_dtypes(include=np.number).columns
            if col != self.target
        ]

    @lru_cache(maxsize=None)
    def _get_categorical_cols(self) -> Sequence[str]:
        return [
            col
            for col in self.df.columns
            if self.df[col].dtype == "object"
            or isinstance(self.df[col].dropna().iloc[0], list)
        ]

    def _init_population(self) -> MutableSequence[dict]:
        create_rule = partial(
            self._create_rule, min_user_conds=2, max_conditions=5
        )
        return [create_rule() for _ in range(self.pop_size)]

    def _create_rule(
        self, min_user_conds: int = 2, max_conditions: int = 5
    ) -> dict:
        conditions = []

        user_conds = self.rng.choice(
            self.user_cols, size=min_user_conds, replace=False
        )
        conditions.extend(self._create_condition(col) for col in user_conds)

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
        if col in self._numeric_cols:
            op = self.rng.choice(["<", ">"])
            value = float(self.rng.choice(self._percentiles[col]))
            return (col, op, value)
        else:
            raw_values = self.df[col].dropna()

            if isinstance(raw_values.iloc[0], list):
                all_vals = list(set(itertools.chain.from_iterable(raw_values)))
            else:
                all_vals = list(raw_values.astype(str).unique())

            if not all_vals:
                logger.warning(f"No unique values found for column '{col}'")
                return (col, "==", "UNKNOWN")

            return (col, "==", str(self.rng.choice(all_vals)))

    def fitness(self, rule: dict) -> float:
        """Calculate rule fitness (support * confidence)."""
        support = self._vectorized_support(rule["conditions"])
        confidence = self._vectorized_confidence(rule)
        return support * confidence

    def _build_condition_mask(self, conditions: Sequence[tuple]) -> np.ndarray:
        mask = np.ones(len(self.df), dtype=bool)
        for col, op, value in conditions:
            col_data = self.df[col].values
            if op == "<":
                mask &= col_data < value
            elif op == ">":
                mask &= col_data > value
            elif op == "==":
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    mask &= col_data == float(value)
                else:
                    mask &= self.df[col].astype(str).values == str(value)
        return mask

    def _vectorized_support(self, conditions: Sequence[tuple]) -> float:
        return np.mean(self._build_condition_mask(conditions))

    def _vectorized_confidence(self, rule: dict) -> float:
        conditions_mask = self._vectorized_support(rule["conditions"])

        # Si no hay filas que cumplan las condiciones, la confianza es 0
        if np.sum(conditions_mask) == 0:
            return 0.0

        # Crear la máscara booleana para la columna objetivo
        target_mask = self.df[self.target].values == rule["target"][1]

        # Asegurarse de que las máscaras tengan la misma longitud
        if len(target_mask) != len(conditions_mask):
            raise ValueError(
                f"Longitudes no coincidentes: {len(target_mask)} vs {len(conditions_mask)}"
            )

        # Calcula la confianza como la media de las filas que cumplen las condiciones y el objetivo
        # Indexando correctamente y calculando la media de la coincidencia
        return np.mean(target_mask[conditions_mask])

    def calculate_support(self, rule: dict) -> float:
        return np.mean(self._vectorized_support(rule["conditions"]))

    def _vectorized_confidence(self, rule: dict) -> float:
        """Calcula la confianza como proporción de filas que cumplen las condiciones y también el objetivo."""
        conditions_mask = self._build_condition_mask(rule["conditions"])

        if np.sum(conditions_mask) == 0:
            return 0.0

        target_mask = self.df[self.target].values == rule["target"][1]
        return np.mean(target_mask[conditions_mask])

    def format_rule(self, rule: dict) -> str:
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
        new_rule = rule.copy()
        new_rule["conditions"] = rule["conditions"][:]

        if (
            len(new_rule["conditions"]) > 0
            and self.rng.random() < self.mutation_rate
        ):
            idx = self.rng.integers(0, len(new_rule["conditions"]))
            col = new_rule["conditions"][idx][0]
            new_rule["conditions"][idx] = self._create_condition(col)

        return new_rule

    def crossover(self, parent1: dict, parent2: dict) -> tuple[dict, dict]:
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

        validator = partial(
            self._validate_rule_conditions, min_user_conds=2, max_conditions=5
        )
        validator(child1)
        validator(child2)

        return child1, child2

    def _validate_rule_conditions(
        self, rule: dict, min_user_conds: int, max_conditions: int
    ) -> None:
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

    def _select_parents(self) -> Sequence[dict]:
        selected = []
        tournament_size = 3
        for _ in range(self.pop_size):
            tournament = self.rng.choice(
                self.population, size=tournament_size, replace=False
            )
            best = max(tournament, key=self.fitness)
            selected.append(best)
        return selected

    def _create_new_generation(
        self, parents: Sequence[dict]
    ) -> MutableSequence[dict]:
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
        fitness_scores = np.array(
            [self.fitness(rule) for rule in self.population]
        )
        return self.population[np.argmax(fitness_scores)]

    def _update_tracking(self, generation: int) -> None:
        current_fitness = np.array(
            [self.fitness(rule) for rule in self.population]
        )
        best_idx = np.argmax(current_fitness)
        best_rule = self.population[best_idx]
        best_fitness = current_fitness[best_idx]
        best_support = self._vectorized_support(best_rule["conditions"])
        best_confidence = self._vectorized_confidence(best_rule)
        formatted_rule = self.format_rule(best_rule)

        self.best_fitness_history.append(best_fitness)

        logger.info(
            f"Generation {generation}: "
            f"Best Fitness={best_fitness:.4f}, "
            f"Support={best_support:.4f}, "
            f"Confidence={best_confidence:.4f}, "
            f"Rule: {formatted_rule}"
        )

    def evolve(self) -> dict:
        for generation in range(self.generations):
            parents = self._select_parents()
            new_population = self._create_new_generation(parents)

            if generation > 0:
                new_population[0] = self._get_best_individual()

            self.population = new_population
            self._update_tracking(generation)

        return self._compile_results()

    def _compile_results(self) -> dict:
        best_idx = np.argmax(self.best_fitness_history)
        best_rule = self._get_best_individual()
        best_support = self._vectorized_support(best_rule["conditions"])
        best_confidence = self._vectorized_confidence(best_rule)

        return {
            "best_rule": best_rule,
            "formatted_rule": self.format_rule(best_rule),
            "best_fitness": self.best_fitness_history[best_idx],
            "best_support": best_support,
            "best_confidence": best_confidence,
            "generations": self.generations,
        }
