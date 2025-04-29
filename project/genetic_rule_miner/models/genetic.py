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

        # Se calculan los percentiles para las columnas numéricas
        # para crear las condiciones de  < y >
        self._percentiles: Mapping[str, np.ndarray] = {
            col: np.percentile(self.df[col].dropna().values, np.arange(25, 76))
            for col in self._numeric_cols
        }

        self.population = self._init_population()

    def _deduplicate_conditions(self, conditions):
        return list({cond[0]: cond for cond in conditions}.values())

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
        user_conds = self.rng.choice(
            self.user_cols, size=min_user_conds, replace=False
        )
        conditions = [self._create_condition(col) for col in user_conds]
        rule = {"conditions": conditions, "target": (self.target, "high")}
        self._complete_conditions(rule, min_user_conds, max_conditions)
        return rule

    def _create_condition(
        self, col: str
    ) -> tuple[str, str, Union[float, str]]:
        if col in self._numeric_cols:
            op = self.rng.choice(["<", ">"])
            value = float(self.rng.choice(self._percentiles[col]))
            return (col, op, value)
        else:
            raw_values = self.df[col].dropna()
            all_vals = (
                list(set(itertools.chain.from_iterable(raw_values)))
                if isinstance(raw_values.iloc[0], list)
                else list(raw_values.astype(str).unique())
            )
            if not all_vals:
                logger.warning(f"No unique values found for column '{col}'")
                return (col, "==", "UNKNOWN")
            return (col, "==", str(self.rng.choice(all_vals)))

    def _add_condition(self, rule, existing_cols, column_pool):
        available_cols = [c for c in column_pool if c not in existing_cols]
        if not available_cols:
            return None
        col = self.rng.choice(available_cols)
        condition = self._create_condition(col)
        rule["conditions"].append(condition)
        existing_cols.add(col)
        return condition

    def _complete_conditions(
        self, rule: dict, min_user_conds: int, max_conditions: int
    ) -> None:
        """Ensure a rule has the required number of user conditions and total conditions."""
        existing_cols = {cond[0] for cond in rule["conditions"]}
        user_conds = [
            cond for cond in rule["conditions"] if cond[0] in self.user_cols
        ]

        while (
            len(user_conds) < min_user_conds
            and len(rule["conditions"]) < max_conditions
        ):
            condition = self._add_condition(
                rule, existing_cols, self.user_cols
            )
            if condition:
                user_conds.append(condition)

        while len(rule["conditions"]) < max_conditions:
            self._add_condition(rule, existing_cols, self.df.columns)

    def fitness(self, rule: dict) -> float:
        """Calculate rule fitness (support * confidence)."""
        return self._vectorized_support(
            rule["conditions"]
        ) * self._vectorized_confidence(rule)

    def _build_condition_mask(self, conditions: Sequence[tuple]) -> np.ndarray:
        # Crea una máscara booleana para las condiciones
        # de las filas que cumplen las condiciones
        # Inicializa la máscara como True para todas las filas
        mask = np.ones(len(self.df), dtype=bool)
        for col, op, value in conditions:
            col_data = self.df[col].values
            # Modifica la máscara booleana mask,
            # dejando en True solo aquellos elementos que
            # ya eran True y además cumplen la condición
            # Es un operador AND bit a bit
            if op == "<":
                mask &= col_data < value
            elif op == ">":
                mask &= col_data > value
            elif op == "==":
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    mask &= col_data == float(value)
                else:
                    mask &= (
                        col_data == float(value)
                        if pd.api.types.is_numeric_dtype(self.df[col])
                        else self.df[col].astype(str).values == str(value)
                    )
        return mask

    def _vectorized_support(self, conditions: Sequence[tuple]) -> float:
        """Calcula el soporte como proporción de filas que cumplen las condiciones."""
        return np.mean(self._build_condition_mask(conditions))

    def _vectorized_confidence(self, rule: dict) -> float:
        """Calcula la confianza como proporción de filas que cumplen las condiciones y también el objetivo."""
        conditions_mask = self._build_condition_mask(rule["conditions"])

        # Evitar división por cero
        # Si no hay filas que cumplan las condiciones, la confianza es 0
        if np.sum(conditions_mask) == 0:
            return 0.0
        # Calcula la confianza como proporción de filas que cumplen el objetivo
        # entre las filas que cumplen las condiciones
        target_mask = self.df[self.target].values == rule["target"][1]
        return np.mean(target_mask[conditions_mask])

    def _format_condition(
        self, col: str, op: str, value: Union[float, str]
    ) -> str:
        readable_op = self.OPERATOR_MAP.get(op, op)
        return (
            f"{col} {readable_op} {value:.2f}"
            if col in self._numeric_cols
            else f"{col} {readable_op} '{value}'"
        )

    def format_rule(self, rule: dict) -> str:
        conditions = [
            self._format_condition(col, op, value)
            for col, op, value in rule["conditions"]
        ]
        target_col, target_val = rule["target"]
        return (
            f"IF {' AND '.join(conditions)} THEN {target_col} = {target_val}"
        )

    def mutate(self, rule: dict) -> dict:
        new_rule = rule.copy()
        # Crea una lista independiente para que los cambios no afecten a la regla original
        new_rule["conditions"] = rule["conditions"][:]

        if (
            len(new_rule["conditions"]) > 0
            and self.rng.random() < self.mutation_rate
        ):
            # Selecciona un índice aleatorio para mutar
            idx = self.rng.integers(0, len(new_rule["conditions"]))
            # Nombre de la columna a mutar
            col = new_rule["conditions"][idx][0]
            # Genera una nueva condición aleatoria sobre la misma columna
            new_rule["conditions"][idx] = self._create_condition(col)
        self._complete_conditions(new_rule, min_user_conds=2, max_conditions=5)
        return new_rule

    def crossover(self, parent1: dict, parent2: dict) -> tuple[dict, dict]:
        min_len = min(len(parent1["conditions"]), len(parent2["conditions"]))
        # Asegurar que mínimo tienen dos condiciones; no tiene sentido
        # cruzar reglas con 0 o 1 condición porque no se puede cruzar nada
        if min_len < 2:
            return parent1.copy(), parent2.copy()

        # Selecciona un punto de cruce aleatorio
        # entre 1 y min_len - 1 para asegurar que ambos padres
        # tengan al menos una condición en cada hijo
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

        # Elimina condiciones duplicadas en ambos hijos
        child1["conditions"] = self._deduplicate_conditions(
            child1["conditions"]
        )
        child2["conditions"] = self._deduplicate_conditions(
            child2["conditions"]
        )

        # Asegurarse de que ambos hijos tengan al menos 2 condiciones (si no se añaden más del usuario)
        # Añadir condiciones aleatorias de otras columnas
        # para completar hasta 5 condiciones
        # Esto es importante porque si no, los hijos pueden ser inválidos
        # y no se pueden evaluar
        self._complete_conditions(child1, min_user_conds=2, max_conditions=5)
        self._complete_conditions(child2, min_user_conds=2, max_conditions=5)
        return child1, child2

    def _select_parents(self) -> Sequence[dict]:
        selected = []
        tournament_size = 3
        for _ in range(self.pop_size):
            # Selecciona un torneo de padres mediante fitness
            # Escoge los padres aleatoriamente
            # y selecciona el mejor de ellos
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
        # Recorre la lista de padres en pasos de 2
        # para crear dos hijos a partir de cada par de padres
        for i in range(0, self.pop_size, 2):

            child1, child2 = self.crossover(parents[i], parents[i + 1])
            new_population.extend([self.mutate(child1), self.mutate(child2)])

        return new_population

    def _evaluate_population(self) -> np.ndarray:
        return np.array([self.fitness(rule) for rule in self.population])

    def _get_best_individual(
        self, fitness_scores: Optional[np.ndarray] = None
    ) -> dict:
        """Optional fitness_scores parameter to avoid recalculating."""
        if fitness_scores is None:
            fitness_scores = self._evaluate_population()
        return self.population[np.argmax(fitness_scores)]

    def _update_tracking(self, generation: int) -> None:
        fitness_scores = self._evaluate_population()
        best_rule = self._get_best_individual(fitness_scores)
        best_support = self._vectorized_support(best_rule["conditions"])
        best_confidence = self._vectorized_confidence(best_rule)
        self.best_fitness_history.append(
            fitness_scores[np.argmax(fitness_scores)]
        )
        logger.info(
            f"Generation {generation}: Best Fitness={fitness_scores[np.argmax(fitness_scores)]:.4f}, "
            f"Support={best_support:.4f}, Confidence={best_confidence:.4f}, "
            f"Rule: {self.format_rule(best_rule)}"
        )

    def evolve(self) -> dict:
        """Evolves the population over a number of generations."""

        # La población inicial se genera en el constructor
        for generation in range(self.generations):
            parents = self._select_parents()
            new_population = self._create_new_generation(parents)

            # Preservar el mejor individuo de la generación anterior
            if generation > 0:
                new_population[0] = self._get_best_individual()

            self.population = new_population
            self._update_tracking(generation)

        return self._compile_results()

    def _compile_results(self) -> dict:
        best_rule = self._get_best_individual()
        return {
            "best_rule": best_rule,
            "formatted_rule": self.format_rule(best_rule),
            "best_fitness": self.best_fitness_history[-1],
            "best_support": self._vectorized_support(best_rule["conditions"]),
            "best_confidence": self._vectorized_confidence(best_rule),
        }
