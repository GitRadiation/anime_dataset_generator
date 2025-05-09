"""Genetic algorithm implementation with collections.abc and NumPy."""

import copy
import itertools
from collections.abc import Mapping, MutableSequence, Sequence
from functools import partial, wraps
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from numpy.random import Generator

from genetic_rule_miner.utils.logging import LogManager

logger = LogManager.get_logger(__name__)


Condition = Tuple[str, str, Union[float, str]]


# El decorador para cachear las condiciones
def make_hashable(obj):
    """Recursivamente convierte listas/diccionarios/sets a tuplas para el hashing."""
    if isinstance(obj, (tuple, list)):
        return tuple(make_hashable(e) for e in obj)
    elif isinstance(obj, dict):
        return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
    elif isinstance(obj, set):
        return tuple(sorted(make_hashable(e) for e in obj))
    return obj  # caso base (int, str, etc.)


def cache_conditions(method):
    cache_name = f"_{method.__name__}_cache"

    @wraps(method)
    def wrapper(self, conditions):
        key = make_hashable(conditions)
        cache = getattr(self, cache_name, None)
        if cache is None:
            cache = {}
            setattr(self, cache_name, cache)
        if key not in cache:
            cache[key] = method(self, conditions)
        return cache[key]

    return wrapper


class GeneticRuleMiner:
    OPERATOR_MAP = {"<": "less than", ">": "greater than", "==": "equals"}

    def __init__(
        self,
        df: pd.DataFrame,
        target: str,
        user_cols: Sequence[str],
        pop_size: int = 50,
        generations: int = 100,
        mutation_rate: float = 0.03,
        random_seed: Optional[int] = None,
    ):
        # Filter the DataFrame to include only rows with 'rating' == 'high'
        df = df[df["rating"] == "high"].copy()

        # Drop the 'rating' column
        df = df.drop(columns=["rating"])

        # Set the target to a randomly selected series ID
        self.target = target
        if self.target not in df.columns:
            raise ValueError(
                "The DataFrame must contain a 'series_id' column."
            )

        self.targets = df[target].unique()
        self.df = df
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self._condition_mask_cache: dict[int, np.ndarray] = {}
        self._fitness_cache: dict[tuple, float] = {}
        self._condition_cache: dict[tuple, np.ndarray] = {}

        self.user_cols = [col for col in user_cols if col in df.columns]

        if not self.user_cols:
            raise ValueError("No valid user columns provided.")

        self.rng: Generator = np.random.default_rng(random_seed)

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
        """Ensure conditions are unique within a rule, keeping the last occurrence."""
        seen = {}
        for cond in reversed(conditions):
            col = cond[0]
            if col != self.target:
                seen[col] = cond
        return list(seen.values())

    def _get_numeric_cols(self) -> Sequence[str]:
        return [
            col
            for col in self.df.select_dtypes(include=np.number).columns
            if col != self.target
        ]

    def _get_categorical_cols(self) -> Sequence[str]:
        return [
            col
            for col in self.df.columns
            if self.df[col].dtype == "object"
            or isinstance(self.df[col].dropna().iloc[0], list)
        ]

    def _init_population(self) -> MutableSequence[dict]:
        create_rule = partial(
            self._create_rule,
            min_user_conds=1,
            max_user_conds=5,
            max_conditions=10,
        )
        return [create_rule() for _ in range(self.pop_size)]

    def _create_rule(
        self,
        min_user_conds: int = 1,
        max_user_conds: int = 5,
        max_conditions: int = 10,
    ) -> dict:
        """Create a rule with at least one user condition and one other condition."""
        num_user_conds = self.rng.integers(min_user_conds, max_user_conds + 1)
        user_conds = self.rng.choice(
            self.user_cols, size=num_user_conds, replace=False
        )
        conditions = [self._create_condition(col) for col in user_conds]

        # Ensure at least one other condition
        other_cols = [
            col
            for col in self.df.columns
            if col not in self.user_cols and col != self.target
        ]

        if other_cols:
            other_condition = self._create_condition(
                self.rng.choice(other_cols)
            )
            conditions.append(other_condition)

        # Deduplicate conditions
        conditions = self._deduplicate_conditions(conditions)

        # Set a random target ID for the rule (unique for each rule)
        chosen_target = self.rng.choice(self.targets)
        rule = {
            "conditions": conditions,
            "target": (self.target, chosen_target),
        }

        self._complete_conditions(rule, max_conditions)
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
        """Add a new condition to the rule."""
        available_cols = [
            c
            for c in column_pool
            if c not in existing_cols and c != self.target
        ]

        if not available_cols:
            return None
        col = self.rng.choice(available_cols)
        condition = self._create_condition(col)
        rule["conditions"].append(condition)
        existing_cols.add(col)
        rule["conditions"] = self._deduplicate_conditions(rule["conditions"])

        return condition

    def _complete_conditions(self, rule: dict, max_conditions: int) -> None:
        """Ensure a rule has the required number of total conditions."""
        existing_cols = {cond[0] for cond in rule["conditions"]}
        all_cols = [col for col in self.df.columns if col != self.target]

        while len(rule["conditions"]) < max_conditions:
            condition = self._add_condition(rule, existing_cols, all_cols)
            if not condition:
                break
        rule["conditions"] = self._deduplicate_conditions(rule["conditions"])

    def fitness(self, rule: dict) -> float:
        """Calculate rule fitness (support * confidence)."""
        rule_key = tuple(sorted(rule["conditions"])) + (rule["target"],)
        if rule_key in self._fitness_cache:
            return self._fitness_cache[rule_key]
        fitness_value = self._vectorized_confidence(rule)
        self._fitness_cache[rule_key] = fitness_value
        return fitness_value

    @cache_conditions
    def _build_condition_mask(
        self, conditions: Sequence[Tuple[str, str, Union[float, str]]]
    ) -> np.ndarray:
        # Inicializa la máscara como True para todas las filas
        mask = np.ones(len(self.df), dtype=bool)

        for condition in conditions:
            if condition in self._condition_cache:
                condition_mask = self._condition_cache[condition]
            else:
                col, op, value = condition
                col_data = self.df[col].values

                if op == "<":
                    condition_mask = col_data < value
                elif op == ">":
                    condition_mask = col_data > value
                elif op == "==":
                    if pd.api.types.is_numeric_dtype(self.df[col]):
                        condition_mask = self.df[col] == float(value)
                    else:
                        condition_mask = self.df[col].astype(str) == str(value)
                elif op == "!=":
                    if pd.api.types.is_numeric_dtype(self.df[col]):
                        condition_mask = self.df[col] != float(value)
                    else:
                        condition_mask = self.df[col].astype(str) != str(value)
                elif op == ">=":
                    condition_mask = col_data >= value
                elif op == "<=":
                    condition_mask = col_data <= value
                else:
                    raise ValueError(f"Unsupported operator: {op}")

                # Cachea el resultado de la condición
                self._condition_cache[condition] = condition_mask

            # Combina la nueva máscara con las anteriores (AND)
            mask &= condition_mask

        return mask

    def _vectorized_support(self, conditions: Sequence[Condition]) -> float:
        """Calcula el soporte como proporción de filas que cumplen las condiciones."""
        condition_mask = self._build_condition_mask(conditions)
        return np.sum(condition_mask) / len(self.df)

    def _vectorized_confidence(self, rule: dict) -> float:
        """Calcula la confianza como proporción de filas que cumplen las condiciones y también el objetivo."""
        conditions_mask = self._build_condition_mask(rule["conditions"])
        target_mask = self.df[self.target].values == rule["target"][1]

        # Evitar división por cero
        support_conditions = np.sum(conditions_mask)
        if support_conditions == 0:
            return 0.0

        # Calcula la confianza como proporción de filas que cumplen las condiciones y el objetivo
        support_conditions_and_target = np.sum(conditions_mask & target_mask)
        return support_conditions_and_target / support_conditions

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
        """Mutate a rule by adding, removing, or replacing conditions."""
        new_rule = copy.deepcopy(rule)
        if (
            len(new_rule["conditions"]) > 0
            and self.rng.random() < self.mutation_rate
        ):
            action = self.rng.choice(["replace", "add", "remove"])
            if action == "replace":
                idx = self.rng.integers(len(new_rule["conditions"]))
                col = new_rule["conditions"][idx][0]
                new_rule["conditions"][idx] = self._create_condition(col)
            elif action == "add":
                used_cols = set(cond[0] for cond in new_rule["conditions"])
                unused_cols = [
                    col
                    for col in self.df.columns
                    if col not in used_cols and col != self.target
                ]
                if unused_cols:
                    new_col = self.rng.choice(unused_cols)
                    new_rule["conditions"].append(
                        self._create_condition(new_col)
                    )
            elif action == "remove" and len(new_rule["conditions"]) > 2:
                # Recreate the rule instead of deleting it
                new_rule = self._create_rule()
        # Nueva mutación del target
        elif self.rng.random() < 0.1:  # 10% de probabilidad
            new_target = self.rng.choice(self.targets)
            new_rule["target"] = (self.target, new_target)

        new_rule["conditions"] = self._deduplicate_conditions(
            new_rule["conditions"]
        )

        new_rule = self._ensure_min_user_and_other_conditions(new_rule)

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
        self._complete_conditions(child1, max_conditions=10)
        self._complete_conditions(child2, max_conditions=10)
        child1 = self._ensure_min_user_and_other_conditions(child1)
        child2 = self._ensure_min_user_and_other_conditions(child2)

        return child1, child2

    def _ensure_min_user_and_other_conditions(self, rule: dict) -> dict:
        user_present = any(
            cond[0] in self.user_cols for cond in rule["conditions"]
        )
        other_present = any(
            cond[0] not in self.user_cols for cond in rule["conditions"]
        )

        if not user_present:
            user_col = self.rng.choice(self.user_cols)
            rule["conditions"].append(self._create_condition(user_col))

        if not other_present:
            other_cols = [
                col
                for col in self.df.columns
                if col not in self.user_cols and col != self.target
            ]
            if other_cols:
                other_col = self.rng.choice(other_cols)
                rule["conditions"].append(self._create_condition(other_col))

        rule["conditions"] = self._deduplicate_conditions(rule["conditions"])
        return rule

    def _select_parents(self) -> Sequence[dict]:
        selected = []
        tournament_size = 3
        loser_win_prob = 0.2  # ← Probabilidad de que NO gane el más apto

        for _ in range(self.pop_size):
            tournament = self.rng.choice(
                self.population, size=tournament_size, replace=False
            )
            tournament = sorted(tournament, key=self.fitness, reverse=True)

            if self.rng.random() < loser_win_prob:
                # Elige aleatoriamente uno que no sea el mejor
                candidate = self.rng.choice(tournament[1:])
            else:
                candidate = tournament[0]

            selected.append(candidate)

        return selected

    def _create_new_generation(
        self, parents: Sequence[dict]
    ) -> MutableSequence[dict]:
        """Create a new generation of unique rules."""
        new_population: MutableSequence[dict] = []
        seen_rules = set()

        for i in range(0, len(parents) - 1, 2):
            child1, child2 = self.crossover(parents[i], parents[i + 1])
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)

            # Ensure unique rules
            child1_conditions = tuple(sorted(child1["conditions"]))
            child2_conditions = tuple(sorted(child2["conditions"]))
            if child1_conditions not in seen_rules:
                new_population.append(child1)
                seen_rules.add(child1_conditions)
            if child2_conditions not in seen_rules:
                new_population.append(child2)
                seen_rules.add(child2_conditions)

        # If there's an odd number of parents, handle the last one
        if len(parents) % 2 == 1:
            last_parent = self.mutate(parents[-1])
            last_conditions = tuple(sorted(last_parent["conditions"]))
            if last_conditions not in seen_rules:
                new_population.append(last_parent)

        return new_population

    def _evaluate_population(self) -> np.ndarray:
        fitness_scores = []
        for rule in self.population:
            rule_key = tuple(sorted(rule["conditions"])) + (rule["target"],)
            if rule_key not in self._fitness_cache:
                self._fitness_cache[rule_key] = self._vectorized_confidence(
                    rule
                )
            fitness_scores.append(self._fitness_cache[rule_key])
        return np.array(fitness_scores)

    def _get_best_individual(
        self, fitness_scores: Optional[tuple] = None
    ) -> dict:
        """Retrieve the best rule based on the highest product of support and fitness."""
        if fitness_scores is None:
            fitness_scores = tuple(
                self._evaluate_population()
            )  # Convert to tuple

        best_index = None
        best_score = -1

        for i, rule in enumerate(self.population):
            support = self._vectorized_support(rule["conditions"])
            score = support * fitness_scores[i]
            if score > best_score:
                best_score = score
                best_index = i

        return self.population[best_index]

    def _update_tracking(self, generation: int) -> None:
        fitness_scores = tuple(self._evaluate_population())  # Convert to tuple
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

    def _reset_rule(self) -> dict:
        """Reset a rule by creating a new one."""
        return self._create_rule()

    def _reset_population(self) -> None:
        """Reset rules in the population if their fitness is 0 or if they are duplicates."""
        self._fitness_cache.clear()
        self._condition_cache.clear()

        # Identificar reglas duplicadas
        seen_rules = set()
        for i, rule in enumerate(self.population):
            rule_key = tuple(sorted(rule["conditions"])) + (rule["target"],)
            if self.fitness(rule) == 0 or rule_key in seen_rules:
                # Reinicializar reglas duplicadas o con fitness 0
                self.population[i] = self._reset_rule()
            else:
                seen_rules.add(rule_key)

    def evolve(self) -> dict:
        """Evolves the population over a number of generations with forced diversity."""
        stable_high_fitness_start = None
        reference_high_fitness = None
        stable_generations = 0
        stable_threshold = 50  # ±50 reglas
        required_stable_generations = (
            5  # puedes ajustar esto si quieres más robustez
        )

        for generation in range(self.generations):
            fitness_scores = self._evaluate_population()
            high_fitness_rules, ids_set = self.get_high_fitness_rules(
                threshold=0.9
            )
            num_high_fitness = len(high_fitness_rules)
            num_unique_ids = len(ids_set)

            logger.info(
                f"Generation {generation}: {num_high_fitness} rules with fitness > 0.9, "
                f"{num_unique_ids} unique target IDs with fitness > 0.9; {ids_set}"
            )

            # Early stopping si se cubre un gran porcentaje
            threshold = 0.9
            num_above_threshold = np.sum(fitness_scores >= threshold)
            if num_above_threshold >= 0.63 * self.pop_size and len(
                ids_set
            ) >= 0.7 * len(self.targets):
                logger.info(
                    f"Early stopping: {num_above_threshold} rules ({num_above_threshold / self.pop_size:.2%}) "
                    f"have fitness >= {threshold} in generation {generation}, "
                    f"covering {len(ids_set) / len(self.targets):.2%} of target IDs."
                )
                break

            # Verificar estabilidad de reglas con fitness > 0.9
            if num_high_fitness >= 100:
                if reference_high_fitness is None:
                    reference_high_fitness = num_high_fitness
                    stable_high_fitness_start = generation
                    stable_generations = 1
                elif (
                    abs(num_high_fitness - reference_high_fitness)
                    <= stable_threshold
                ):
                    stable_generations += 1
                    if stable_generations >= required_stable_generations:
                        logger.info(
                            f"Stopping due to stable high-fitness rules: variation within ±{stable_threshold} "
                            f"for {stable_generations} generations since generation {stable_high_fitness_start}."
                        )
                        break
                else:
                    # Reset stability tracking if variation is too high
                    reference_high_fitness = num_high_fitness
                    stable_high_fitness_start = generation
                    stable_generations = 1

            # Proceso de evolución continúa
            elite_rules_by_target = self.get_elite_rules_by_target(
                threshold=0.9
            )
            parents = self._select_parents()
            new_population = self._create_new_generation(parents)

            for target_id, elite_rules in elite_rules_by_target.items():
                for elite_rule in elite_rules:
                    if all(
                        tuple(sorted(elite_rule["conditions"]))
                        != tuple(sorted(rule["conditions"]))
                        or elite_rule["target"] != rule["target"]
                        for rule in new_population
                    ):
                        new_population[
                            self.rng.integers(len(new_population))
                        ] = elite_rule

            self.population = new_population
            self._update_tracking(generation)
            self._reset_population()

    def get_elite_rules_by_target(self, threshold: float = 0.9) -> dict:
        """
        Devuelve un diccionario donde cada clave es un target_id (valor del objetivo)
        y el valor es una lista de reglas con fitness >= threshold.
        """
        elite_rules = {}
        for rule in self.population:
            fitness_val = self.fitness(rule)
            target_id = rule["target"][1]

            if fitness_val >= threshold:
                if target_id not in elite_rules:
                    elite_rules[target_id] = []
                elite_rules[target_id].append(rule)

        return elite_rules

    def get_high_fitness_rules(self, threshold: float = 0.9) -> list[dict]:
        """Retrieve all rules with fitness >= threshold."""
        high_fitness_rules = []
        ids_set = set()
        for rule in self.population:
            if self.fitness(rule) >= threshold:
                high_fitness_rules.append(rule)
                ids_set.add(rule["target"][1])
        return high_fitness_rules, ids_set
