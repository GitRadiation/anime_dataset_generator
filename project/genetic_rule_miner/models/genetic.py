"""Genetic algorithm implementation with collections.abc and NumPy."""

import copy
import itertools
import random
import time
from collections.abc import MutableSequence, Sequence
from functools import wraps
from typing import Optional

import numpy as np
import pandas as pd
from cachetools import LRUCache

from genetic_rule_miner.utils.logging import LogManager
from genetic_rule_miner.utils.rule import Condition, Rule

logger = LogManager.get_logger(__name__)


# El decorador para cachear las condiciones
def make_hashable(obj):
    if isinstance(obj, (list, tuple)):
        return tuple(make_hashable(x) for x in obj)
    elif isinstance(obj, dict):
        return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
    elif isinstance(obj, np.ndarray):
        return obj.tobytes()  # Más eficiente para arrays
    return obj


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
    def __init__(
        self,
        df: pd.DataFrame,
        target_column: str,
        user_cols: Sequence[str],
        pop_size: int = 50,
        generations: int = 100,
        mutation_rate: float = 0.10,
        random_seed: Optional[int] = None,
    ):
        # Optimizar DataFrame primero
        df = self._optimize_dataframe(df.drop(columns=["rating"]))

        self.target = target_column
        self.df = df
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.user_cols = [col for col in user_cols if col in df.columns]

        self.rng = np.random.default_rng(random_seed)

        # Cachés con límite de tamaño
        self._fitness_cache = LRUCache(maxsize=1000)  # Limitar a 1000 entradas

        self._condition_cache = LRUCache(
            maxsize=5000
        )  # Limitar a 5000 entradas
        self.cache_expiration = 60 * 2.5

        self._initialize_data_structures()

    def _optimize_dataframe(self, df):
        for col in df.select_dtypes(include=["int64"]):
            df[col] = pd.to_numeric(df[col], downcast="integer")
        for col in df.select_dtypes(include=["float64"]):
            df[col] = pd.to_numeric(df[col], downcast="float")
        return df

    def _check_cache_expiration(self, cache_dict, key):
        """Verifica si una entrada del caché ha expirado y la elimina si es así."""
        if key in cache_dict:
            value, timestamp = cache_dict[key]
            if time.time() - timestamp > self.cache_expiration:
                del cache_dict[key]
                return None
            return value
        return None

    def _initialize_data_structures(self):
        self._numeric_cols = self._get_numeric_cols()
        self._categorical_cols = self._get_categorical_cols()

        # Usar float32 para percentiles
        self._percentiles = {
            col: np.percentile(
                self.df[col].dropna().to_numpy(dtype=np.float32),
                np.arange(25, 76, 25),
                method="midpoint",
            )
            for col in self._numeric_cols
        }

    def _deduplicate_conditions(self, user_conditions, other_conditions):
        """Ensure conditions are unique within each group, keeping the last occurrence."""

        def dedup(conds):
            seen = {}
            for col, cond in reversed(conds):
                seen[col] = (col, cond)
            return list(seen.values())

        return dedup(user_conditions), dedup(other_conditions)

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

    def _create_rule(
        self,
        target_id: int | np.int64,
        min_user_conds: int = 1,
        max_user_conds: int = 5,
        max_conditions: int = 10,
    ) -> Rule:
        """
        Crea una instancia de Rule con condiciones aleatorias para un target específico.
        """
        num_user_conds = self.rng.integers(min_user_conds, max_user_conds + 1)
        user_cols = self.rng.choice(
            self.user_cols, size=num_user_conds, replace=False
        )
        user_conditions = [
            (col, (cond["operator"], cond["value"]))
            for col in user_cols
            for cond in [self._create_condition_tuple(col)]
        ]

        # Asegura al menos una condición de otra columna
        other_cols = [
            col
            for col in self.df.columns
            if col not in self.user_cols and col != self.target
        ]
        other_conditions = []
        if other_cols:
            other_col = self.rng.choice(other_cols)
            cond = self._create_condition_tuple(other_col)
            other_conditions.append(
                (other_col, (cond["operator"], cond["value"]))
            )

        # Deduplicar
        user_conditions, other_conditions = self._deduplicate_conditions(
            user_conditions, other_conditions
        )

        rule = Rule(
            columns=[col for col, _ in user_conditions + other_conditions],
            conditions={
                "user_conditions": user_conditions,
                "other_conditions": other_conditions,
            },
            target=np.int64(target_id),
        )

        self._complete_conditions(rule, max_conditions)
        return rule

    def _create_condition_tuple(self, col: str) -> Condition:
        if col in self._numeric_cols:
            op = self.rng.choice(["<", ">="])
            value = round(float(self.rng.choice(self._percentiles[col])), 2)
            return Condition(column=col, operator=op, value=value)
        else:
            raw_values = self.df[col].dropna()
            all_vals = (
                list(set(itertools.chain.from_iterable(raw_values)))
                if isinstance(raw_values.iloc[0], list)
                else list(raw_values.astype(str).unique())
            )
            if not all_vals:
                logger.warning(f"No unique values found for column '{col}'")
                return Condition(column=col, operator="==", value="UNKNOWN")
            return Condition(
                column=col, operator="==", value=str(self.rng.choice(all_vals))
            )

    def _add_condition(
        self, rule: Rule, existing_cols, column_pool, user=False
    ):
        available_cols = [
            c
            for c in column_pool
            if c not in existing_cols and c != self.target
        ]
        if not available_cols:
            return None
        col = self.rng.choice(available_cols)
        cond = self._create_condition_tuple(col)
        if user:
            rule.conditions[0].append((col, (cond["operator"], cond["value"])))
        else:
            rule.conditions[1].append((col, (cond["operator"], cond["value"])))
        # Deduplicar
        rule.conditions = self._deduplicate_conditions(
            rule.conditions[0], rule.conditions[1]
        )
        return cond

    def _complete_conditions(self, rule: Rule, max_conditions: int) -> None:
        existing_cols = set(
            [col for col, _ in rule.conditions[0] + rule.conditions[1]]
        )
        all_user_cols = [col for col in self.user_cols if col != self.target]
        all_other_cols = [
            col
            for col in self.df.columns
            if col not in self.user_cols and col != self.target
        ]
        while (
            len(rule.conditions[0]) + len(rule.conditions[1]) < max_conditions
        ):
            if self.rng.random() < 0.5 and all_user_cols:
                cond = self._add_condition(
                    rule, existing_cols, all_user_cols, user=True
                )
            else:
                cond = self._add_condition(
                    rule, existing_cols, all_other_cols, user=False
                )
            if not cond:
                break
            existing_cols = set(
                [col for col, _ in rule.conditions[0] + rule.conditions[1]]
            )

    def fitness(self, rule: Rule) -> float:
        rule_key = hash(rule)
        cached_value = self._check_cache_expiration(
            self._fitness_cache, rule_key
        )
        if cached_value is not None:
            return cached_value

        fitness_value = self._vectorized_confidence(rule)
        self._fitness_cache[rule_key] = (fitness_value, time.time())
        return fitness_value

    @cache_conditions
    def _build_condition_mask(self, rule: Rule) -> np.ndarray:
        mask = np.ones(len(self.df), dtype=bool)
        for cond_list in rule.conditions:
            for col, (op, value) in cond_list:
                condition = (col, op, value)
                cached_mask = self._check_cache_expiration(
                    self._condition_cache, condition
                )
                if cached_mask is not None:
                    condition_mask = cached_mask
                else:
                    col_data = self.df[col].values
                    if op == "<":
                        condition_mask = col_data < value
                    elif op == ">=":
                        condition_mask = col_data >= value
                    elif op == "==":
                        if pd.api.types.is_numeric_dtype(self.df[col]):
                            condition_mask = self.df[col] == float(value)
                        else:
                            condition_mask = self.df[col].astype(str) == str(
                                value
                            )
                    elif op == "!=":
                        if pd.api.types.is_numeric_dtype(self.df[col]):
                            condition_mask = self.df[col] != float(value)
                        else:
                            condition_mask = self.df[col].astype(str) != str(
                                value
                            )
                    else:
                        raise ValueError(f"Unsupported operator: {op}")

                    self._condition_cache[condition] = (
                        condition_mask,
                        time.time(),
                    )
                mask &= condition_mask
        return mask

    def _vectorized_support(self, rule: Rule) -> float:
        condition_mask = self._build_condition_mask(rule)
        return np.sum(condition_mask) / len(self.df)

    def _vectorized_confidence(self, rule: Rule) -> float:
        conditions_mask = self._build_condition_mask(rule)
        target_mask = self.df[self.target].values == rule.target
        support_conditions = np.sum(conditions_mask)
        if support_conditions == 0:
            return 0.0
        support_conditions_and_target = np.sum(conditions_mask & target_mask)
        return support_conditions_and_target / support_conditions

    def mutate(self, rule: Rule) -> Rule:
        new_rule = copy.copy(rule)
        new_rule.conditions = copy.deepcopy(rule.conditions)
        target = new_rule.target
        total_conds = len(new_rule.conditions[0]) + len(new_rule.conditions[1])
        if total_conds > 0 and self.rng.random() < self.mutation_rate:
            action = self.rng.choice(
                ["replace", "add", "remove", "regenerate"]
            )
            if action == "replace":
                # Pick from user or other
                if total_conds == 0:
                    return new_rule
                cond_type = (
                    0
                    if (
                        self.rng.random() < 0.5
                        and len(new_rule.conditions[0]) > 0
                    )
                    else 1
                )
                if len(new_rule.conditions[cond_type]) == 0:
                    cond_type = 1 - cond_type
                idx = self.rng.integers(len(new_rule.conditions[cond_type]))
                col, _ = new_rule.conditions[cond_type][idx]
                cond = self._create_condition_tuple(col)
                new_rule.conditions[cond_type][idx] = (
                    col,
                    (cond["operator"], cond["value"]),
                )
            elif action == "add":
                used_cols = set(
                    [
                        col
                        for col, _ in new_rule.conditions[0]
                        + new_rule.conditions[1]
                    ]
                )
                unused_user_cols = [
                    col
                    for col in self.user_cols
                    if col not in used_cols and col != self.target
                ]
                unused_other_cols = [
                    col
                    for col in self.df.columns
                    if col not in used_cols
                    and col not in self.user_cols
                    and col != self.target
                ]
                if self.rng.random() < 0.5 and unused_user_cols:
                    new_col = self.rng.choice(unused_user_cols)
                    cond = self._create_condition_tuple(new_col)
                    new_rule.conditions[0].append(
                        (new_col, (cond["operator"], cond["value"]))
                    )
                elif unused_other_cols:
                    new_col = self.rng.choice(unused_other_cols)
                    cond = self._create_condition_tuple(new_col)
                    new_rule.conditions[1].append(
                        (new_col, (cond["operator"], cond["value"]))
                    )
            elif action == "remove" and total_conds > 2:
                # Remove from user or other
                cond_type = (
                    0
                    if (
                        self.rng.random() < 0.5
                        and len(new_rule.conditions[0]) > 0
                    )
                    else 1
                )
                if len(new_rule.conditions[cond_type]) > 0:
                    idx = self.rng.integers(
                        len(new_rule.conditions[cond_type])
                    )
                    del new_rule.conditions[cond_type][idx]
                else:
                    new_rule = self._create_rule(target)
            elif action == "regenerate":
                new_rule = self._create_rule(target)
            # Deduplicar
            new_rule.conditions = self._deduplicate_conditions(
                new_rule.conditions[0], new_rule.conditions[1]
            )
            new_rule = self._ensure_min_user_and_other_conditions(new_rule)
            return new_rule

        # Deduplicar
        new_rule.conditions = self._deduplicate_conditions(
            new_rule.conditions[0], new_rule.conditions[1]
        )
        new_rule = self._ensure_min_user_and_other_conditions(new_rule)
        return new_rule

    def crossover(self, parent1: Rule, parent2: Rule) -> tuple[Rule, Rule]:
        # Mezcla las condiciones de usuario y de otras columnas con splits independientes

        def crossover_split(list1, list2):
            min_len = min(len(list1), len(list2))
            if min_len == 0:
                return copy.deepcopy(list1), copy.deepcopy(list2)
            split = self.rng.integers(1, min_len + 1)
            child1 = list1[:split] + list2[split:]
            child2 = list2[:split] + list1[split:]
            return child1, child2

        # Split independiente para user y otras columnas
        child1_user, child2_user = crossover_split(
            parent1.conditions[0], parent2.conditions[0]
        )
        child1_other, child2_other = crossover_split(
            parent1.conditions[1], parent2.conditions[1]
        )

        # Deduplicar
        child1_user, child1_other = self._deduplicate_conditions(
            child1_user, child1_other
        )
        child2_user, child2_other = self._deduplicate_conditions(
            child2_user, child2_other
        )

        child1 = Rule(
            columns=[col for col, _ in child1_user + child1_other],
            conditions={
                "user_conditions": child1_user,
                "other_conditions": child1_other,
            },
            target=parent1.target,
        )
        child2 = Rule(
            columns=[col for col, _ in child2_user + child2_other],
            conditions={
                "user_conditions": child2_user,
                "other_conditions": child2_other,
            },
            target=parent2.target,
        )
        self._complete_conditions(child1, max_conditions=10)
        self._complete_conditions(child2, max_conditions=10)
        child1 = self._ensure_min_user_and_other_conditions(child1)
        child2 = self._ensure_min_user_and_other_conditions(child2)
        return child1, child2

    def _ensure_min_user_and_other_conditions(self, rule: Rule) -> Rule:
        user_present = len(rule.conditions[0]) > 0
        other_present = len(rule.conditions[1]) > 0
        if not user_present:
            user_col = self.rng.choice(self.user_cols)
            cond = self._create_condition_tuple(user_col)
            rule.conditions[0].append(
                (user_col, (cond["operator"], cond["value"]))
            )
        if not other_present:
            other_cols = [
                col
                for col in self.df.columns
                if col not in self.user_cols and col != self.target
            ]
            if other_cols:
                other_col = self.rng.choice(other_cols)
                cond = self._create_condition_tuple(other_col)
                rule.conditions[1].append(
                    (other_col, (cond["operator"], cond["value"]))
                )
        # Deduplicar
        rule.conditions = self._deduplicate_conditions(
            rule.conditions[0], rule.conditions[1]
        )
        return rule

    def _select_parents(self, population: Sequence[Rule]) -> Sequence[Rule]:
        selected = []
        tournament_size = 3
        loser_win_prob = 0.2

        for _ in range(len(population)):
            tournament = random.sample(list(population), tournament_size)
            tournament = sorted(
                tournament,
                key=lambda rule: self.fitness(rule)
                * self._vectorized_support(rule),
                reverse=True,
            )
            if self.rng.random() < loser_win_prob and len(tournament) > 1:
                candidate = random.choice(tournament[1:])
            else:
                candidate = tournament[0]
            selected.append(candidate)
        return selected

    def _evaluate_population(self, population: Sequence[Rule]) -> np.ndarray:
        fitness_scores = []
        for rule in population:
            rule_key = hash(rule)
            if rule_key not in self._fitness_cache:
                self._fitness_cache[rule_key] = (
                    self._vectorized_confidence(rule),
                    time.time(),
                )
            fitness_scores.append(self._fitness_cache[rule_key][0])
        return np.array(fitness_scores)

    def _get_best_individual(
        self,
        population: Sequence[Rule],
        fitness_scores: Optional[tuple] = None,
    ) -> Rule:
        if fitness_scores is None:
            fitness_scores = tuple(self._evaluate_population(population))
        best_index = None
        best_score = -1
        for i, rule in enumerate(population):
            support = self._vectorized_support(rule)
            score = support * fitness_scores[i]
            if score > best_score:
                best_score = score
                best_index = i
        if best_index is not None:
            return population[best_index]
        else:
            raise ValueError("No best individual found in the population.")

    def _create_new_generation(
        self,
        parents: Sequence[Rule],
        valid_rules: Sequence[Rule] = (),
    ) -> MutableSequence[Rule]:
        new_population: MutableSequence[Rule] = []
        seen_rules = set()

        # 1. Añadir reglas válidas únicas sin mutar
        for rule in valid_rules:
            if rule not in seen_rules:
                new_population.append(rule)
                seen_rules.add(rule)

        # 2. Generar hijos por crossover y mutación hasta llenar población
        i = 0
        while len(new_population) < self.pop_size and i < len(parents) - 1:
            child1, child2 = self.crossover(parents[i], parents[i + 1])
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)

            if (
                child1 not in seen_rules
                and len(new_population) < self.pop_size
            ):
                new_population.append(child1)
                seen_rules.add(child1)
            if (
                child2 not in seen_rules
                and len(new_population) < self.pop_size
            ):
                new_population.append(child2)
                seen_rules.add(child2)
            i += 2

        # 3. Si hay padres sin usar y aún falta población, mutar y agregar
        if len(new_population) < self.pop_size and i == len(parents) - 1:
            last_parent = self.mutate(parents[-1])
            if last_parent not in seen_rules:
                new_population.append(last_parent)

        # 4. Si aún falta población, rellena con nuevas reglas aleatorias
        while len(new_population) < self.pop_size:
            new_rule = self._create_rule(parents[0].target)
            if new_rule not in seen_rules:
                new_population.append(new_rule)
                seen_rules.add(new_rule)

        return new_population

    def _reset_population(
        self,
        population: Sequence[Rule],
        target_id: int | np.int64,
    ) -> Sequence[Rule]:

        self._fitness_cache.clear()
        self._condition_cache.clear()
        seen_rules = set()
        new_population = []
        for rule in population:
            rule_key = (
                tuple(
                    [col for col, _ in rule.conditions[0] + rule.conditions[1]]
                )
                + tuple(
                    [
                        cond
                        for _, cond in rule.conditions[0] + rule.conditions[1]
                    ]
                )
                + (rule.target,)
            )
            if self.fitness(rule) == 0 or (
                rule_key in seen_rules
                and len(rule_key) == len(next(iter(seen_rules), ()))
            ):
                new_population.append(self._create_rule(target_id))
            else:
                new_population.append(rule)
                seen_rules.add(rule_key)
        return new_population

    def evolve_per_target(
        self,
        target_id: int | np.int64,
        max_rules: int = 720,
        fitness_threshold: float = 1.0,
        confidence_threshold: float = 0.9,
    ) -> list[Rule]:
        rules_for_target = []
        generation = 0
        stagnation_counter = 0
        max_stagnation = 250
        seen_rule_hashes = set()

        # Inicializar población específica para este target
        population = [
            self._create_rule(target_id) for _ in range(self.pop_size)
        ]

        while (
            len(rules_for_target) < max_rules and generation < self.generations
        ):

            found_new = False
            for rule in population:
                fit = self.fitness(rule)
                conf = self._vectorized_confidence(rule)

                if (
                    abs(fit - fitness_threshold) < 1e-6
                    and conf >= confidence_threshold
                ):
                    rule_hash = hash(rule)
                    if rule_hash not in seen_rule_hashes:
                        rules_for_target.append(rule)
                        seen_rule_hashes.add(rule_hash)
                        found_new = True
                        logger.info(
                            f"[Target {target_id}] Regla válida encontrada en generación {generation}: "
                            f"Fitness = {fit:.4f}, Confianza = {conf:.4f}, Total acumulado: {len(rules_for_target)}"
                        )

                        if len(rules_for_target) >= max_rules:
                            break

            # Control de estancamiento
            if found_new:
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            if stagnation_counter >= max_stagnation:
                logger.info(
                    f"[Target {target_id}] Estancamiento detectado. Terminando búsqueda."
                )
                break

            parents = self._select_parents(population)
            population = self._create_new_generation(parents, rules_for_target)
            population = self._reset_population(population, target_id)
            generation += 1

        return rules_for_target

    def _update_tracking(
        self, generation: int, population: Sequence[Rule]
    ) -> None:
        fitness_scores = tuple(self._evaluate_population(population))
        best_rule = self._get_best_individual(population, fitness_scores)
        best_support = self._vectorized_support(best_rule)

        logger.info(
            f"Generation {generation}: Best Fitness={fitness_scores[np.argmax(fitness_scores)]:.4f}, "
            f"Support={best_support:.4f} "
            f"Rule: {(best_rule)}"
        )
