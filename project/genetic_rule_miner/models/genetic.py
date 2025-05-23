"""Genetic algorithm implementation with collections.abc and NumPy."""

import copy
import itertools
import random
import time
from collections import deque
from collections.abc import Sequence
from sys import getsizeof
from typing import Optional

import numpy as np
import pandas as pd
from cachetools import LRUCache
from joblib import Parallel, delayed

from genetic_rule_miner.utils.logging import LogManager
from genetic_rule_miner.utils.rule import Condition, Rule

logger = LogManager.get_logger(__name__)


def getsizeof_rule(item):
    return getsizeof(item)


def getsizeof_condition(item) -> int:
    """Estimar tamaño de una condición en bytes."""
    if isinstance(item, tuple):
        return sum(getsizeof(x) for x in item)
    return getsizeof(item)


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
        convergence_threshold: float = 0.001,
        max_stagnation: int = 20,
    ):
        # Optimizar DataFrame para acceso secuencial
        self.df = df.copy()
        self.target = target_column

        # Convertir columnas relevantes a arrays numpy para acceso más rápido
        self._target_values = self.df[target_column].values
        self._numeric_cols_data = {
            col: self.df[col].values for col in self._get_numeric_cols()
        }

        # Estructuras de datos optimizadas
        self.user_cols = [col for col in user_cols if col in df.columns]
        self.user_cols_set = set(self.user_cols)
        self.all_cols_set = set(df.columns) - {target_column}

        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.convergence_threshold = convergence_threshold
        self.max_stagnation = max_stagnation

        self.rng = np.random.default_rng(random_seed)

        # Configuración de caché con límite de tamaño (8MB)
        self._cache_maxsize = 8 * 1024 * 1024  # 8MB
        self._fitness_cache = LRUCache(
            maxsize=self._cache_maxsize, getsizeof=getsizeof_rule
        )
        self._condition_cache = LRUCache(
            maxsize=self._cache_maxsize, getsizeof=getsizeof_condition
        )
        self.cache_expiration = 60 * 2.5

        # Variables para seguimiento de convergencia
        self._best_fitness_history = deque(maxlen=10)
        self._stagnation_counter = 0

        self._initialize_data_structures()

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
        """Inicializa estructuras de datos optimizadas para acceso rápido."""
        self._numeric_cols = self._get_numeric_cols()
        self._categorical_cols = self._get_categorical_cols()

        # Precalcular percentiles para acceso rápido
        self._percentiles = {
            col: np.percentile(
                np.asarray(self._numeric_cols_data[col], dtype=float)[
                    ~np.isnan(
                        np.asarray(self._numeric_cols_data[col], dtype=float)
                    )
                ],
                np.arange(25, 76, 25),
                method="midpoint",
            )
            for col in self._numeric_cols
        }

        # Precalcular valores únicos para columnas categóricas
        self._unique_values = {}
        for col in self._categorical_cols:
            if pd.api.types.is_list_like(self.df[col].iloc[0]):
                self._unique_values[col] = list(
                    set(itertools.chain.from_iterable(self.df[col]))
                )
            else:
                self._unique_values[col] = (
                    self.df[col].astype(str).unique().tolist()
                )

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

    def _build_condition_mask(self, rule: Rule) -> np.ndarray:
        """Versión optimizada de la construcción de máscaras."""
        condition_masks = []

        for cond_list in rule.conditions:
            for col, (op, value) in cond_list:
                condition = (col, op, value)
                cached_entry = self._condition_cache.get(condition)

                if cached_entry is not None:
                    condition_mask = cached_entry
                else:
                    if col in self._numeric_cols_data:
                        col_data = self._numeric_cols_data[col]
                        if op == "<":
                            condition_mask = col_data < value
                        elif op == ">=":
                            condition_mask = col_data >= value
                        else:
                            raise ValueError(
                                f"Unsupported operator '{op}' for numeric column '{col}'"
                            )
                    else:
                        col_data = self.df[col].values
                        if op == "==":
                            if pd.api.types.is_numeric_dtype(self.df[col]):
                                condition_mask = col_data == float(value)
                            else:
                                condition_mask = self.df[col].astype(
                                    str
                                ) == str(value)
                        elif op == "!=":
                            if pd.api.types.is_numeric_dtype(self.df[col]):
                                condition_mask = col_data != float(value)
                            else:
                                condition_mask = self.df[col].astype(
                                    str
                                ) != str(value)
                        else:
                            raise ValueError(
                                f"Unsupported operator '{op}' for categorical column '{col}'"
                            )

                    # Almacenar en caché
                    self._condition_cache[condition] = condition_mask

                condition_masks.append(condition_mask)

        # Combinar todas las máscaras con operación AND vectorizada
        if condition_masks:
            return np.logical_and.reduce(condition_masks)
        return np.ones(len(self.df), dtype=bool)

    def _vectorized_support(self, rule: Rule) -> float:
        condition_mask = self._build_condition_mask(rule)
        return np.sum(condition_mask) / len(self.df)

    def _vectorized_confidence(self, rule: Rule) -> float:
        """Versión optimizada del cálculo de confianza."""
        conditions_mask = self._build_condition_mask(rule)
        target_mask = self._target_values == rule.target
        support_conditions = np.sum(conditions_mask)

        if support_conditions == 0:
            return 0.0

        support_conditions_and_target = np.sum(conditions_mask & target_mask)
        return support_conditions_and_target / support_conditions

    def mutate(self, rule: Rule) -> Rule:
        new_rule = copy.copy(rule)
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

    def evaluate_rules_vectorized(self, rules: list[Rule]) -> np.ndarray:
        """
        Evalúa muchas reglas simultáneamente mediante operaciones vectorizadas.
        Devuelve una máscara booleana de shape (n_rules, n_instances).
        Corrige el error de conversión a bool cuando hay NaNs y soporta columnas pandas 'string[python]'.
        """
        n = len(self.df)
        m = len(rules)
        masks = np.ones((m, n), dtype=bool)
        for i, rule in enumerate(rules):
            for col, (op, val) in rule.conditions[0] + rule.conditions[1]:
                if col in self._numeric_cols_data:
                    data = self._numeric_cols_data[col]
                    # Asegura tipo float para operaciones numéricas
                    data = np.asarray(data, dtype=float)
                    data = np.nan_to_num(data, nan=np.inf)
                else:
                    # Soporta columnas 'string[python]' y object
                    data = self.df[col]
                    # Si es pandas StringDtype, convertir a object/str
                    if pd.api.types.is_string_dtype(data):
                        data = data.astype("object")
                    data = np.asarray(data)
                    data = np.where(pd.isna(data), "__NAN__", data)
                if op == "<":
                    cond_mask = data < val
                elif op == ">=":
                    cond_mask = data >= val
                elif op == "==":
                    cond_mask = data == val
                elif op == "!=":
                    cond_mask = data != val
                else:
                    raise NotImplementedError(f"Operador no soportado: {op}")
                masks[i] &= cond_mask
        return masks

    def genotype_to_rule(
        self, geno: list[tuple[str, str, object]], target: int
    ) -> Rule:
        """
        Convierte un genotipo (lista de tuplas) en un objeto Rule.
        """
        user_conditions = [c for c in geno if c[0] in self.user_cols]
        other_conditions = [c for c in geno if c[0] not in self.user_cols]
        return Rule(
            columns=[c[0] for c in geno],
            conditions={
                "user_conditions": [
                    (col, (op, val)) for col, op, val in user_conditions
                ],
                "other_conditions": [
                    (col, (op, val)) for col, op, val in other_conditions
                ],
            },
            target=np.int64(target),
        )

    def parallel_fitness_evaluation(self, rules: list[Rule]) -> np.ndarray:
        """
        Evalúa la aptitud de una lista de reglas en paralelo usando joblib.
        """
        return np.array(
            Parallel(n_jobs=-1, backend="loky")(
                delayed(self.fitness)(rule) for rule in rules
            )
        )

    def hill_climb(self, rule: Rule, iterations: int = 5) -> Rule:
        """
        Aplica mutación guiada (búsqueda local) para refinar reglas de alto fitness.
        """
        best = rule
        best_fitness = self.fitness(rule)
        for _ in range(iterations):
            neighbor = self.mutate(copy.deepcopy(rule))
            f = self.fitness(neighbor)
            if f > best_fitness:
                best = neighbor
                best_fitness = f
        return best

    def _rule_to_genotype(self, rule: Rule) -> list[tuple[str, str, object]]:
        """
        Convierte un Rule a su genotipo (lista de tuplas (col, op, val)).
        """
        return [
            (col, op, val)
            for col, (op, val) in rule.conditions[0] + rule.conditions[1]
        ]

    def _population_to_genotypes(
        self, population: list[Rule]
    ) -> list[list[tuple[str, str, object]]]:
        return [self._rule_to_genotype(rule) for rule in population]

    def _genotypes_to_rules(
        self, genotypes: list[list[tuple[str, str, object]]], target: int
    ) -> list[Rule]:
        return [self.genotype_to_rule(geno, target) for geno in genotypes]

    def _mutate_genotype(
        self, geno: list[tuple[str, str, object]], target: int
    ) -> list[tuple[str, str, object]]:
        """
        Aplica mutación sobre el genotipo y devuelve el nuevo genotipo.
        """
        rule = self.genotype_to_rule(geno, target)
        mutated_rule = self.mutate(rule)
        return self._rule_to_genotype(mutated_rule)

    def _crossover_genotypes(self, geno1, geno2, target):
        """
        Realiza crossover entre dos genotipos y devuelve dos nuevos genotipos.
        """
        rule1 = self.genotype_to_rule(geno1, target)
        rule2 = self.genotype_to_rule(geno2, target)
        child1, child2 = self.crossover(rule1, rule2)
        return self._rule_to_genotype(child1), self._rule_to_genotype(child2)

    def _create_random_genotype(
        self, target: int
    ) -> list[tuple[str, str, object]]:
        rule = self._create_rule(target)
        return self._rule_to_genotype(rule)

    def _adaptive_mutation_rate(
        self, fitness_values: np.ndarray, base_rate: float = 0.10
    ) -> float:
        """
        Ajusta la tasa de mutación dinámicamente según la diversidad de fitness.
        """
        if len(fitness_values) == 0 or np.mean(fitness_values) == 0:
            return base_rate
        diversity = float(np.std(fitness_values)) / float(
            np.mean(fitness_values)
        )
        return min(0.8, max(0.01, float(base_rate) * (1 + diversity)))

    def evolve_per_target(
        self,
        target_id: int | np.int64,
        max_rules: int = 720,
        fitness_threshold: float = 1.0,
        confidence_threshold: float = 0.9,
    ) -> list[Rule]:
        generation = 0
        stagnation_counter = 0
        max_stagnation = 250
        best_rules_by_signature = {}
        logger.info(
            f"Starting evolution for target {target_id} with max rules {max_rules}"
        )
        # Inicializa población como genotipos
        population_geno = [
            self._create_random_genotype(int(target_id))
            for _ in range(self.pop_size)
        ]

        while (
            len(best_rules_by_signature) < max_rules
            and generation < self.generations
        ):
            found_new = False
            # Convertir genotipos a reglas solo para evaluación
            population_rules = [
                self.genotype_to_rule(geno, int(target_id))
                for geno in population_geno
            ]
            # Evaluación vectorizada
            masks = self.evaluate_rules_vectorized(population_rules)
            # Fitness: confianza (support & target)
            supports = masks.sum(axis=1) / len(self.df)
            targets = (masks & (self._target_values == target_id)).sum(axis=1)
            fitness_scores = np.where(
                supports > 0, targets / (masks.sum(axis=1) + 1e-12), 0.0
            )
            # Ajuste adaptativo de tasa de mutación
            self.mutation_rate = self._adaptive_mutation_rate(
                fitness_scores, base_rate=0.10
            )
            # Hill climbing sobre el top 10% de la población
            n_top = max(1, int(0.1 * len(population_geno)))
            top_indices = np.argsort(fitness_scores)[-n_top:]
            for idx in top_indices:
                improved_rule = self.hill_climb(
                    population_rules[idx], iterations=3
                )
                population_geno[idx] = self._rule_to_genotype(improved_rule)

            gen_fitness = []
            gen_conf = []
            for i, rule in enumerate(population_rules):
                fit = fitness_scores[i]
                conf = fit  # ya es confianza
                if (
                    abs(fit - fitness_threshold) < 1e-6
                    and conf >= confidence_threshold
                ):
                    sig = rule.cond_signature()
                    if sig not in best_rules_by_signature or len(rule) > len(
                        best_rules_by_signature[sig]
                    ):
                        best_rules_by_signature[sig] = rule
                        found_new = True
                        gen_fitness.append(fit)
                        gen_conf.append(conf)
                        if len(best_rules_by_signature) >= max_rules:
                            break
            # Log solo uno por generación
            if gen_fitness and gen_conf:
                logger.info(
                    f"[Target {target_id}] Generación {generation}: "
                    f"Fitness max={max(gen_fitness):.4f}, min={min(gen_fitness):.4f}, "
                    f"Confianza max={max(gen_conf):.4f}, min={min(gen_conf):.4f}, "
                    f"Total acumulado: {len(best_rules_by_signature)}"
                )
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

            # Selección de padres (por torneo, usando fitness vectorizado)
            tournament_size = 3
            loser_win_prob = 0.2
            selected_geno = []
            for _ in range(len(population_geno)):
                idxs = np.random.choice(
                    len(population_geno), tournament_size, replace=False
                )
                tournament = sorted(
                    idxs,
                    key=lambda i: fitness_scores[i] * supports[i],
                    reverse=True,
                )
                if self.rng.random() < loser_win_prob and len(tournament) > 1:
                    candidate = np.random.choice(tournament[1:])
                else:
                    candidate = tournament[0]
                selected_geno.append(population_geno[candidate])

            # Generación de nueva población (crossover y mutación en genotipo)
            new_population_geno = []
            seen_signatures = set()
            # Añadir reglas válidas únicas sin mutar
            for rule in best_rules_by_signature.values():
                sig = rule.cond_signature()
                if sig not in seen_signatures:
                    new_population_geno.append(self._rule_to_genotype(rule))
                    seen_signatures.add(sig)
            # Generar hijos por crossover y mutación
            i = 0
            while (
                len(new_population_geno) < self.pop_size
                and i < len(selected_geno) - 1
            ):
                child1_geno, child2_geno = self._crossover_genotypes(
                    selected_geno[i], selected_geno[i + 1], target_id
                )
                # Mutación probabilística
                if self.rng.random() < self.mutation_rate:
                    child1_geno = self._mutate_genotype(
                        child1_geno, int(target_id)
                    )
                if self.rng.random() < self.mutation_rate:
                    child2_geno = self._mutate_genotype(
                        child2_geno, int(target_id)
                    )
                # Unicidad por firma
                rule1 = self.genotype_to_rule(child1_geno, int(target_id))
                rule2 = self.genotype_to_rule(child2_geno, int(target_id))
                sig1 = rule1.cond_signature()
                sig2 = rule2.cond_signature()
                if (
                    sig1 not in seen_signatures
                    and len(new_population_geno) < self.pop_size
                ):
                    new_population_geno.append(child1_geno)
                    seen_signatures.add(sig1)
                if (
                    sig2 not in seen_signatures
                    and len(new_population_geno) < self.pop_size
                ):
                    new_population_geno.append(child2_geno)
                    seen_signatures.add(sig2)
                i += 2
            # Si falta población, rellena con nuevas reglas aleatorias
            while len(new_population_geno) < self.pop_size:
                geno = self._create_random_genotype(int(target_id))
                rule = self.genotype_to_rule(geno, int(target_id))
                sig = rule.cond_signature()
                if sig not in seen_signatures:
                    new_population_geno.append(geno)
                    seen_signatures.add(sig)
            population_geno = new_population_geno
            logger.info(
                "Generación %d para el target %d", generation, target_id
            )
            generation += 1

        self._fitness_cache.clear()
        self._condition_cache.clear()
        del self._fitness_cache
        del self._condition_cache
        # Solo reglas más específicas al final
        final_rules = [
            self.genotype_to_rule(geno, int(target_id))
            for geno in population_geno
        ]
        return self._filter_most_specific_rules(
            list(best_rules_by_signature.values()) + final_rules
        )

    def _filter_most_specific_rules(self, rules: list[Rule]) -> list[Rule]:
        """
        Filtra reglas dejando solo las más generales según el criterio de inclusión de (col, op).
        Si una regla es subconjunto de otra, se queda la más grande (la que tiene más condiciones).
        """
        filtered = []
        for rule in rules:
            is_subsumed = False
            to_remove = []
            for i, other in enumerate(filtered):
                if other.is_subset_of(rule):
                    # other es más específica o igual, no la añadimos
                    is_subsumed = True
                    break
                elif rule.is_subset_of(other):
                    # rule es más general, quitamos la más pequeña
                    to_remove.append(i)
            if not is_subsumed:
                # Eliminar reglas más pequeñas
                for idx in reversed(to_remove):
                    del filtered[idx]
                filtered.append(rule)
        return filtered

    # def _update_tracking(
    #     self, generation: int, population: Sequence[Rule]
    # ) -> None:
    #     fitness_scores = tuple(self._parallel_evaluate_population(population))
    #     best_rule = self._get_best_individual(population, fitness_scores)
    #     best_support = self._vectorized_support(best_rule)

    #     logger.info(
    #         f"Generation {generation}: Best Fitness={fitness_scores[np.argmax(fitness_scores)]:.4f}, "
    #         f"Support={best_support:.4f} "
    #         f"Rule: {(best_rule)}"
    #     )
