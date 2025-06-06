"""Genetic algorithm implementation with collections.abc and NumPy."""

import copy
import itertools
import random
import time
from collections import deque
from collections.abc import MutableSequence, Sequence
from sys import getsizeof
from typing import List, Optional

import numpy as np
import pandas as pd
from cachetools import LRUCache

from genetic_rule_miner.data.database import DatabaseManager
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
        db_manager: Optional[DatabaseManager] = None,
        pop_size: int = 512,
        generations: int = 720,
        mutation_rate: float = 0.10,
        random_seed: Optional[int] = None,
        max_stagnation: int = 100,
    ):
        # Optimizar DataFrame para acceso secuencial
        df = self._optimize_dataframe(df.drop(columns=["rating"]))
        self.df = df.copy()
        self.target = target_column
        self.db_manager = db_manager
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

    def _optimize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimiza el DataFrame para acceso secuencial y reduce fragmentación de memoria."""
        # Ordenar columnas por tipo para mejorar localidad
        cols_ordered = [
            col for col in df.columns if df[col].dtype.kind in "biufc"
        ] + [  # Numéricos
            col for col in df.columns if df[col].dtype.kind in "O"
        ]  # Objetos
        df = df[cols_ordered].copy()

        # Convertir a tipos más eficientes
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
        """
        Devuelve las columnas categóricas (tipo object o listas).
        Si la columna está vacía, la omite para evitar errores de indexación.
        """
        categorical_cols = []
        for col in self.df.columns:
            try:
                # Evitar error si la columna está vacía
                non_na = self.df[col].dropna()
                if non_na.empty:
                    continue
                if self.df[col].dtype == "object" or isinstance(
                    non_na.iloc[0], list
                ):
                    categorical_cols.append(col)
            except Exception:
                continue
        return categorical_cols

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
            if raw_values.empty:
                logger.debug(f"No values found for column '{col}'")
                return Condition(column=col, operator="==", value="UNKNOWN")

            # Seleccionar aleatoriamente un valor de la columna
            candidate = self.rng.choice(raw_values)

            # Evaluar si el valor es un array (lista) o texto
            if isinstance(candidate, list):
                # Elegir aleatoriamente un elemento del array
                value = str(self.rng.choice(candidate))
            elif (
                isinstance(candidate, str)
                and (
                    candidate.strip().startswith("[")
                    or candidate.strip().startswith("'[")
                    or candidate.strip().startswith('"[')
                )
                and (
                    candidate.strip().endswith("]")
                    or candidate.strip().endswith("]'")
                    or candidate.strip().endswith(']"')
                )
            ):
                # Intentar parsear el string como array usando ast.literal_eval
                try:
                    import ast

                    parsed_list = ast.literal_eval(candidate)
                    if isinstance(parsed_list, list) and parsed_list:
                        value = str(self.rng.choice(parsed_list))
                    else:
                        value = candidate  # fallback si el parseo falla o la lista está vacía
                except Exception as e:
                    logger.warning(
                        f"Error parsing string as array for column '{col}': {e}"
                    )
                    value = candidate  # fallback
            else:
                value = str(candidate)  # texto simple

            return Condition(
                column=col, operator=self.rng.choice(["==", "!="]), value=value
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

    def _vectorized_support(self, rule: Rule) -> float:
        """
        Calcula el soporte de una sola regla de forma vectorizada.
        """
        mask = self._build_condition_mask_single(rule)
        return np.sum(mask) / len(self.df)

    def _vectorized_confidence(self, rule: Rule) -> float:
        """
        Calcula la confianza de una sola regla de forma vectorizada.
        """
        mask = self._build_condition_mask_single(rule)
        support_count = mask.sum(dtype=np.float64)
        if support_count == 0:
            return 0.0
        target_mask = self._target_values == rule.target
        positives = np.count_nonzero(mask & target_mask)
        return float(positives) / support_count

    def _build_condition_mask_single(self, rule: Rule) -> np.ndarray:
        """
        Construye la máscara booleana para una sola regla (no batch).
        """
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
                        col_data = self.df[col]
                        if op == "==":
                            if pd.api.types.is_numeric_dtype(col_data):
                                condition_mask = col_data.values == float(
                                    value
                                )
                            else:
                                # Si la columna contiene listas/arrays
                                if col_data.apply(
                                    lambda x: isinstance(x, list)
                                ).all():
                                    condition_mask = col_data.apply(
                                        lambda x: (
                                            value in x
                                            if isinstance(x, list)
                                            else False
                                        )
                                    ).values
                                else:
                                    condition_mask = col_data.astype(
                                        str
                                    ).values == str(value)
                        elif op == "!=":
                            if pd.api.types.is_numeric_dtype(col_data):
                                condition_mask = col_data.values != float(
                                    value
                                )
                            else:
                                if col_data.apply(
                                    lambda x: isinstance(x, list)
                                ).all():
                                    condition_mask = col_data.apply(
                                        lambda x: (
                                            value not in x
                                            if isinstance(x, list)
                                            else True
                                        )
                                    ).values
                                else:
                                    condition_mask = col_data.astype(
                                        str
                                    ).values != str(value)
                        else:
                            raise ValueError(
                                f"Unsupported operator '{op}' for categorical column '{col}'"
                            )
                    self._condition_cache[condition] = condition_mask
                condition_masks.append(condition_mask)
        if condition_masks:
            return np.logical_and.reduce(condition_masks)
        return np.ones(len(self.df), dtype=bool)

    def evaluate_rules_vectorized(self, rules: list[Rule]) -> np.ndarray:
        """
        Evalúa muchas reglas simultáneamente mediante operaciones vectorizadas.
        Devuelve una máscara booleana de shape (n_rules, n_instances).
        Optimizado para columnas numéricas y categóricas.
        Usa self._condition_cache para cachear condiciones individuales.
        """
        n = len(self.df)
        m = len(rules)
        masks = np.ones((m, n), dtype=bool)

        # Precache data arrays para acelerar acceso
        col_data_cache = {}
        for rule in rules:
            for col, _ in rule.conditions[0] + rule.conditions[1]:
                if col not in col_data_cache:
                    if col in self._numeric_cols_data:
                        col_data_cache[col] = np.asarray(
                            self._numeric_cols_data[col]
                        )
                    else:
                        col_data_cache[col] = self.df[col].astype(str).values

        for i, rule in enumerate(rules):
            rule_mask = np.ones(n, dtype=bool)
            for col, (op, val) in rule.conditions[0] + rule.conditions[1]:
                condition_key = (col, op, val)
                cond_mask = self._condition_cache.get(condition_key)
                if cond_mask is None:
                    data = col_data_cache[col]
                    if op == "<":
                        cond_mask = data < val
                    elif op == ">=":
                        cond_mask = data >= val
                    elif op == "==":
                        cond_mask = data == val
                    elif op == "!=":
                        cond_mask = data != val
                    else:
                        raise NotImplementedError(
                            f"Operador no soportado: {op}"
                        )
                    self._condition_cache[condition_key] = cond_mask
                rule_mask &= cond_mask
                if not np.any(rule_mask):
                    break  # Early exit si ya es todo False
            masks[i] = rule_mask

        return masks

    def batch_vectorized_support(self, rules: list[Rule]) -> np.ndarray:
        """
        Calcula el soporte de múltiples reglas de forma vectorizada (batch).
        Devuelve un array de soporte para cada regla.
        """
        masks = self.evaluate_rules_vectorized(rules)
        supports = masks.sum(axis=1) / len(self.df)
        return supports

    def batch_vectorized_confidence(self, rules: list[Rule]) -> np.ndarray:
        """
        Calcula la confianza de múltiples reglas de forma vectorizada (batch).
        Devuelve un array de confianza para cada regla.
        """
        masks = self.evaluate_rules_vectorized(rules)
        supports = masks.sum(axis=1)
        # Limpiar soportes negativos o NaN
        supports = np.where(np.isnan(supports) | (supports <= 0), 1, supports)
        targets = np.array([rule.target for rule in rules])
        target_matrix = self._target_values[None, :] == targets[:, None]
        positives = np.logical_and(masks, target_matrix).sum(axis=1)
        confidences = positives / supports
        confidences = np.where(
            np.isnan(confidences) | (supports == 1), 0.0, confidences
        )
        return confidences

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

    def _evaluate_population_chunk(
        self, population_chunk: Sequence[Rule]
    ) -> List[float]:
        """Evalúa un chunk de la población."""
        return [self.fitness(rule) for rule in population_chunk]

    def _create_new_generation(
        self,
        parents: Sequence[Rule],
        valid_rules: Sequence[Rule] = (),
    ) -> MutableSequence[Rule]:
        new_population: MutableSequence[Rule] = []
        seen_rules = set()

        # --- ELITISMO: priorizar reglas con fitness 1 ---
        fitnesses = [self.fitness(rule) for rule in parents]
        supports = [self._vectorized_support(rule) for rule in parents]
        elite_candidates = [
            (i, fitnesses[i], supports[i]) for i in range(len(parents))
        ]
        # Primero, elites con fitness 1, ordenados por soporte descendente
        elites_fit1 = sorted(
            [e for e in elite_candidates if abs(e[1] - 1.0) < 1e-6],
            key=lambda x: x[2],
            reverse=True,
        )
        # Elitismo del 10%
        n_elite = max(1, int(0.1 * len(parents)))
        elites = elites_fit1[:n_elite]
        if len(elites) < n_elite:
            # Agregar el resto por score (sin duplicar)
            remaining = [
                e
                for e in elite_candidates
                if e not in elites
                and e[0] not in [idx for idx, _, _ in elites]
            ]
            remaining = sorted(
                remaining,
                key=lambda x: x[1] * x[2],
                reverse=True,
            )
            elites += remaining[: n_elite - len(elites)]
        elite_indices = [e[0] for e in elites]
        for i in elite_indices:
            rule = parents[i]
            if rule not in seen_rules:
                new_population.append(rule)
                seen_rules.add(rule)
        # ---------------------------------------------------------------

        # 1. Añadir reglas válidas únicas sin mutar (no duplicar elites)
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
            rule_key = hash(rule)
            if self.fitness(rule) == 0 or rule_key in seen_rules:
                new_population.append(self._create_rule(target_id))
            else:
                new_population.append(rule)
                seen_rules.add(rule_key)

        return new_population

    def _filter_most_specific_rules(self, rules: list[Rule]) -> list[Rule]:
        """
        Filtra reglas dejando solo las más específicas (más condiciones).
        Si una regla es más general (subconjunto de otra), se descarta.
        """
        filtered = []
        for rule in rules:
            is_more_general = False
            to_remove = []
            for i, other in enumerate(filtered):
                if other.is_subset_of(rule):
                    # other es más general, lo eliminamos
                    to_remove.append(i)
                elif rule.is_subset_of(other):
                    # rule es más general, no la añadimos
                    is_more_general = True
                    break
            if not is_more_general:
                for idx in reversed(to_remove):
                    del filtered[idx]
                filtered.append(rule)
        return filtered

    def evolve_per_target(
        self,
        target_id: int | np.int64,
        max_rules: int = 720,
        fitness_threshold: float = 1.0,
        support_threshold: float = 0.95,
    ) -> list[Rule]:
        generation = 0
        stagnation_counter = 0
        valid_rules = []
        max_stagnation = self.max_stagnation
        logger.info(
            f"Starting evolution for target {target_id} with max rules {max_rules}"
        )
        population = [
            self._create_rule(target_id) for _ in range(self.pop_size)
        ]

        while generation < self.generations:
            found_new = False
            # --- EVALUACIÓN BATCH ---
            fitness_arr = self.batch_vectorized_confidence(list(population))
            support_arr = self.batch_vectorized_support(list(population))
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
            valid_rules = [
                population[idx]
                for idx, (fit, sup) in enumerate(zip(fitness_arr, support_arr))
                if abs(fit - fitness_threshold) < 1e-6
                and sup >= support_threshold
            ]
            valid_rules = self._filter_most_specific_rules(valid_rules)

            max_fitness = float(np.max(fitness_arr))
            idx_max = int(np.argmax(fitness_arr))
            max_support = float(support_arr[idx_max])
            logger.info(
                f"[Target {target_id}] Generación {generation}: "
                f"Fitness max={max_fitness:.4f}, Support max={max_support:.4f}, "
                f"Reglas válidas en población: {len(valid_rules)}"
            )
            if len(valid_rules) == max_rules:
                break
            generation += 1

            parents = self._select_parents(population)
            # No se filtran reglas guardadas, solo se evoluciona la población
            population = self._create_new_generation(parents)
            population = self._reset_population(population, target_id)
        self._fitness_cache.clear()
        self._condition_cache.clear()
        del self._fitness_cache
        del self._condition_cache

        if self.db_manager is not None:
            all_existing_rules = []
            offset = 0
            page_size = 500  # ajusta según tus necesidades

            while True:
                page = self.db_manager.get_rules_by_target_value_paginated(
                    int(target_id), offset=offset, limit=page_size
                )
                if not page:
                    break  # No hay más datos
                all_existing_rules.extend(page)
                offset += page_size

            existing_conditions_set = {
                tuple(sorted(str(cond) for cond in rule.conditions))
                for rule in all_existing_rules
            }

            # Solo conservar las reglas que no están en la base de datos
            unique_rules = [
                rule
                for rule in valid_rules
                if tuple(sorted(str(cond) for cond in rule.conditions))
                not in existing_conditions_set
            ]

            logger.info(
                f"[Target {target_id}] {len(unique_rules)} nuevas reglas encontradas (no repetidas)"
            )
            return unique_rules

        return valid_rules
