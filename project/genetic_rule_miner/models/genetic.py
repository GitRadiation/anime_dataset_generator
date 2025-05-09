"""Genetic algorithm implementation with collections.abc and NumPy."""

import copy
import itertools
from collections.abc import Mapping, MutableSequence, Sequence
from functools import lru_cache, partial
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from numpy.random import Generator

from genetic_rule_miner.utils.logging import LogManager

logger = LogManager.get_logger(__name__)


Condition = Tuple[str, str, Union[float, str]]


def cache_conditions(method):
    cache_name = f"_{method.__name__}_cache"

    def wrapper(self, conditions):
        key = tuple(conditions)
        if not hasattr(self, cache_name):
            setattr(self, cache_name, {})
        cache = getattr(self, cache_name)
        if key in cache:
            return cache[key]
        result = method(self, conditions)
        cache[key] = result
        return result

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
        mutation_rate: float = 0.7,
        random_seed: Optional[int] = None,
    ):
        self.df = df.copy()
        self.target = target
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self._condition_mask_cache: dict[int, np.ndarray] = {}

        self.user_cols = [col for col in user_cols if col in df.columns]

        if not self.user_cols:
            raise ValueError("No valid user columns provided.")

        if "high" not in self.df[self.target].unique():
            logger.warning(
                f"Target value 'high' not found in column '{self.target}'."
            )
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
        """
        Adds a new condition to the given rule by selecting a column not already used.
        """
        
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

        # Completar las condiciones hasta 5 si es necesario
        while len(rule["conditions"]) < max_conditions:
            # Asegurarse de que 'target' no sea considerado en las reglas; evitar target => target
            self._add_condition(
                rule,
                existing_cols,
                [col for col in self.df.columns if col != self.target],
            )

    def fitness(self, rule: dict) -> float:
        """Calculate rule fitness (support * confidence)."""
        # Soporte de la lista de condiciones asociadas a una regla
        # Se calcula como la proporción de filas que cumplen las condiciones
        # y la confianza como la proporción de filas que cumplen
        # las condiciones y también el objetivo
        
        return self._vectorized_support(
            rule["conditions"]
        ) * self._vectorized_confidence(rule)

    @cache_conditions
    def _build_condition_mask(
        self, conditions: Sequence[Condition]
    ) -> np.ndarray:
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
                    mask &= self.df[col] == float(value)
                else:
                    mask &= self.df[col].astype(str) == str(value)
        return mask

    def _vectorized_support(self, conditions: Sequence[Condition]) -> float:
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
        new_rule = copy.deepcopy(rule)
        # Crea una lista independiente para que los cambios no afecten a la regla original
        new_rule["conditions"] = rule["conditions"][:]

        if (
            len(new_rule["conditions"]) > 0
            and self.rng.random() < self.mutation_rate
        ):
            action = self.rng.choice(["replace", "add", "remove"])

            if action == "replace" and len(new_rule["conditions"]) > 0:
                # Reemplazar una condición con una nueva aleatoria
                idx = self.rng.integers(len(new_rule["conditions"]))
                col = new_rule["conditions"][idx][0]  # Columna actual
                new_rule["conditions"][idx] = self._create_condition(col)

            elif action == "add":
                # Agregar una nueva condición, seleccionando columnas no usadas
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

            elif action == "remove" and len(new_rule["conditions"]) > 1:
                # Eliminar una condición aleatoria
                idx = self.rng.integers(len(new_rule["conditions"]))
  

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
        for i in range(0, len(parents) - 1, 2):
            child1, child2 = self.crossover(parents[i], parents[i + 1])
            new_population.extend([self.mutate(child1), self.mutate(child2)])

        # Si hay un padre sobrante, se copia o mutac directamente
        if len(parents) % 2 == 1:
            new_population.append(self.mutate(parents[-1]))

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
        """Evolves the population over a number of generations with forced diversity."""
        stagnation = 0 # Estancamiento : Control de mejora del modelo
        best_fitness = -np.inf

        for generation in range(self.generations):
            # Evaluar la población para obtener los puntajes de aptitud (fitness)
            fitness_scores = self._evaluate_population()

            # Obtener el número de reglas únicas en la población
            unique_rules = len(
                {tuple(rule["conditions"]) for rule in self.population}
            )
            logger.info(
                f"Generation {generation}: {unique_rules} unique rules"
            )


            # Selección de padres para el siguiente ciclo de evolución
            parents = self._select_parents()
            new_population = self._create_new_generation(parents)

            # Preservar los mejores individuos de la generación anterior (elitismo)
            if generation > 0:
                # Top-5% élite
                num_elite = max(1, int(self.pop_size * 0.05))
                # Se ordenan los índices de los individuos según sus fitness scores en orden ascendente 
                # [::-1]: invierte ese orden para obtenerlos de mayor a menor 
                elite_indices = np.argsort(fitness_scores)[::-1][:num_elite] 
                elites = [self.population[i] for i in elite_indices]
                new_population[:num_elite] = elites

            # Actualizar la población
            self.population = new_population

            # Actualizar el seguimiento de las estadísticas
            self._update_tracking(generation)

            # Verificar si se ha alcanzado un nuevo mejor puntaje de fitness
            best_idx = np.argmax(fitness_scores)
            if fitness_scores[best_idx] > best_fitness:
                best_fitness = fitness_scores[best_idx]
                stagnation = 0
            else:
                stagnation += 1

            # Early stopping si no hay mejoras en 20 generaciones
            if stagnation >= 20:
                logger.info(
                    "Early stopping: no improvement in 20 generations."
                )
                break

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
