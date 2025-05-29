from typing import TypedDict

import numpy as np


class Condition(TypedDict):
    column: str
    operator: str
    value: str | int | float


class Rule:
    """
    Representa una regla para clasificación basada en condiciones sobre columnas.
    - columns: lista de nombres de columnas de entrada (sin incluir target)
    - conditions: dict con claves "user_conditions" y "other_conditions", cada una lista de tuplas (col, (op, value))
        o lista de Condition (TypedDict) para compatibilidad.
    - target: valor objetivo al que apunta la regla
    """

    def __init__(
        self,
        columns: list[str],
        conditions: dict,
        target: np.int64,
    ):
        # columns is now just for compatibility, not used for logic
        self.columns = list(columns)

        def parse_conds(cond_list):
            parsed = []
            for cond in cond_list:
                if isinstance(cond, dict):
                    # Assume Condition TypedDict
                    parsed.append(
                        (cond["column"], (cond["operator"], cond["value"]))
                    )
                elif isinstance(cond, tuple) and isinstance(cond[1], tuple):
                    parsed.append(cond)
                else:
                    raise TypeError(f"Condición en formato inesperado: {cond}")
            return parsed

        user_conditions = parse_conds(conditions.get("user_conditions", []))
        other_conditions = parse_conds(conditions.get("other_conditions", []))
        self.conditions = (user_conditions, other_conditions)
        self.target = target

    def __repr__(self):
        user_conds = [
            f"{col} {op} {val!r}" for col, (op, val) in self.conditions[0]
        ]
        other_conds = [
            f"{col} {op} {val!r}" for col, (op, val) in self.conditions[1]
        ]
        conds = user_conds + other_conds
        return f"IF {' AND '.join(conds)} THEN target = {self.target!r}"

    def __len__(self):
        """
        Devuelve el número de condiciones en la regla.
        """
        return len(self.conditions[0]) + len(self.conditions[1])

    def _cond_key_set(self, conds):
        """
        Devuelve un frozenset de (columna, operador) para un bloque de condiciones.
        """
        return frozenset((col, op) for col, (op, _) in conds)

    def cond_signature(self):
        """
        Devuelve la firma de la regla como dos frozensets: uno para user_conditions y otro para other_conditions.
        """
        return (
            self._cond_key_set(self.conditions[0]),
            self._cond_key_set(self.conditions[1]),
            self.target,
        )

    def __eq__(self, other):
        if not isinstance(other, Rule):
            return False
        # Igualdad: mismos pares (col, op) en ambos bloques y mismo target
        return self.cond_signature() == other.cond_signature()

    def __hash__(self):
        # Hash basado en la firma de condiciones y el target
        return hash(self.cond_signature())

    def is_subset_of(self, other: "Rule") -> bool:
        """
        Devuelve True si esta regla es subconjunto de otra (en ambos bloques y mismo target).
        """
        if self.target != other.target:
            return False
        user_self, other_self, _ = self.cond_signature()
        user_other, other_other, _ = other.cond_signature()
        return user_self.issubset(user_other) and other_self.issubset(
            other_other
        )

    def is_more_specific_than(self, other: "Rule") -> bool:
        """
        Devuelve True si esta regla es más específica (tiene más condiciones) que otra,
        pero es subconjunto en pares (col, op) y mismo target.
        """
        if not self.is_subset_of(other):
            return False
        # Más específica si tiene más condiciones totales
        return len(self.conditions[0]) + len(self.conditions[1]) > len(
            other.conditions[0]
        ) + len(other.conditions[1])
