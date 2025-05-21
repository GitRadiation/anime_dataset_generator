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