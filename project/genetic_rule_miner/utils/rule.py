import numpy as np


class Rule:
    """
    Representa una regla para clasificación basada en condiciones sobre columnas.
    - columns: lista de nombres de columnas de entrada (sin incluir target)
    - conditions: tuple (user_conditions, other_conditions)
        - user_conditions: list of tuples (col, (op, value)) for user columns
        - other_conditions: list of tuples (col, (op, value)) for other columns
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

    def __eq__(self, other):
        if not isinstance(other, Rule):
            return False
        return (
            self.conditions == other.conditions and self.target == other.target
        )

    def __hash__(self):
        return hash((tuple(tuple(x) for x in self.conditions), self.target))

    def __len__(self):
        """
        Devuelve el número de condiciones en la regla.
        """
        return len(self.conditions[0]) + len(self.conditions[1])

    def match(self, instance: dict) -> bool:
        """
        Verifica si una instancia (dict columna->valor) cumple la regla.
        """
        for cond_list in self.conditions:
            for col, (op, val) in cond_list:
                if col not in instance:
                    return False
                x = instance[col]
                if op == "==":
                    if x != val:
                        return False
                elif op == "!=":
                    if x == val:
                        return False
                elif op == "<":
                    if not x < val:
                        return False
                elif op == ">":
                    if not x > val:
                        return False
                elif op == "<=":
                    if not x <= val:
                        return False
                elif op == ">=":
                    if not x >= val:
                        return False
                else:
                    raise ValueError(f"Operador no soportado: {op}")
        return True
