"""Custom decorators using the existing LogConfig."""

from functools import wraps
from typing import Any, Callable

import pandas as pd


def validate_dataframe(*required_columns: str) -> Callable:
    """Decorator to validate DataFrame columns."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(df: pd.DataFrame, *args, **kwargs) -> Any:
            missing = [col for col in required_columns if col not in df.columns]
            if missing:
                raise ValueError(f"Missing required columns: {missing}")
            return func(df, *args, **kwargs)
        return wrapper
    return decorator
