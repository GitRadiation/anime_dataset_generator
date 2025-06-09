# config.py
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class APIConfig:
    """Configuration for external API services"""

    base_url: str = "https://api.jikan.moe/v4/"
    max_retries: int = 3
    timeout: float = 10.0
    request_delay: float = 0.35
    rate_limit: int = 3

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if any(val < 0 for val in (self.timeout, self.request_delay)):
            raise ValueError("Negative values not allowed for time settings")


@dataclass
class DBConfig:
    """Configuration for database connections"""

    host: str = os.getenv("DB_HOST", "postgres")
    port: int = int(os.getenv("DB_PORT", 5432))
    database: str = os.getenv("DB_NAME", "mydatabase")
    user: str = os.getenv("DB_USER", "postgres")
    password: str = os.getenv("DB_PASS", "postgres")
    sqlite_file: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate database configuration."""
        if self.port <= 0 or self.port > 65535:
            raise ValueError("Invalid port number")
