"""Advanced database connection management with connection pooling."""

import sqlite3
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Iterator, Optional, TypeVar

import psycopg2
from psycopg2 import pool
from psycopg2.extensions import connection as PgConnection
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from genetic_rule_miner.config import DBConfig
from genetic_rule_miner.utils.exceptions import DatabaseError
from genetic_rule_miner.utils.logging import LogManager

logger = LogManager.get_logger(__name__)
F = TypeVar("F", bound=Callable[..., Any])


class DatabaseManager:
    """Unified database interface with connection pooling using SQLAlchemy."""

    def __init__(self, config: DBConfig) -> None:
        """Initialize with configuration."""
        self.config = config
        self._pg_pool: Optional[pool.ThreadedConnectionPool] = None
        self._engine = None
        self._session_factory = None

    def initialize(self) -> None:
        """Initialize connection pools and SQLAlchemy engine."""
        self._init_postgres_pool()
        self._init_sqlalchemy_engine()

    def _init_postgres_pool(self) -> None:
        """Initialize PostgreSQL connection pool with psycopg2."""
        try:
            conn_params = {
                "host": self.config.host,
                "port": self.config.port,
                "dbname": self.config.database,
                "user": self.config.user,
                "password": self.config.password,
            }

            self._pg_pool = pool.ThreadedConnectionPool(
                minconn=1, maxconn=self.config.pool_size, **conn_params
            )
            logger.info("PostgreSQL connection pool initialized")
        except psycopg2.Error as e:
            logger.critical(
                "Failed to initialize PostgreSQL pool", exc_info=True
            )
            raise DatabaseError("PostgreSQL pool initialization failed") from e

    def _init_sqlalchemy_engine(self) -> None:
        """Initialize SQLAlchemy engine with PostgreSQL connection pool."""
        try:
            # Crear la cadena de conexión para SQLAlchemy usando psycopg2
            conn_str = f"postgresql+psycopg2://{self.config.user}:{self.config.password}@{self.config.host}:{self.config.port}/{self.config.database}"
            self._engine = create_engine(conn_str)
            self._session_factory = sessionmaker(bind=self._engine)
            logger.info("SQLAlchemy engine initialized")
        except Exception as e:
            logger.critical(
                "Failed to initialize SQLAlchemy engine", exc_info=True
            )
            raise DatabaseError(
                "SQLAlchemy engine initialization failed"
            ) from e

    @contextmanager
    def connection(self) -> Iterator[PgConnection]:
        """Provide a database connection from the SQLAlchemy engine."""
        conn = None
        try:
            if self._engine:
                conn = self._engine.connect()
                yield conn
            else:
                raise DatabaseError("Database engine not initialized")
        except Exception as e:
            logger.error("Database connection error", exc_info=True)
            if conn:
                conn.rollback()
            raise DatabaseError("Connection failed") from e
        finally:
            if conn:
                conn.close()

    def error_handler(self, func: F) -> F:
        """Decorator for database error handling and transaction management."""

        @wraps(func)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            try:
                result = func(self, *args, **kwargs)
                if hasattr(self, "connection") and self.connection:
                    self.connection.commit()
                return result
            except (psycopg2.Error, sqlite3.Error) as e:
                logger.error(
                    f"Database error in {func.__name__}", exc_info=True
                )
                if hasattr(self, "connection") and self.connection:
                    self.connection.rollback()
                raise DatabaseError(
                    f"Operation failed: {func.__name__}"
                ) from e
            except Exception:
                logger.critical(
                    f"Unexpected error in {func.__name__}", exc_info=True
                )
                raise

        return wrapper

    def __del__(self) -> None:
        """Clean up resources on destruction."""
        if self._pg_pool:
            self._pg_pool.closeall()
            logger.info("PostgreSQL connection pool closed")
