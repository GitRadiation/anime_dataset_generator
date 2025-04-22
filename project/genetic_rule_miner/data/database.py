"""Advanced database connection management with connection pooling."""

import sqlite3
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Iterator, Optional, TypeVar, Union

import psycopg2
from psycopg2 import pool
from psycopg2.extensions import connection as PgConnection

from genetic_rule_miner.config import DBConfig
from genetic_rule_miner.utils.exceptions import DatabaseError
from genetic_rule_miner.utils.logging import LogManager

logger = LogManager.get_logger(__name__)
F = TypeVar('F', bound=Callable[..., Any])

class DatabaseManager:
    """Unified database interface with connection pooling."""
    
    def __init__(self, config: DBConfig) -> None:
        """Initialize with configuration."""
        self.config = config
        self._pg_pool: Optional[pool.ThreadedConnectionPool] = None
        self._sqlite_conn: Optional[sqlite3.Connection] = None
        
    def initialize(self) -> None:
        """Initialize connection pools."""
        if not self.config.sqlite_file:
            self._init_postgres_pool()
        
    def _init_postgres_pool(self) -> None:
        """Initialize PostgreSQL connection pool."""
        try:
            # Extract only the relevant connection parameters
            conn_params = {
                'host': self.config.host,
                'port': self.config.port,
                'dbname': self.config.database,
                'user': self.config.user,
                'password': self.config.password,
            }
            
            self._pg_pool = pool.ThreadedConnectionPool(
                minconn=1,
                maxconn=self.config.pool_size,
                **conn_params
            )
            logger.info("PostgreSQL connection pool initialized")
        except psycopg2.Error as e:
            logger.critical("Failed to initialize PostgreSQL pool", exc_info=True)
            raise DatabaseError("PostgreSQL pool initialization failed") from e

    @contextmanager
    def connection(self) -> Iterator[Union[sqlite3.Connection, PgConnection]]:
        """Provide a database connection from the appropriate pool."""
        conn: Union[sqlite3.Connection, PgConnection, None] = None
        try:
            if self.config.sqlite_file:
                conn = sqlite3.connect(self.config.sqlite_file)
                yield conn
            elif self._pg_pool:
                # Remove timeout parameter from getconn()
                conn = self._pg_pool.getconn()
                yield conn
                self._pg_pool.putconn(conn)
            else:
                raise DatabaseError("Database not initialized")
        except (psycopg2.Error, sqlite3.Error) as e:
            logger.error("Database connection error", exc_info=True)
            if conn:
                conn.rollback()
            raise DatabaseError("Connection failed") from e
        finally:
            if self.config.sqlite_file and conn:
                conn.close()

    @staticmethod
    def error_handler(func: F) -> F:
        """Decorator for database error handling and transaction management."""
        @wraps(func)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            try:
                result = func(self, *args, **kwargs)
                if hasattr(self, 'connection') and self.connection:
                    self.connection.commit()
                return result
            except (psycopg2.Error, sqlite3.Error) as e:
                logger.error(f"Database error in {func.__name__}", exc_info=True)
                if hasattr(self, 'connection') and self.connection:
                    self.connection.rollback()
                raise DatabaseError(f"Operation failed: {func.__name__}") from e
            except Exception:
                logger.critical(f"Unexpected error in {func.__name__}", exc_info=True)
                raise
        return wrapper

    def __del__(self) -> None:
        """Clean up resources on destruction."""
        if self._pg_pool:
            self._pg_pool.closeall()
            logger.info("PostgreSQL connection pool closed")