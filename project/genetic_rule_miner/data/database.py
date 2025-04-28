import csv
from contextlib import contextmanager

from sqlalchemy import create_engine, text

from genetic_rule_miner.config import DBConfig
from genetic_rule_miner.utils.exceptions import DatabaseError
from genetic_rule_miner.utils.logging import LogManager

logger = LogManager.get_logger(__name__)


class DatabaseManager:
    """Unified database interface using SQLAlchemy with PostgreSQL."""

    def __init__(self, config: DBConfig) -> None:
        """Initialize with configuration."""
        self.config = config
        self._engine = None
        self._session_factory = None
        self.initialize()

    def __del__(self) -> None:
        """Clean up resources on destruction."""
        if self._engine:
            logger.info("SQLAlchemy engine closed")

    @contextmanager
    def connection(self):
        """Provide a database connection from the SQLAlchemy engine."""
        # Return the engine directly, instead of the session
        try:
            if self._engine:
                yield self._engine.connect()  # Yield a raw connection
            else:
                raise DatabaseError("Database engine not initialized")
        except Exception as e:
            logger.error("Database connection error", exc_info=True)
            raise DatabaseError("Connection failed") from e
        finally:
            # Ensure we close the connection after usage
            if self._engine:
                self._engine.dispose()

    def initialize(self) -> None:
        """Initialize SQLAlchemy engine and session factory."""
        self._init_sqlalchemy_engine()

    def _init_sqlalchemy_engine(self) -> None:
        """Initialize SQLAlchemy engine with PostgreSQL connection."""
        try:
            # Create the SQLAlchemy engine for PostgreSQL
            conn_str = f"postgresql+psycopg2://{self.config.user}:{self.config.password}@{self.config.host}:{self.config.port}/{self.config.database}"
            self._engine = create_engine(conn_str)
            logger.info("SQLAlchemy engine initialized")
        except Exception as e:
            logger.critical(
                "Failed to initialize SQLAlchemy engine", exc_info=True
            )
            raise DatabaseError(
                "SQLAlchemy engine initialization failed"
            ) from e

    def copy_from_buffer(
        self, buffer, table: str, conflict_action="DO UPDATE"
    ):
        """Copy data from the buffer to the PostgreSQL table."""
        buffer.seek(0)
        reader = csv.DictReader(buffer)
        columns = reader.fieldnames

        with self.connection() as conn:
            for row in reader:
                cleaned_row = {
                    key: None if value == "\\N" else value
                    for key, value in row.items()
                }

                # Determine conflict columns based on table
                if table == "user_score":
                    conflict_columns = ["user_id", "anime_id"]
                elif table == "anime_dataset":
                    conflict_columns = ["anime_id"]
                elif table == "user_details":
                    conflict_columns = ["mal_id"]
                else:
                    conflict_columns = []  # Default to no conflict resolution

                placeholders = ", ".join([f":{col}" for col in columns])

                if conflict_columns and conflict_action == "DO UPDATE":
                    update_clause = ", ".join(
                        [
                            f"{col} = EXCLUDED.{col}"
                            for col in columns
                            if col not in conflict_columns
                        ]
                    )

                    sql = f"""
                        INSERT INTO {table} ({', '.join(columns)})
                        VALUES ({placeholders})
                        ON CONFLICT ({', '.join(conflict_columns)}) 
                        DO UPDATE SET {update_clause}
                    """
                elif conflict_columns:
                    sql = f"""
                        INSERT INTO {table} ({', '.join(columns)})
                        VALUES ({placeholders})
                        ON CONFLICT ({', '.join(conflict_columns)}) {conflict_action}
                    """
                else:
                    sql = f"""
                        INSERT INTO {table} ({', '.join(columns)})
                        VALUES ({placeholders})
                    """

                # Execute the query using the raw connection
                conn.execute(text(sql), cleaned_row)
