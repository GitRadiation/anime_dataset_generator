import csv
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import Connection, create_engine, text

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
    def connection(self) -> Generator[Connection, None, None]:
        """Provide a database connection from the SQLAlchemy engine."""
        connection = None
        try:
            if self._engine:
                connection = self._engine.connect()  # Create a raw connection
                yield connection  # Yield the connection
            else:
                raise DatabaseError("Database engine not initialized")
        except Exception as e:
            logger.error("Database connection error", exc_info=True)
            raise DatabaseError("Connection failed") from e

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

    def _construct_sql(
        self,
        table,
        columns,
        conflict_columns,
        conflict_action,
        update_clause=None,
    ) -> str:
        """Construct the SQL query for inserting or updating data."""
        placeholders = ", ".join([f":{col}" for col in columns])

        if conflict_columns:
            conflict_str = f"ON CONFLICT ({', '.join(conflict_columns)})"
            if conflict_action == "DO UPDATE":
                update_clause = update_clause or ", ".join(
                    [
                        f"{col} = EXCLUDED.{col}"
                        for col in columns
                        if col not in conflict_columns
                    ]
                )
                sql = f"""
                    INSERT INTO {table} ({', '.join(columns)})
                    VALUES ({placeholders})
                    {conflict_str} DO UPDATE SET {update_clause}
                """
            else:
                sql = f"""
                    INSERT INTO {table} ({', '.join(columns)})
                    VALUES ({placeholders})
                    {conflict_str} {conflict_action}
                """
        else:
            sql = f"""
                INSERT INTO {table} ({', '.join(columns)})
                VALUES ({placeholders})
            """
        return sql

    def copy_from_buffer(
        self, conn: Connection, buffer, table: str, conflict_action="DO UPDATE"
    ) -> None:
        """Copy data from the buffer to the PostgreSQL table."""
        buffer.seek(0)
        reader = csv.DictReader(buffer)
        columns = reader.fieldnames

        # Determine conflict columns based on table
        table_conflict_columns = self._get_conflict_columns(table)

        for row in reader:
            cleaned_row = {
                key: None if value == "\\N" else value
                for key, value in row.items()
            }

            # Construct the SQL query for the insert or update operation
            sql = self._construct_sql(
                table, columns, table_conflict_columns, conflict_action
            )

            # Execute the query using the raw connection
            conn.execute(text(sql), cleaned_row)

    def _get_conflict_columns(self, table: str) -> list:
        """Return the list of conflict columns based on the table."""
        if table == "user_score":
            return ["user_id", "anime_id"]
        elif table == "anime_dataset":
            return ["anime_id"]
        elif table == "user_details":
            return ["mal_id"]
        else:
            return []  # Default to no conflict resolution
