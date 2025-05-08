import csv
import json
import uuid
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import Connection, create_engine, text
from sqlalchemy.engine import Engine

from genetic_rule_miner.config import DBConfig
from genetic_rule_miner.utils.exceptions import DatabaseError
from genetic_rule_miner.utils.logging import LogManager

logger = LogManager.get_logger(__name__)


class DatabaseManager:
    """Singleton Database Manager using SQLAlchemy for PostgreSQL."""

    _instance = None
    _engine: Engine = None

    def __new__(cls, config: DBConfig) -> "DatabaseManager":
        if cls._instance is None:
            cls._instance = super(DatabaseManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config: DBConfig) -> None:
        if self._initialized:
            return
        self.config = config
        self._engine = None
        self._session_factory = None
        self.initialize()
        self._initialized = True

    def __del__(self) -> None:
        if self._engine:
            logger.info("SQLAlchemy engine closed")

    @contextmanager
    def connection(self) -> Generator[Connection, None, None]:
        connection = None
        try:
            if self._engine:
                connection = self._engine.connect()
                yield connection
                connection.commit()  # Commit to persist changes
            else:
                raise DatabaseError("Database engine not initialized")
        except Exception as e:
            logger.error("Database connection error", exc_info=True)
            raise DatabaseError("Connection failed") from e
        finally:
            if connection:
                connection.close()

    def initialize(self) -> None:
        self._init_sqlalchemy_engine()

    def _init_sqlalchemy_engine(self) -> None:
        try:
            conn_str = (
                f"postgresql+psycopg2://{self.config.user}:{self.config.password}"
                f"@{self.config.host}:{self.config.port}/{self.config.database}"
            )
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
        buffer.seek(0)
        reader = csv.DictReader(buffer)
        columns = reader.fieldnames
        table_conflict_columns = self._get_conflict_columns(table)

        for row in reader:
            cleaned_row = {
                key: None if value == "\\N" else value
                for key, value in row.items()
            }
            sql = self._construct_sql(
                table, columns, table_conflict_columns, conflict_action
            )
            conn.execute(text(sql), cleaned_row)

    def save_rules(self, rules: list[dict], table: str = "rules") -> None:
        """
        Save rules to the PostgreSQL database.
        """
        with self.connection() as conn:
            for rule in rules:
                rule_id = str(uuid.uuid4())
                target_column, target_value = rule["target"]
                conditions_json = json.dumps(
                    [
                        {"column": col, "operator": op, "value": value}
                        for col, op, value in rule["conditions"]
                    ]
                )

                sql = f"""
                    INSERT INTO {table} (rule_id, conditions, target_column, target_value)
                    VALUES (:rule_id, :conditions, :target_column, :target_value)
                """

                conn.execute(
                    text(sql),
                    {
                        "rule_id": rule_id,
                        "conditions": conditions_json,
                        "target_column": target_column,
                        "target_value": str(target_value),
                    },
                )
            conn.commit()  # Ensure data is saved

    def _get_conflict_columns(self, table: str) -> list:
        if table == "user_score":
            return ["user_id", "anime_id"]
        elif table == "anime_dataset":
            return ["anime_id"]
        elif table == "user_details":
            return ["mal_id"]
        else:
            return []
