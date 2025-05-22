import ast
import csv
import json
import uuid
from contextlib import contextmanager
from io import BytesIO, StringIO
from typing import Generator, Optional

import pandas as pd
from sqlalchemy import Connection, create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError

from genetic_rule_miner.config import DBConfig
from genetic_rule_miner.utils.exceptions import DatabaseError
from genetic_rule_miner.utils.logging import LogManager
from genetic_rule_miner.utils.rule import Rule

logger = LogManager.get_logger(__name__)


class DatabaseManager:
    """Singleton Database Manager using SQLAlchemy for PostgreSQL."""

    _instance = None

    _engine: Optional[Engine] = None

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
        self,
        conn: Connection,
        buffer: StringIO,
        table: str,
        conflict_action="DO UPDATE",
    ) -> bool:
        def to_pg_array(arr):
            def escape_element(el):
                el = str(el).replace("\\", "\\\\").replace('"', '\\"')
                # Si el elemento contiene comas, espacios o llaves, lo ponemos entre comillas
                if any(c in el for c in [",", " ", "{", "}", '"']):
                    return f'"{el}"'
                return el

            return "{" + ",".join(escape_element(e) for e in arr) + "}"

        buffer.seek(0)
        reader = csv.DictReader(buffer)
        columns = reader.fieldnames
        table_conflict_columns = self._get_conflict_columns(table)

        array_columns = [
            "genres",
            "keywords",
            "producers",
        ]  # agrega aquí todas las columnas array

        inserted = 0  # contador de filas insertadas

        for row in reader:
            cleaned_row = {
                key: None if value == "\\N" else value
                for key, value in row.items()
            }

            for col in array_columns:
                if col in cleaned_row and cleaned_row[col]:
                    try:
                        value = cleaned_row[col]
                        if isinstance(value, str):
                            lst = ast.literal_eval(value)
                            if isinstance(lst, list):
                                cleaned_row[col] = to_pg_array(lst)
                    except Exception:
                        pass

            sql = self._construct_sql(
                table, columns, table_conflict_columns, conflict_action
            )
            try:
                conn.execute(text(sql), cleaned_row)
                inserted += 1
            except IntegrityError as e:
                if "foreign key constraint" in str(e.orig).lower():
                    logger.warning(
                        f"Skipping row due to foreign key violation: {cleaned_row}"
                    )
                    continue
                else:
                    logger.error(f"Database error on row {cleaned_row}: {e}")
                    raise
        if inserted > 0:
            conn.commit()
            logger.info("✅ Data loading completed successfully")
            return True
        else:
            logger.info("No data loaded (empty or all rows skipped)")
            return False

    def save_rules(self, rules: list[Rule], table: str = "rules") -> None:
        """
        Save Rule objects to the PostgreSQL database.
        """
        with self.connection() as conn:
            # Preparar todos los datos primero
            insert_data = []
            for rule in rules:
                user_conditions = [
                    {"column": col, "operator": op, "value": value}
                    for col, (op, value) in rule.conditions[0]
                ]
                other_conditions = [
                    {"column": col, "operator": op, "value": value}
                    for col, (op, value) in rule.conditions[1]
                ]

                insert_data.append(
                    {
                        "rule_id": str(uuid.uuid4()),
                        "conditions": json.dumps(
                            {
                                "user_conditions": user_conditions,
                                "other_conditions": other_conditions,
                            }
                        ),
                        "target_value": rule.target.item(),
                    }
                )

            # Insertar todo en una sola operación
            conn.execute(
                text(
                    f"""
                    INSERT INTO {table} (rule_id, conditions, target_value)
                    VALUES (:rule_id, :conditions, :target_value)
                """
                ),
                insert_data,
            )
            conn.commit()

    def _get_conflict_columns(self, table: str) -> list:
        if table == "user_score":
            return ["user_id", "anime_id"]
        elif table == "anime_dataset":
            return ["anime_id"]
        elif table == "user_details":
            return ["mal_id"]
        else:
            return []

    def export_users_to_csv_buffer(
        self, table: str = "user_details"
    ) -> BytesIO:
        """
        Export mal_id, username y user_url desde la tabla indicada a un buffer CSV en memoria.
        """
        # Ajusta los nombres de columnas según tu tabla real
        columns = ["mal_id", "username"]
        query = f"SELECT {', '.join(columns)} FROM {table}"

        buffer = StringIO()
        writer = csv.writer(buffer)
        writer.writerow(["mal_id", "username", "user_url"])

        with self.connection() as conn:
            result = conn.execute(text(query))
            print(result)
            for row in result.mappings():
                mal_id = row.get("mal_id")
                username = row.get("username")

                user_url = f"https://myanimelist.net/profile/{username}"
                writer.writerow([mal_id, username, user_url])

        buffer.seek(0)
        return BytesIO(buffer.getvalue().encode("utf-8"))

    def get_anime_ids_without_rules(
        self, export_path: Optional[str] = None
    ) -> Optional[list[int]]:
        # Consulta original: anime_ids sin reglas
        query_no_rules = """
            WITH rules_exist AS (
                SELECT EXISTS (SELECT 1 FROM rules) AS has_rules
            )
            SELECT DISTINCT a.anime_id
            FROM anime_dataset a
            CROSS JOIN rules_exist
            LEFT JOIN rules r ON CAST(r.target_value AS INTEGER) = a.anime_id
            WHERE (rules_exist.has_rules = FALSE AND r.rule_id IS NULL)
            OR rules_exist.has_rules = TRUE
        """
        # Nueva consulta: primeros 250 target_value de rules
        query_rules_targets = """
            SELECT DISTINCT ON (r.target_value) r.rule_id, r.target_value
            FROM rules r
            ORDER BY r.target_value, r.rule_id
            LIMIT 250
        """
        with self.connection() as conn:
            # IDs de anime_dataset
            result1 = conn.execute(text(query_no_rules))
            anime_ids = {row["anime_id"] for row in result1.mappings()}

            # IDs de target_value en rules
            result2 = conn.execute(text(query_rules_targets))
            rule_targets = {row["target_value"] for row in result2.mappings()}

            # Unión sin repetidos
            all_ids = list(anime_ids.union(rule_targets))

            # Exportar a Excel si se proporciona la ruta
            if export_path:
                df = pd.DataFrame(all_ids, columns=["anime_id"])
                df.to_excel(export_path, index=False)

            return all_ids or []
