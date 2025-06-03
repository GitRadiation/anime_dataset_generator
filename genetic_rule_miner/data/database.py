import ast
import csv
import json
import uuid
from contextlib import contextmanager
from io import BytesIO, StringIO
from typing import Generator, Optional

import numpy as np
import pandas as pd
from sqlalchemy import Connection, bindparam, create_engine, text
from sqlalchemy.dialects.postgresql import JSONB
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

    def __new__(
        cls, config: Optional[DBConfig] = DBConfig()
    ) -> "DatabaseManager":
        if cls._instance is None:
            cls._instance = super(DatabaseManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config: Optional[DBConfig] = DBConfig()) -> None:
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
            if (
                self.config
                and self.config.password
                and self.config.user
                and self.config.host
                and self.config.port
                and self.config.database
            ):
                conn_str = (
                    f"postgresql+psycopg2://{self.config.user}:{self.config.password}"
                    f"@{self.config.host}:{self.config.port}/{self.config.database}"
                )
                self._engine = create_engine(conn_str)
                logger.info("SQLAlchemy engine initialized")
            else:
                logger.error("Config is empty")
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
        ]

        rows_to_insert = []
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
            rows_to_insert.append(cleaned_row)

        if not rows_to_insert:
            logger.info("No data loaded (empty or all rows skipped)")
            return False

        sql = self._construct_sql(
            table, columns, table_conflict_columns, conflict_action
        )
        try:
            conn.execute(text(sql), rows_to_insert)
            conn.commit()
            logger.info("✅ Data loading completed successfully")
            return True
        except IntegrityError as e:
            logger.error(f"Database error: {e}")
            raise

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
        # Nueva consulta: reglas cuyo target_value tenga <= 250 reglas
        query_rules_targets = """
            SELECT r.rule_id, r.target_value
            FROM rules r
            JOIN (
                SELECT target_value
                FROM rules
                GROUP BY target_value
                HAVING COUNT(*) <= 250
            ) AS t ON r.target_value = t.target_value;
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

    def get_rules_by_target_value(self, target_value: int) -> list[Rule]:
        """
        Devuelve todas las reglas asociadas a un target_value específico,
        en forma de lista de objetos Rule.
        """
        with self.connection() as conn:
            result = (
                conn.execute(
                    text(
                        "SELECT conditions, target_value FROM rules WHERE target_value::integer = :tid"
                    ),
                    {"tid": target_value},
                )
                .mappings()
                .all()
            )

            rules = []
            for row in result:
                try:
                    conditions = row["conditions"]
                    if isinstance(conditions, str):
                        conditions = ast.literal_eval(conditions)
                    rule = Rule(
                        columns=[],
                        conditions=conditions,
                        target=np.int64(row["target_value"]),
                    )
                    rules.append(rule)
                except Exception as e:
                    logger.warning(
                        f"No se pudo construir Rule para target {target_value}: {e}"
                    )
            return rules

    def get_rules_series_by_json(self, json_objeto: dict):
        """
        Ejecuta la función SQL get_rules_series pasando el JSON completo obtenido de la API.

        :param json_objeto: Diccionario completo del JSON recibido desde la API.
        :return: Lista de tuplas (nombre, cantidad) con los resultados.
        """

        def clean_nans(obj):
            if isinstance(obj, dict):
                return {k: clean_nans(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_nans(elem) for elem in obj]
            elif isinstance(obj, float) and np.isnan(obj):
                return None
            return obj

        cleaned_data = clean_nans(json_objeto)
        json_payload = json.dumps(cleaned_data)
        with self.connection() as conn:
            # Ejecutar la función que devuelve las series de reglas personalizadas
            result = conn.execute(
                text("SELECT * FROM get_rules_series(:input_json)").bindparams(
                    bindparam("input_json", type_=JSONB)
                ),
                {"input_json": json_payload},
            ).fetchall()

            # Convertimos a lista para poder contarlos
            result_list = [dict(row) for row in result]

            # Si no hay resultados o son menos de 5, ejecutamos la consulta de respaldo
            if len(result_list) < 5:
                # Ajustar el límite para completar hasta 5
                limit = 5 - len(result_list)

                # Ejecutar la consulta de respaldo
                fallback_result = conn.execute(
                    text(
                        f"""
                        SELECT 
                            ad.anime_id as anime_id,
                            ad.name as nombre,
                            COUNT(*) AS cantidad
                        FROM 
                            rules r
                        JOIN 
                            anime_dataset ad ON ad.anime_id = r.target_value
                        GROUP BY 
                            ad.anime_id, ad.name
                        ORDER BY 
                            cantidad DESC
                        LIMIT {limit};
                    """
                    )
                ).fetchall()

                # Convertimos y añadimos los resultados de respaldo
                fallback_result_list = [
                    dict(row._mapping) for row in fallback_result
                ]
                result_list.extend(fallback_result_list)

            # Ahora result_list contiene hasta 5 resultados
            return result_list
