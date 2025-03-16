import csv
import logging
import sqlite3
from contextlib import contextmanager
from functools import wraps
from typing import Callable, Iterator, TypeVar, Union

import psycopg2
from config import DBConfig
from psycopg2.extensions import connection

F = TypeVar('F', bound=Callable)

logger = logging.getLogger(__name__)
# database.py
class DatabaseManager:
    def __init__(self, config: DBConfig = None, use_memory: bool = False, sqlite_file: str = None):  # <- Nuevo parámetro
        self.config = config
        self.use_memory = use_memory
        self.sqlite_file = sqlite_file  # <- Nueva variable
        self.memory_conn: Union[sqlite3.Connection, None] = None
    @staticmethod
    def error_handler(func: F) -> F:
        """Maneja errores de base de datos y hace rollback automático"""
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except (psycopg2.Error, sqlite3.Error) as e:
                logger.error(f"Error DB en {func.__name__}: {e}", exc_info=True)
                if hasattr(self, 'connection') and self.connection:
                    self.connection.rollback()
                raise
            except Exception as e:
                logger.critical(f"Error crítico en {func.__name__}: {e}")
                raise
        return wrapper

    @contextmanager
    def connection(self) -> Iterator[Union[sqlite3.Connection, connection]]:
        conn = None
        try:
            if self.use_memory:
                if not self.memory_conn:
                    self.memory_conn = sqlite3.connect(":memory:")
                conn = self.memory_conn
                yield conn
            elif self.sqlite_file:  # <- Nueva condición para SQLite en archivo
                conn = sqlite3.connect(self.sqlite_file)
                yield conn
            else:
                conn = psycopg2.connect(**self.config.__dict__)
                yield conn
        except (psycopg2.Error, sqlite3.Error) as e:
            logger.error(f"Database error: {e}")
            raise
        finally:
            # Cerrar solo conexiones que no son en memoria o SQLite file
            if not self.use_memory and not self.sqlite_file and conn:
                conn.close()

    def copy_from_buffer(self, buffer, table: str, conflict_action="DO UPDATE"):
        buffer.seek(0)
        reader = csv.DictReader(buffer)
        columns = reader.fieldnames
        
        with self.connection() as conn, conn.cursor() as cur:
            for row in reader:
                cleaned_row = {
                    key: None if value == '\\N' else value 
                    for key, value in row.items()
                }
                
                # Generar SQL dinámico para UPSERT
                placeholders = ', '.join([f"%({col})s" for col in columns])
                update_clause = ', '.join([f"{col} = EXCLUDED.{col}" 
                                        for col in columns if col not in ('user_id', 'anime_id')])
                
                sql = f"""
                    INSERT INTO {table} ({', '.join(columns)})
                    VALUES ({placeholders})
                    ON CONFLICT (user_id, anime_id) 
                    DO UPDATE SET {update_clause}
                """ if conflict_action == "DO UPDATE" else f"""
                    INSERT INTO {table} ({', '.join(columns)})
                    VALUES ({placeholders})
                    ON CONFLICT {conflict_action}
                """
                
                cur.execute(sql, cleaned_row)
            conn.commit()