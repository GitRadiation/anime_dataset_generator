import csv
import json
import logging
import time
from datetime import datetime
from io import BytesIO, StringIO
from typing import List, Optional

import requests
from config import APIConfig, LogConfig
from database import DatabaseManager

LogConfig.setup()
logger = logging.getLogger(__name__)

class UserService:
    def __init__(self, config: APIConfig = APIConfig()):
        self.config = config
        self.db = DatabaseManager(sqlite_file="user_cache.db")
        self._create_cache_table()
        
    def _create_cache_table(self):
        """Crea tabla de caché mejorada con registro de existencia"""
        with self.db.connection() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS user_cache (
                    user_id INTEGER PRIMARY KEY,
                    data TEXT,
                    user_exists BOOLEAN,
                    timestamp DATETIME
                )
            ''')
            conn.commit()

    def _fetch_with_retry(self, user_id: int) -> Optional[dict]:
        """Lógica de reintentos con manejo de errores"""
        for attempt in range(self.config.max_retries):
            try:
                response = requests.get(
                    f"https://api.jikan.moe/v4/users/userbyid/{user_id}",
                    timeout=self.config.timeout
                )
                
                if response.status_code == 200:
                    return response.json().get('data')
                elif response.status_code == 404:
                    return None
                    
                response.raise_for_status()
                
            except requests.exceptions.RequestException:
                if attempt < self.config.max_retries - 1:
                    time.sleep(2**attempt)  # Backoff exponencial
                    
        logger.error(f"Usuario ID {user_id} no disponible después de {self.config.max_retries} intentos")
        return None

    def _update_cache(self, user_id: int, data: Optional[dict]):
        """Actualiza caché con datos y estado de existencia"""
        exists = data is not None
        with self.db.connection() as conn:
            conn.execute(
    '''INSERT OR REPLACE INTO user_cache 
    (user_id, data, user_exists, timestamp)
    VALUES (?, ?, ?, ?)''',
    (
        user_id,
        json.dumps(data) if data else None,
        exists,  # Variable sin cambios
        datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    )
)
            conn.commit()

    def generate_userlist(self, start_id: int, end_id: int) -> BytesIO:
        """Genera lista de usuarios con búsqueda por rango de IDs"""
        text_buffer = StringIO()
        writer = csv.DictWriter(
            text_buffer,
            fieldnames=['user_id', 'username', 'user_url'],
            extrasaction='ignore'
        )
        writer.writeheader()
        
        valid_users = 0
        total_processed = 0
        
        for user_id in range(start_id, end_id + 1):
            try:
                cached_data = self.get_cached_user(user_id)
                if cached_data:
                    if cached_data.get('user_exists'):  # Asegurar que se usa el nombre correcto
                        writer.writerow(cached_data)
                        valid_users += 1
                    continue

                data = self._fetch_with_retry(user_id)
                user_record = {
                    'user_id': user_id,
                    'username': data.get('username') if data else None,
                    'user_url': data.get('url') if data else None
                }

                self._update_cache(user_id, data)
                if data:
                    writer.writerow(user_record)
                    valid_users += 1

                time.sleep(self.config.request_delay)
                
            except Exception as e:
                logger.error(f"Error crítico procesando ID {user_id}: {str(e)}")
            finally:
                total_processed += 1
                if total_processed % 100 == 0:
                    logger.info(f"Progreso: {total_processed/(end_id-start_id+1)*100:.1f}%")

        # Convertir a bytes antes de retornar
        text_buffer.seek(0)
        byte_buffer = BytesIO(text_buffer.getvalue().encode('utf-8'))
        logger.info(f"Generación completada. Usuarios válidos: {valid_users}/{total_processed}")
        return byte_buffer

    def get_cached_user(self, user_id: int) -> Optional[dict]:
        """Recupera datos de usuario desde la caché"""
        with self.db.connection() as conn:
            cursor = conn.execute(
            'SELECT data, user_exists FROM user_cache WHERE user_id = ?',  # Columna renombrada
            (user_id,)
        )
            result = cursor.fetchone()
            
            if result:
                data_str, exists = result
                return {
                    'user_id': user_id,
                    'username': json.loads(data_str).get('username') if data_str else None,
                    'user_url': json.loads(data_str).get('url') if data_str else None,
                    'exists': exists
                }
            return None

    def get_users(self, user_ids: List[int]) -> BytesIO:
        """Método existente para obtener múltiples usuarios (compatibilidad)"""
        buffer = BytesIO()
        writer = csv.DictWriter(buffer, fieldnames=['user_id', 'username', 'user_url'])
        writer.writeheader()
        
        for user_id in user_ids:
            if cached := self.get_cached_user(user_id):
                if cached['exists']:
                    writer.writerow(cached)
            else:
                data = self._fetch_with_retry(user_id)
                if data:
                    record = {
                        'user_id': user_id,
                        'username': data.get('username'),
                        'user_url': data.get('url')
                    }
                    writer.writerow(record)
                    self._update_cache(user_id, data)
                time.sleep(self.config.request_delay)
        
        buffer.seek(0)
        return buffer