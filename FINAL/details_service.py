import csv
import json
import logging
import time
from datetime import datetime, timedelta
from io import BytesIO, StringIO
from typing import List, Optional

import requests
from config import APIConfig, LogConfig
from database import DatabaseManager

LogConfig.setup()
logger = logging.getLogger(__name__)

class DetailsService:
    def __init__(self, config: APIConfig = APIConfig()):
        self.config = config
        self.db = DatabaseManager(sqlite_file="user_details_cache.db")
        self._create_cache_table()
        
        # Optimización de parámetros
        self.batch_size = 3
        self.request_delay = 0.35
        self.batch_delay = 1.0
        self.max_retries = 3

    def _create_cache_table(self):
        """Mejora estructura de la tabla de caché"""
        with self.db.connection() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS user_details_cache (
                    username TEXT PRIMARY KEY,
                    mal_id INTEGER,
                    data TEXT,
                    timestamp DATETIME
                )
            ''')
            conn.commit()

    def get_user_details(self, usernames: List[str]) -> BytesIO:
        """Genera CSV con datos detallados optimizados"""
        buffer = StringIO()
        writer = csv.writer(buffer)
        writer.writerow([
            "Mal ID", "Username", "Gender", "Birthday", "Location", "Joined",
            "Days Watched", "Mean Score", "Watching", "Completed", "On Hold",
            "Dropped", "Plan to Watch", "Total Entries", "Rewatched", "Episodes Watched"
        ])
        
        total = len(usernames)
        start_time = time.time()
        
        for i in range(0, total, self.batch_size):
            batch = usernames[i:i+self.batch_size]
            batch_data = []
            
            for username in batch:
                if cached := self._get_cached_details(username):
                    writer.writerow(cached)
                    continue
                
                if data := self._fetch_user_data(username):
                    writer.writerow(data)
                    self._update_cache(username, data)
                    batch_data.append(data)
            
            logger.info(f"Procesados {min(i+self.batch_size, total)}/{total}")
            self._handle_rate_limits(len(batch_data))
        
        logger.info(f"Tiempo total: {time.time()-start_time:.2f}s")
        buffer.seek(0)
        return BytesIO(buffer.getvalue().encode('utf-8'))


    def _fetch_user_data(self, username: str) -> Optional[list]:
        """Obtiene datos de usuario con reintentos"""
        for attempt in range(self.max_retries):
            try:
                response = requests.get(
                    f"https://api.jikan.moe/v4/users/{username}/full",
                    timeout=self.config.timeout
                )
                
                if response.status_code == 200:
                    return self._parse_response(response.json())
                
                if response.status_code == 404:
                    return None
                    
                logger.warning(f"Intento {attempt+1} fallido para {username}")
                time.sleep(2**attempt)
                
            except Exception as e:
                logger.error(f"Error en {username}: {str(e)}")
                time.sleep(2**attempt)
        
        return None

    def _parse_response(self, response: dict) -> list:
        """Extrae y estructura los datos relevantes"""
        data = response.get('data', {})
        stats = data.get('statistics', {}).get('anime', {})
        
        return [
            data.get('mal_id'),
            data.get('username'),
            data.get('gender'),
            data.get('birthday'),
            data.get('location'),
            data.get('joined'),
            stats.get('days_watched'),
            stats.get('mean_score'),
            stats.get('watching'),
            stats.get('completed'),
            stats.get('on_hold'),
            stats.get('dropped'),
            stats.get('plan_to_watch'),
            stats.get('total_entries'),
            stats.get('rewatched'),
            stats.get('episodes_watched')
        ]

    def _get_cached_details(self, username: str) -> Optional[list]:
        """Recupera datos cacheados válidos"""
        with self.db.connection() as conn:
            cursor = conn.execute(
                'SELECT mal_id, data, timestamp FROM user_details_cache WHERE username = ?',
                (username,)
            )
            if row := cursor.fetchone():
                if (datetime.now() - datetime.fromisoformat(row[2])) < timedelta(days=7):
                    return [row[0], username] + json.loads(row[1])
        return None

    def _update_cache(self, username: str, data: list):
        """Actualiza la caché con nuevos datos"""
        with self.db.connection() as conn:
            conn.execute(
                '''INSERT OR REPLACE INTO user_details_cache 
                (username, mal_id, data, timestamp) 
                VALUES (?, ?, ?, ?)''',
                (username, data[0], json.dumps(data[2:]), datetime.now())
            )
            conn.commit()

    def _handle_rate_limits(self, batch_size: int):
        """Gestiona los límites de la API"""
        if batch_size > 0:
            time.sleep(self.batch_delay)