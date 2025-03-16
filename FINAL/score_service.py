import csv
import json
import logging
import random
import time
from datetime import datetime
from io import BytesIO, StringIO
from typing import List, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup
from config import APIConfig, LogConfig
from database import DatabaseManager

LogConfig.setup()
logger = logging.getLogger(__name__)

class ScoreService:
    def __init__(self, config: APIConfig = APIConfig()):
        self.config = config
        self.db = DatabaseManager(sqlite_file="score_cache.db")
        self._create_cache_table()
        
        # Configuración de scraping
        self.status_code = 7  # Completed anime
        self.batch_size = 50
        self.min_delay = 90
        self.max_delay = 120

    def _create_cache_table(self):
        """Actualiza la estructura de la tabla de caché"""
        with self.db.connection() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS score_cache (
                    user_id INTEGER,
                    anime_id INTEGER,
                    score INTEGER,
                    anime_title TEXT,
                    timestamp DATETIME,
                    PRIMARY KEY (user_id, anime_id)
                    )
            ''')
            conn.commit()

    def _process_batch(self, users_batch: List[dict]) -> List[list]:
        """Procesa un lote de usuarios con manejo de errores"""
        batch_data = []
        for user in users_batch:
            try:
                if data := self._scrape_user_scores(user['username'], user['mal_id']):
                    batch_data.extend(data)
                    logger.info(f"Datos obtenidos para {user['username']}")
                else:
                    logger.warning(f"Sin datos para {user['username']}")
            except Exception as e:
                logger.error(f"Error procesando {user['username']}: {str(e)}")
        return batch_data

    def _scrape_user_scores(self, username: str, user_id: int) -> Optional[List[list]]:
        """Lógica principal de scraping con doble estructura de tablas"""
        try:
            url = f"https://myanimelist.net/animelist/{username}?status={self.status_code}"
            response = requests.get(url, timeout=self.config.timeout)
            
            if response.status_code != 200:
                return None

            soup = BeautifulSoup(response.content, 'html.parser')
            return self._parse_modern_table(soup, user_id, username) or \
                   self._parse_legacy_tables(soup, user_id, username)
                   
        except Exception as e:
            logger.error(f"Error de scraping en {username}: {str(e)}")
            return None

    def _parse_modern_table(self, soup, user_id, username):
        """Maneja la tabla moderna con data-items"""
        if table := soup.find('table', {'data-items': True}):
            try:
                return [
                    [user_id, username, item['anime_id'], item['anime_title'], item['score']]
                    for item in json.loads(table['data-items'])
                    if item['score'] > 0
                ]
            except json.JSONDecodeError:
                pass
        return None

    def _parse_legacy_tables(self, soup, user_id, username):
        """Maneja la estructura antigua de tablas"""
        scores = []
        for table in soup.find_all('table', {'border': "0", 'cellpadding': "0", 'cellspacing': "0", 'width': "100%"}):
            for row in table.find_all('tr'):
                cells = row.find_all('td')
                if len(cells) >= 5:
                    anime_data = self._extract_anime_data(cells[1])
                    score_data = self._extract_score_data(cells[2])
                    if anime_data and score_data:
                        scores.append([user_id, username, anime_data[0], anime_data[1], score_data])
        return scores if scores else None

    def _extract_anime_data(self, cell):
        """Extrae ID y título del anime de una celda"""
        if link := cell.find('a', class_='animetitle'):
            return (
                link['href'].split('/')[2],
                link.find('span').text.strip()
            )
        return None

    def _extract_score_data(self, cell):
        """Extrae la puntuación de una celda"""
        if score_label := cell.find('span', class_='score-label'):
            return int(score_label.text.strip()) if score_label.text.strip() != '-' else None
        return None

    def get_scores(self, users_buffer: BytesIO) -> BytesIO:
        """Genera CSV de scores integrando caché y batching"""
        users_df = pd.read_csv(users_buffer)
        buffer = StringIO()
        writer = csv.writer(buffer)
        writer.writerow(["User ID", "Username", "Anime ID", "Anime Title", "Score"])
        
        total_users = len(users_df)
        processed = 0
        
        for i in range(0, total_users, self.batch_size):
            batch = users_df.iloc[i:i+self.batch_size].to_dict('records')
            batch_data = self._process_batch_with_cache(batch)
            
            if batch_data:
                writer.writerows(batch_data)
                self._update_cache(batch_data)
                processed += len(batch)
                logger.info(f"Procesado: {processed}/{total_users} ({processed/total_users:.1%})")
            
            if i + self.batch_size < total_users:
                delay = random.randint(self.min_delay, self.max_delay)
                logger.info(f"Esperando {delay}s para siguiente lote...")
                time.sleep(delay)
        
        buffer.seek(0)
        return BytesIO(buffer.getvalue().encode('utf-8'))

    def _process_batch_with_cache(self, batch: List[dict]) -> List[list]:
        """Combina datos de caché y scraping para un lote"""
        batch_data = []
        users_to_scrape = []
        
        # Verificar caché primero
        for user in batch:
            if cached := self._get_cached_scores(user.get('user_id')):
                batch_data.extend(cached)
            else:
                users_to_scrape.append(user)
        
        # Scrapear usuarios no cacheados
        if users_to_scrape:
            scraped_data = self._process_batch(users_to_scrape)
            batch_data.extend(scraped_data)
        
        return batch_data

    def _get_cached_scores(self, user_id: int) -> Optional[List[list]]:
        """Recupera scores desde la caché"""
        with self.db.connection() as conn:
            cursor = conn.execute(
                'SELECT anime_id, score, anime_title FROM score_cache WHERE user_id = ?',
                (user_id,)
            )
            return [
                [user_id, None, row[0], row[2], row[1]]  # Mantener formato CSV
                for row in cursor.fetchall()
            ]

    def _update_cache(self, data: List[list]):
        """Actualiza la caché con nuevos datos"""
        with self.db.connection() as conn:
            for row in data:
                conn.execute(
                    '''INSERT OR REPLACE INTO score_cache 
                    (user_id, anime_id, score, anime_title, timestamp)
                    VALUES (?, ?, ?, ?, ?)''',
                    (row[0], row[2], row[4], row[3], datetime.now())
                )
            conn.commit()