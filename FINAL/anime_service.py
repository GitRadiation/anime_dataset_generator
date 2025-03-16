import ast
import json
import logging
from datetime import datetime, timedelta
from io import BytesIO
from typing import Optional

import pandas as pd
import requests
from config import APIConfig, LogConfig
from database import DatabaseManager
from rake_nltk import Rake

LogConfig.setup()
logger = logging.getLogger(__name__)
# anime_service.py
class AnimeService:
    def __init__(self, config: APIConfig = APIConfig()):
        self.config = config
        self.db = DatabaseManager(sqlite_file="anime_cache.db")  # <- Usar archivo
        self._create_cache_table()

    def _create_cache_table(self):
        """Crea la tabla de caché si no existe."""
        with self.db.connection() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS anime_cache (
                    anime_id INTEGER PRIMARY KEY,
                    data TEXT,
                    timestamp DATETIME
                )
            ''')
            conn.commit()

    def _fetch_anime(self, anime_id: int) -> Optional[dict]:
        """Obtiene los datos del anime desde la API."""
        for attempt in range(self.config.max_retries):
            try:
                response = requests.get(
                    f"{self.config.base_url}anime/{anime_id}",
                    timeout=self.config.timeout
                )
                response.raise_for_status()
                return response.json().get('data')
            except (requests.RequestException, KeyError):
                continue
        return None

    def get_anime_data(self, start_id: int, end_id: int) -> BytesIO:
        buffer = BytesIO()
        records = []
        r = Rake()  # Inicializar RAKE una sola vez

        for anime_id in range(start_id, end_id + 1):
            # Verificar el caché
            with self.db.connection() as conn:
                cursor = conn.execute('SELECT data, timestamp FROM anime_cache WHERE anime_id = ?', (anime_id,))
                row = cursor.fetchone()
                if row:
                    data_str, timestamp = row
                    timestamp = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
                    if datetime.now() - timestamp < timedelta(weeks=1):
                        try:
                            data_dict = ast.literal_eval(data_str)
                            keywords = ""
                            # Procesar campos complejos
                            synopsis = data_dict.get('synopsis', '')
                            if synopsis:
                                try:
                                    r.extract_keywords_from_text(synopsis)
                                    keywords = ', '.join(r.get_ranked_phrases())
                                except Exception as e:
                                    logger.error(f"Error extrayendo keywords: {e}")
                            
                            records.append({
                                'anime_id': anime_id,
                                'name': data_dict.get('title'),
                                'english_name': data_dict.get('title_english'),
                                'japanese_name': data_dict.get('title_japanese'),
                                'score': data_dict.get('score'),
                                'genres': ', '.join([g['name'] for g in data_dict.get('genres', [])]),
                                'keywords': keywords,
                                'type': data_dict.get('type'),
                                'episodes': data_dict.get('episodes'),
                                'aired': data_dict.get('aired', {}).get('string'),
                                'premiered': f"{data_dict.get('season', '')} {data_dict.get('year', '')}".strip(),
                                'status': data_dict.get('status'),
                                'producers': ', '.join([p['name'] for p in data_dict.get('producers', [])]),
                                'studios': ', '.join([s['name'] for s in data_dict.get('studios', [])]),
                                'source': data_dict.get('source'),
                                'duration': data_dict.get('duration'),
                                'rating': data_dict.get('rating'),
                                'rank': data_dict.get('rank'),
                                'popularity': data_dict.get('popularity'),
                                'favorites': data_dict.get('favorites'),
                                'scored_by': data_dict.get('scored_by'),
                                'members': data_dict.get('members')
                            })
                            continue
                        except Exception as e:
                            logger.warning(f"Error en caché para anime {anime_id}: {e}")

            # Si no está en caché, hacer solicitud
            data = self._fetch_anime(anime_id)
            if data:
                # Procesar sinopsis para keywords
                synopsis = data.get('synopsis', '')
                if synopsis:
                    try:
                        r.extract_keywords_from_text(synopsis)
                        keywords = ', '.join(r.get_ranked_phrases())
                    except Exception as e:
                        logger.error(f"Error extrayendo keywords: {e}")
                
                records.append({
                    'anime_id': anime_id,
                    'name': data.get('title'),
                    'english_name': data.get('title_english'),
                    'japanese_name': data.get('title_japanese'),
                    'score': data.get('score'),
                    'genres': ', '.join([g['name'] for g in data.get('genres', [])]),
                    'keywords': keywords,
                    'type': data.get('type'),
                    'episodes': data.get('episodes'),
                    'aired': data.get('aired', {}).get('string'),
                    'premiered': f"{data.get('season', '')} {data.get('year', '')}".strip(),
                    'status': data.get('status'),
                    'producers': ', '.join([p['name'] for p in data.get('producers', [])]),
                    'studios': ', '.join([s['name'] for s in data.get('studios', [])]),
                    'source': data.get('source'),
                    'duration': data.get('duration'),
                    'rating': data.get('rating'),
                    'rank': data.get('rank'),
                    'popularity': data.get('popularity'),
                    'favorites': data.get('favorites'),
                    'scored_by': data.get('scored_by'),
                    'members': data.get('members')
                })
                
                # Actualizar caché
                with self.db.connection() as conn:
                    conn.execute(
                        'INSERT OR REPLACE INTO anime_cache VALUES (?, ?, ?)',
                        (anime_id, json.dumps(data), datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                    )
                    conn.commit()

        df = pd.DataFrame(records)
        df.to_csv(buffer, index=False)
        buffer.seek(0)
        return buffer

