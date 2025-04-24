import json
import logging
from datetime import datetime
from io import BytesIO
from typing import Optional

import pandas as pd
import requests
from config import APIConfig, LogConfig
from rake_nltk import Rake

LogConfig.setup()
logger = logging.getLogger(__name__)


class AnimeService:
    def __init__(self, config: APIConfig = APIConfig()):
        self.config = config

    def _fetch_anime(self, anime_id: int) -> Optional[dict]:
        """Obtiene los datos del anime desde la API."""
        for attempt in range(self.config.max_retries):
            try:
                response = requests.get(
                    f"{self.config.base_url}anime/{anime_id}",
                    timeout=self.config.timeout,
                )
                response.raise_for_status()
                return response.json().get("data")
            except (requests.RequestException, KeyError):
                continue
        return None

    def get_anime_data(self, start_id: int, end_id: int) -> BytesIO:
        buffer = BytesIO()
        records = []
        r = Rake()  # Inicializar RAKE una sola vez

        for anime_id in range(start_id, end_id + 1):
            # Hacer solicitud
            data = self._fetch_anime(anime_id)
            if data:
                # Procesar sinopsis para keywords
                synopsis = data.get("synopsis", "")
                keywords = ""
                if synopsis:
                    try:
                        r.extract_keywords_from_text(synopsis)
                        keywords = ", ".join(r.get_ranked_phrases())
                    except Exception as e:
                        logger.error(f"Error extrayendo keywords: {e}")

                records.append(
                    {
                        "anime_id": anime_id,
                        "name": data.get("title"),
                        "english_name": data.get("title_english"),
                        "japanese_name": data.get("title_japanese"),
                        "score": data.get("score"),
                        "genres": ", ".join(
                            [g["name"] for g in data.get("genres", [])]
                        ),
                        "keywords": keywords,
                        "type": data.get("type"),
                        "episodes": data.get("episodes"),
                        "aired": data.get("aired", {}).get("string"),
                        "premiered": f"{data.get('season', '')} {data.get('year', '')}".strip(),
                        "status": data.get("status"),
                        "producers": ", ".join(
                            [p["name"] for p in data.get("producers", [])]
                        ),
                        "studios": ", ".join(
                            [s["name"] for s in data.get("studios", [])]
                        ),
                        "source": data.get("source"),
                        "duration": data.get("duration"),
                        "rating": data.get("rating"),
                        "rank": data.get("rank"),
                        "popularity": data.get("popularity"),
                        "favorites": data.get("favorites"),
                        "scored_by": data.get("scored_by"),
                        "members": data.get("members"),
                    }
                )

        df = pd.DataFrame(records)
        df.to_csv(buffer, index=False)
        buffer.seek(0)
        return buffer
