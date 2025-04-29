import logging
import time
from io import BytesIO
from typing import Optional

import pandas as pd
import requests
from genetic_rule_miner.config import APIConfig
from genetic_rule_miner.utils.logging import LogManager
from rake_nltk import Rake

LogManager.configure()
logger = logging.getLogger(__name__)


class AnimeService:
    def __init__(self, config: APIConfig = APIConfig()):
        self.config = config
        logger.info("AnimeService initialized with config: %s", self.config)

    def _fetch_anime(self, anime_id: int) -> Optional[dict]:
        """Obtiene los datos del anime desde la API."""
        for attempt in range(self.config.max_retries):
            try:
                logger.debug(
                    "Fetching anime with ID %d (Attempt %d)",
                    anime_id,
                    attempt + 1,
                )
                response = requests.get(
                    f"{self.config.base_url}anime/{anime_id}",
                    timeout=self.config.timeout,
                )
                if response.status_code == 404:
                    logger.warning(
                        "Anime with ID %d not found (404). Skipping further attempts.",
                        anime_id,
                    )
                    return None
                response.raise_for_status()
                logger.debug("Successfully fetched anime with ID %d", anime_id)
                return response.json().get("data")
            except requests.RequestException as e:
                logger.warning(
                    "Failed to fetch anime with ID %d on attempt %d: %s",
                    anime_id,
                    attempt + 1,
                    e,
                )

                if attempt < self.config.max_retries - 1:
                    logger.debug("Waiting before the next attempt...")
                    time.sleep(2**attempt)  # Exponential backoff
        logger.error(
            "Failed to fetch anime with ID %d after %d attempts",
            anime_id,
            self.config.max_retries,
        )
        return None

    def get_anime_data(self, start_id: int, end_id: int) -> BytesIO:
        logger.info(
            "Starting to fetch anime data from ID %d to %d", start_id, end_id
        )
        buffer = BytesIO()
        records = []
        r = Rake()  # Inicializar RAKE una sola vez

        for anime_id in range(start_id, end_id + 1):
            logger.debug("Processing anime with ID %d", anime_id)
            data = self._fetch_anime(anime_id)
            if data:
                synopsis = data.get("synopsis", "")
                keywords = ""
                if synopsis:
                    try:
                        r.extract_keywords_from_text(synopsis)
                        keywords = ", ".join(r.get_ranked_phrases())
                        logger.debug(
                            "Extracted keywords for anime ID %d", anime_id
                        )
                    except Exception as e:
                        logger.error(
                            "Error extracting keywords for anime ID %d: %s",
                            anime_id,
                            e,
                        )

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
            else:
                logger.info("No data found for anime ID %d", anime_id)

        logger.info(
            "Finished fetching anime data. Total records: %d", len(records)
        )
        df = pd.DataFrame(records)
        df.to_csv(buffer, index=False)
        buffer.seek(0)
        logger.info("Anime data written to buffer")
        return buffer
