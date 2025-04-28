import csv
import json
import logging
import random
import time
from io import BytesIO, StringIO
from typing import List, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup
from genetic_rule_miner.config import APIConfig
from genetic_rule_miner.utils.logging import LogManager

LogManager.configure()
logger = logging.getLogger(__name__)


class ScoreService:
    def __init__(self, config: APIConfig = APIConfig()):
        self.config = config

        # Configuración de scraping
        self.status_code = 7  # Completed anime
        self.batch_size = 50
        self.min_delay = 90
        self.max_delay = 120
        logger.info(
            "ScoreService inicializado con configuración predeterminada."
        )

    def _process_batch(self, users_batch: List[dict]) -> List[list]:
        """Procesa un lote de usuarios con manejo de errores"""
        logger.info(f"Procesando lote de {len(users_batch)} usuarios.")
        batch_data = []
        for user in users_batch:
            try:
                if data := self._scrape_user_scores(
                    user["username"], user["mal_id"]
                ):
                    batch_data.extend(data)
                    logger.info(f"Datos obtenidos para {user['username']}.")
                else:
                    logger.warning(f"Sin datos para {user['username']}.")
            except Exception as e:
                logger.error(f"Error procesando {user['username']}: {str(e)}")
        logger.info(
            f"Lote procesado con {len(batch_data)} registros obtenidos."
        )
        return batch_data

    def _scrape_user_scores(
        self, username: str, user_id: int
    ) -> Optional[List[list]]:
        """Lógica principal de scraping con doble estructura de tablas"""
        logger.debug(f"Iniciando scraping para usuario: {username}.")
        try:
            url = f"https://myanimelist.net/animelist/{username}?status={self.status_code}"
            response = requests.get(url, timeout=self.config.timeout)

            if response.status_code != 200:
                logger.warning(
                    f"Respuesta HTTP {response.status_code} para {username}."
                )
                return None

            soup = BeautifulSoup(response.content, "html.parser")
            logger.debug(f"Contenido HTML obtenido para {username}.")
            return self._parse_modern_table(
                soup, user_id, username
            ) or self._parse_legacy_tables(soup, user_id, username)

        except Exception as e:
            logger.error(f"Error de scraping en {username}: {str(e)}")
            return None

    def _parse_modern_table(self, soup, user_id, username):
        """Maneja la tabla moderna con data-items"""
        logger.debug(f"Intentando parsear tabla moderna para {username}.")
        if table := soup.find("table", {"data-items": True}):
            try:
                data = [
                    [
                        user_id,
                        username,
                        item["anime_id"],
                        item["anime_title"],
                        item["score"],
                    ]
                    for item in json.loads(table["data-items"])
                    if item["score"] > 0
                ]
                logger.debug(
                    f"Tabla moderna parseada con {len(data)} registros."
                )
                return data
            except json.JSONDecodeError:
                logger.warning(
                    f"Error decodificando JSON en tabla moderna para {username}."
                )
        return None

    def _parse_legacy_tables(self, soup, user_id, username):
        """Maneja la estructura antigua de tablas"""
        logger.debug(f"Intentando parsear tablas antiguas para {username}.")
        scores = []
        for table in soup.find_all(
            "table",
            {
                "border": "0",
                "cellpadding": "0",
                "cellspacing": "0",
                "width": "100%",
            },
        ):
            for row in table.find_all("tr"):
                cells = row.find_all("td")
                if len(cells) >= 5:
                    anime_data = self._extract_anime_data(cells[1])
                    score_data = self._extract_score_data(cells[2])
                    if anime_data and score_data:
                        scores.append(
                            [
                                user_id,
                                username,
                                anime_data[0],
                                anime_data[1],
                                score_data,
                            ]
                        )
        logger.debug(f"Tablas antiguas parseadas con {len(scores)} registros.")
        return scores if scores else None

    def _extract_anime_data(self, cell):
        """Extrae ID y título del anime de una celda"""
        if link := cell.find("a", class_="animetitle"):
            return (link["href"].split("/")[2], link.find("span").text.strip())
        return None

    def _extract_score_data(self, cell):
        """Extrae la puntuación de una celda"""
        if score_label := cell.find("span", class_="score-label"):
            return (
                int(score_label.text.strip())
                if score_label.text.strip() != "-"
                else None
            )
        return None

    def get_scores(self, users_buffer: BytesIO) -> BytesIO:
        """Genera CSV de scores procesando lotes"""
        logger.info("Iniciando procesamiento de usuarios para generar CSV.")
        users_df = pd.read_csv(users_buffer)
        buffer = StringIO()
        writer = csv.writer(buffer)
        writer.writerow(
            ["User ID", "Username", "Anime ID", "Anime Title", "Score"]
        )

        total_users = len(users_df)
        logger.info(f"Total de usuarios a procesar: {total_users}.")
        processed = 0

        for i in range(0, total_users, self.batch_size):
            batch = users_df.iloc[i : i + self.batch_size].to_dict("records")
            logger.info(f"Procesando lote {i // self.batch_size + 1}.")
            batch_data = self._process_batch(batch)

            if batch_data:
                writer.writerows(batch_data)
                processed += len(batch)
                logger.info(
                    f"Procesado: {processed}/{total_users} ({processed/total_users:.1%})."
                )

            if i + self.batch_size < total_users:
                delay = random.randint(self.min_delay, self.max_delay)
                logger.info(f"Esperando {delay}s para siguiente lote...")
                time.sleep(delay)

        logger.info("Procesamiento completo. Generando archivo CSV.")
        buffer.seek(0)
        return BytesIO(buffer.getvalue().encode("utf-8"))
