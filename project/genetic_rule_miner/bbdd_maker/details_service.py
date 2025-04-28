import csv
import logging
import time
from io import BytesIO, StringIO
from typing import List, Optional

import requests
from genetic_rule_miner.config import APIConfig
from genetic_rule_miner.utils.logging import LogManager

LogManager.configure()
logger = logging.getLogger(__name__)


class DetailsService:
    def __init__(self, config: APIConfig = APIConfig()):
        self.config = config

        # Optimización de parámetros
        self.batch_size = 3
        self.request_delay = 0.35
        self.batch_delay = 1.0
        self.max_retries = 3

        logger.info(
            "DetailsService inicializado con configuración predeterminada."
        )

    def get_user_details(self, usernames: List[str]) -> BytesIO:
        """Genera CSV con datos detallados optimizados"""
        logger.info(f"Inicio de procesamiento para {len(usernames)} usuarios.")
        buffer = StringIO()
        writer = csv.writer(buffer)
        writer.writerow(
            [
                "Mal ID",
                "Username",
                "Gender",
                "Birthday",
                "Location",
                "Joined",
                "Days Watched",
                "Mean Score",
                "Watching",
                "Completed",
                "On Hold",
                "Dropped",
                "Plan to Watch",
                "Total Entries",
                "Rewatched",
                "Episodes Watched",
            ]
        )

        total = len(usernames)
        start_time = time.time()

        for i in range(0, total, self.batch_size):
            batch = usernames[i : i + self.batch_size]
            logger.debug(
                f"Procesando batch {i // self.batch_size + 1}: {batch}"
            )
            batch_data = []

            for username in batch:
                if data := self._fetch_user_data(username):
                    writer.writerow(data)
                    batch_data.append(data)
                else:
                    logger.warning(
                        f"No se encontraron datos para el usuario: {username}"
                    )

            logger.info(f"Procesados {min(i+self.batch_size, total)}/{total}")
            self._handle_rate_limits(len(batch_data))

        logger.info(
            f"Tiempo total de procesamiento: {time.time()-start_time:.2f}s"
        )
        buffer.seek(0)
        return BytesIO(buffer.getvalue().encode("utf-8"))

    def _fetch_user_data(self, username: str) -> Optional[list]:
        """Obtiene datos de usuario con reintentos"""
        logger.debug(f"Solicitando datos para el usuario: {username}")
        for attempt in range(self.max_retries):
            try:
                response = requests.get(
                    f"https://api.jikan.moe/v4/users/{username}/full",
                    timeout=self.config.timeout,
                )

                if response.status_code == 200:
                    logger.debug(
                        f"Datos obtenidos exitosamente para {username}."
                    )
                    return self._parse_response(response.json())

                if response.status_code == 404:
                    logger.info(f"Usuario no encontrado: {username}")
                    return None

                logger.warning(
                    f"Intento {attempt+1} fallido para {username} (HTTP {response.status_code})."
                )
                time.sleep(2**attempt)

            except Exception as e:
                logger.error(
                    f"Error en intento {attempt+1} para {username}: {str(e)}"
                )
                time.sleep(2**attempt)

        logger.error(
            f"Fallaron todos los intentos para obtener datos de {username}."
        )
        return None

    def _parse_response(self, response: dict) -> list:
        """Extrae y estructura los datos relevantes"""
        logger.debug("Parseando respuesta de la API.")
        data = response.get("data", {})
        stats = data.get("statistics", {}).get("anime", {})

        return [
            data.get("mal_id"),
            data.get("username"),
            data.get("gender"),
            data.get("birthday"),
            data.get("location"),
            data.get("joined"),
            stats.get("days_watched"),
            stats.get("mean_score"),
            stats.get("watching"),
            stats.get("completed"),
            stats.get("on_hold"),
            stats.get("dropped"),
            stats.get("plan_to_watch"),
            stats.get("total_entries"),
            stats.get("rewatched"),
            stats.get("episodes_watched"),
        ]

    def _handle_rate_limits(self, batch_size: int):
        """Gestiona los límites de la API"""
        if batch_size > 0:
            logger.debug(
                f"Aplicando retraso de {self.batch_delay}s para respetar límites de la API."
            )
            time.sleep(self.batch_delay)
