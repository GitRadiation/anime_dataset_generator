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

        # Parameter optimization
        self.batch_size = 3
        self.request_delay = 0.35
        self.batch_delay = 1.0
        self.max_retries = 3

        logger.info("DetailsService initialized with default configuration.")

    def get_user_details(self, usernames: List[str]) -> BytesIO:
        """Generates a CSV with detailed optimized data"""
        logger.info(f"Starting processing for {len(usernames)} users.")
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
                f"Processing batch {i // self.batch_size + 1}: {batch}"
            )
            batch_data = []

            for username in batch:
                if data := self._fetch_user_data(username):
                    writer.writerow(data)
                    batch_data.append(data)
                else:
                    logger.warning(f"No data found for user: {username}")

            logger.info(f"Processed {min(i+self.batch_size, total)}/{total}")
            self._handle_rate_limits(len(batch_data))

        logger.info(f"Total processing time: {time.time()-start_time:.2f}s")
        buffer.seek(0)
        return BytesIO(buffer.getvalue().encode("utf-8"))

    def _fetch_user_data(self, username: str) -> Optional[list]:
        """Fetches user data with retries"""
        logger.debug(f"Requesting data for user: {username}")
        for attempt in range(self.max_retries):
            try:
                response = requests.get(
                    f"https://api.jikan.moe/v4/users/{username}/full",
                    timeout=self.config.timeout,
                )

                if response.status_code == 200:
                    logger.debug(
                        f"Successfully retrieved data for {username}."
                    )
                    return self._parse_response(response.json())

                if response.status_code == 404:
                    logger.info(
                        f"User not found: {username}. Skipping further attempts."
                    )
                    return None

                logger.warning(
                    f"Attempt {attempt+1} failed for {username} (HTTP {response.status_code})."
                )
                time.sleep(2**attempt)

            except Exception as e:
                logger.error(
                    f"Error on attempt {attempt+1} for {username}: {str(e)}"
                )
                time.sleep(2**attempt)

        logger.error(f"All attempts to fetch data for {username} failed.")
        return None

    def _parse_response(self, response: dict) -> list:
        """Extracts and structures relevant data"""
        logger.debug("Parsing API response.")
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
        """Handles API rate limits"""
        if batch_size > 0:
            logger.debug(
                f"Applying a delay of {self.batch_delay}s to respect API rate limits."
            )
            time.sleep(self.batch_delay)

    def get_user_detail(self, username: str) -> Optional[list]:
        """Obtiene los detalles de un solo usuario."""
        return self._fetch_user_data(username)

    def get_users_details(self, usernames: List[str]) -> list:
        """Obtiene los detalles de múltiples usuarios."""
        return [self._fetch_user_data(u) for u in usernames]
