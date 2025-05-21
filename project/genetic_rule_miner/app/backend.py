import logging
from io import StringIO
from typing import Optional, Set

import numpy as np
import pandas as pd
from genetic_rule_miner.bbdd_maker.anime_service import AnimeService
from genetic_rule_miner.bbdd_maker.details_service import DetailsService
from genetic_rule_miner.bbdd_maker.score_service import ScoreService
from genetic_rule_miner.bbdd_maker.user_service import UserService
from genetic_rule_miner.config import APIConfig
from genetic_rule_miner.data.preprocessing import preprocess_data

logger = logging.getLogger(__name__)

# ------------- Inicialización de servicios ------------- #
api_config = APIConfig()
user_service = UserService(api_config)
details_service = DetailsService(api_config)
score_service = ScoreService(api_config)
anime_service = AnimeService(api_config)


# ------------- Funciones principales ------------- #


def get_user_id_from_username(username: str) -> Optional[int]:
    if isinstance(username, int):
        user = user_service.get_user_by_id(username)
    else:
        user_id = user_service.get_user_id_by_username(username)
        if user_id is None:
            return None
        user = user_service.get_user_by_id(user_id)

    return user.get("mal_id") or user.get("user_id") if user else None


def get_user_profile(username: str) -> dict:
    details = details_service.get_user_detail(username)
    keys = [
        "mal_id",
        "username",
        "gender",
        "birthday",
        "location",
        "joined",
        "days_watched",
        "mean_score",
        "watching",
        "completed",
        "on_hold",
        "dropped",
        "plan_to_watch",
        "total_entries",
        "rewatched",
        "episodes_watched",
    ]
    return dict(zip(keys, details)) if details else {}


def get_all_known_anime_ids(
    username: str, user_id: int, score_service: ScoreService
) -> Set[int]:
    logger.info(
        f"Recolectando mal_ids del usuario '{username}' con rating 'high' y la mitad aleatoria de 'medium'..."
    )

    scraped = score_service.get_user_scores(username, user_id)
    if not scraped:
        return set()

    df = pd.DataFrame(scraped)
    df.columns = [f"col_{i}" for i in range(df.shape[1])]
    df = df.fillna("unknown").astype(str)

    df["score"] = pd.to_numeric(df["col_2"], errors="coerce")
    df["rating"] = pd.cut(
        df["score"],
        bins=[0, 6.9, 7.9, 10],
        labels=["low", "medium", "high"],
        right=False,
    ).astype(str)

    ids = set(df[df["rating"] == "high"]["col_2"].astype(int))

    medium_ids = df[df["rating"] == "medium"]["col_2"].astype(int).tolist()
    if medium_ids:
        np.random.seed(42)
        ids.update(
            np.random.choice(
                medium_ids, size=len(medium_ids) // 2, replace=False
            )
        )

    logger.info(f"✅ Total de animes únicos seleccionados: {len(ids)}")
    return ids


def get_all_relevant_anime_ids(username: str, user_id: int) -> Set[int]:
    ids = set()

    # Scores
    ids |= get_all_known_anime_ids(username, user_id, score_service)

    # Favoritos
    favs = user_service.get_user_favorites(username)
    ids |= {anime["mal_id"] for anime in favs.get("data", {}).get("anime", [])}

    # Actualizaciones recientes
    updates = user_service.get_user_updates(username)
    ids |= {
        entry["entry"]["mal_id"]
        for entry in updates.get("data", {}).get("anime", [])
    }

    # Historial
    history = user_service.get_user_history(username, type="anime")
    ids |= {h["entry"]["mal_id"] for h in history.get("data", [])}

    # Reseñas
    reviews = user_service.get_user_reviews(username)
    for review in reviews.get("data", []):
        ids |= {entry["mal_id"] for entry in review.get("entry", [])}

    return ids


def generate_user_anime_profile(
    username: str, user_id: int = 0
) -> tuple[pd.DataFrame, dict]:
    logger.info(f"🚀 Generando perfil de usuario de anime para '{username}'")

    user_profile = get_user_profile(username)
    user_id = (
        user_id
        or user_profile.get("mal_id")
        or user_profile.get("user_id")
        or 0
    )

    relevant_ids = get_all_relevant_anime_ids(username, user_id)
    anime_ids = relevant_ids

    if not anime_ids:
        logger.warning(
            f"No se encontraron animes asociados al usuario '{username}'."
        )
        return pd.DataFrame(), user_profile

    logger.info(f"🎬 Descargando información de {len(anime_ids)} animes...")
    anime_buffer = anime_service.get_anime_by_ids(list(anime_ids))
    anime_df = pd.read_csv(StringIO(anime_buffer.getvalue().decode("utf-8")))

    logger.info(f"✅ Perfil generado con éxito: {anime_df.shape[0]} animes")
    return anime_df, user_profile


def save_user_anime_profile_to_excel(username: str, output_path: str):
    anime_df, user_profile = generate_user_anime_profile(username)

    if not anime_df.empty:
        anime_df = preprocess_data(anime_df)
    else:
        logger.warning(
            f"No se procesó dataframe de animes para usuario '{username}' porque está vacío."
        )

    profile_df = pd.DataFrame([user_profile])
    profile_df = preprocess_data(profile_df)
    profile_df = profile_df.drop(
        columns=["location", "joined"], errors="ignore"
    )
    with pd.ExcelWriter(output_path) as writer:
        anime_df.to_excel(writer, sheet_name="animes", index=False)
        profile_df.to_excel(writer, sheet_name="perfil", index=False)


# --- Ejecución directa ---
if __name__ == "__main__":
    save_user_anime_profile_to_excel("Fador", "perfil_usuario.xlsx")
