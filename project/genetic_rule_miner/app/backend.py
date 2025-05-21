import logging
from io import StringIO
from typing import Any, Dict, Set

import diskcache
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from genetic_rule_miner.bbdd_maker.anime_service import AnimeService
from genetic_rule_miner.bbdd_maker.details_service import DetailsService
from genetic_rule_miner.bbdd_maker.score_service import ScoreService
from genetic_rule_miner.bbdd_maker.user_service import UserService
from genetic_rule_miner.config import APIConfig
from genetic_rule_miner.data.preprocessing import preprocess_data

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# --- Inicialización de servicios ---
api_config = APIConfig()
user_service = UserService(api_config)
details_service = DetailsService(api_config)
score_service = ScoreService(api_config)
anime_service = AnimeService(api_config)

# --- Cache persistente ---
USER_CACHE_TTL = 30 * 60  # 30 minutos
ANIME_CACHE_TTL = 7 * 24 * 3600  # 1 semana

user_cache = diskcache.Cache(
    directory="./.cache/users", size_limit=500 * 1024 * 1024
)
anime_cache = diskcache.Cache(
    directory="./.cache/anime", size_limit=500 * 1024 * 1024
)


app = FastAPI(title="Anime Dataset API", version="1.0")


def cache_key_user(username: str):
    return f"user_profile:{username.lower()}"


def cache_key_anime_ids(username: str):
    return f"anime_ids:{username.lower()}"


def cache_key_anime_data(anime_ids: Set[int]):
    return f"animes_data:{'-'.join(map(str, sorted(anime_ids)))}"


def cache_key_anime_detail(anime_id: int):
    return f"anime_detail:{anime_id}"


# --- Funciones con cache manual ---


def get_user_profile_cached(username: str) -> Dict[str, Any]:
    key = cache_key_user(username)
    if key in user_cache:
        cached_value = user_cache[key]
        if isinstance(cached_value, dict):
            # Ensure keys are str (decode if bytes)
            if all(isinstance(k, str) for k in cached_value.keys()):
                return cached_value
            elif all(isinstance(k, bytes) for k in cached_value.keys()):
                return {k.decode("utf-8"): v for k, v in cached_value.items()}
            else:
                raise HTTPException(
                    status_code=500,
                    detail="Cached user profile has invalid key types",
                )
        else:
            raise HTTPException(
                status_code=500,
                detail="Cached user profile is not a valid dictionary",
            )

    profile = details_service.get_user_detail(username)
    if not profile or not isinstance(profile, (list, tuple)):
        raise HTTPException(
            status_code=404, detail=f"Usuario '{username}' no encontrado"
        )

    keys = [
        "mal_id",
        "username",
        "gender",
        "birthday",
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
    profile_dict = pd.DataFrame([dict(zip(keys, profile))])

    profile_dict = preprocess_data(profile_dict).to_dict(orient="records")[0]

    user_cache.set(key, profile_dict, expire=USER_CACHE_TTL)
    return profile_dict


def get_relevant_anime_ids_cached(username: str, user_id: int) -> Set[int]:
    key = cache_key_anime_ids(username)

    cached_value = user_cache.get(key)
    if isinstance(cached_value, set) and all(
        isinstance(i, int) for i in cached_value
    ):
        return cached_value

    ids: Set[int] = set()

    # Scores
    scraped = score_service.get_user_scores(username, int(user_id))
    if isinstance(scraped, list) and scraped:
        try:
            df = pd.DataFrame(scraped)
            df.columns = [f"col_{i}" for i in range(df.shape[1])]
            df = df.fillna("unknown").astype(str)

            # Limpiar col_1 para que solo tenga valores numéricos
            df["col_1"] = pd.to_numeric(df["col_1"], errors="coerce")
            df = df.dropna(subset=["col_1"])
            df["col_1"] = df["col_1"].astype(int)

            df["score"] = pd.to_numeric(df["col_2"], errors="coerce")
            df["rating"] = pd.cut(
                df["score"],
                bins=[0, 6.9, 7.9, 10],
                labels=["low", "medium", "high"],
                right=False,
            ).astype(str)

            high_ids = df[df["rating"] == "high"]["col_1"].tolist()
            ids.update(high_ids)

            medium_ids = df[df["rating"] == "medium"]["col_1"].tolist()
            if medium_ids:
                np.random.seed(42)
                sampled_ids = np.random.choice(
                    medium_ids, size=len(medium_ids) // 2, replace=False
                )
                ids.update(sampled_ids)
        except Exception as e:
            logger.warning(f"Error procesando scores para '{username}': {e}")

    # Favoritos
    favs = user_service.get_user_favorites(username)
    fav_ids = {
        anime.get("mal_id")
        for anime in favs.get("data", {}).get("anime", [])
        if isinstance(anime.get("mal_id"), int)
    }
    ids.update(fav_ids)

    # Actualizaciones recientes
    updates = user_service.get_user_updates(username)

    try:
        anime_list = updates.get("data", {}).get("anime", [])
    except AttributeError:
        # updates is probably a list
        anime_list = updates if isinstance(updates, list) else []

    update_ids = {
        entry.get("entry", {}).get("mal_id")
        for entry in anime_list
        if isinstance(entry.get("entry", {}).get("mal_id"), int)
    }
    ids.update(update_ids)

    # Historial
    history = user_service.get_user_history(username, type="anime")
    history_ids = {
        entry.get("entry", {}).get("mal_id")
        for entry in history.get("data", [])
        if isinstance(entry.get("entry", {}).get("mal_id"), int)
    }
    ids.update(history_ids)

    # Reseñas
    reviews = user_service.get_user_reviews(username)
    for review in reviews.get("data", []):
        review_ids = {
            entry.get("mal_id")
            for entry in review.get("entry", [])
            if isinstance(entry.get("mal_id"), int)
        }
        ids.update(review_ids)

    if not ids:
        logger.warning(
            f"No se encontraron animes relevantes para el usuario '{username}'"
        )

    user_cache.set(key, ids, expire=USER_CACHE_TTL)
    return ids


def get_anime_data_cached(anime_ids: Set[int]) -> pd.DataFrame:
    key = cache_key_anime_data(anime_ids)

    cached_value = anime_cache.get(key)
    if isinstance(cached_value, pd.DataFrame):
        return cached_value

    if not anime_ids:
        return pd.DataFrame()

    try:
        anime_buffer = anime_service.get_anime_by_ids(list(anime_ids))

        if not hasattr(anime_buffer, "getvalue"):
            raise ValueError(
                "El objeto devuelto por get_anime_by_ids no tiene 'getvalue'."
            )

        buffer_content = anime_buffer.getvalue()
        if not isinstance(buffer_content, (bytes, str)):
            raise ValueError("El contenido del buffer no es bytes ni str.")

        content_str = (
            buffer_content.decode("utf-8")
            if isinstance(buffer_content, bytes)
            else buffer_content
        )

        anime_df = pd.read_csv(StringIO(content_str))
        anime_df = preprocess_data(anime_df)

        anime_cache.set(key, anime_df, expire=ANIME_CACHE_TTL)
        return anime_df

    except Exception as e:
        logger.error(f"Error al obtener datos de anime: {e}")
        return pd.DataFrame()


# --- Endpoints FastAPI ---


@app.get("/users/{username}/profile")
def api_get_user_profile(username: str):
    return get_user_profile_cached(username)


@app.get("/users/{username}/anime_ids")
def api_get_user_anime_ids(username: str):
    profile = get_user_profile_cached(username)
    user_id = profile.get("mal_id") or profile.get("user_id") or 0
    anime_ids = get_relevant_anime_ids_cached(username, int(user_id))
    return {"anime_ids": list(anime_ids)}


@app.get("/users/{username}/anime_profile")
def api_get_user_anime_profile(username: str):
    profile = get_user_profile_cached(username)
    user_id = profile.get("mal_id") or profile.get("user_id") or 0
    anime_ids = get_relevant_anime_ids_cached(username, int(user_id))
    anime_df = get_anime_data_cached(anime_ids)
    return {
        "profile": profile,
        "anime_data": anime_df.to_dict(orient="records"),
        "count": len(anime_df),
    }


@app.get("/anime/{anime_id}")
def api_get_anime_detail(anime_id: int):
    key = cache_key_anime_detail(anime_id)
    if key in anime_cache:
        anime_df = anime_cache[key]
    else:
        anime_buffer = anime_service.get_anime_by_ids([anime_id])
        anime_df = pd.read_csv(
            StringIO(anime_buffer.getvalue().decode("utf-8"))
        )
        anime_df = preprocess_data(anime_df)
        anime_cache.set(key, anime_df, expire=ANIME_CACHE_TTL)

    if isinstance(anime_df, pd.DataFrame):
        anime_json = anime_df.to_dict(orient="records")
        if not anime_json:
            raise HTTPException(status_code=404, detail="Anime no encontrado")
        return anime_json[0]
    else:
        raise HTTPException(
            status_code=500, detail="Error procesando los datos del anime"
        )


@app.get("/users/{username}/full_profile")
def api_get_user_full_profile(username: str):
    profile = get_user_profile_cached(username)
    user_id = profile.get("mal_id") or profile.get("user_id") or 0
    anime_ids = get_relevant_anime_ids_cached(username, int(user_id))
    anime_df = get_anime_data_cached(anime_ids)
    return {
        "profile": profile,
        "anime_data": anime_df.to_dict(orient="records"),
        "anime_count": len(anime_df),
    }


@app.get("/health")
def health_check():
    return {"status": "ok"}
