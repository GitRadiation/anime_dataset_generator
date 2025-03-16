import logging
import time
from io import BytesIO, StringIO

import pandas as pd
from anime_service import AnimeService
from config import APIConfig, DBConfig, LogConfig
from database import DatabaseManager
from details_service import DetailsService
from score_service import ScoreService
from user_service import UserService

# Configurar el logging
LogConfig.setup()
logger = logging.getLogger(__name__)

def clean_string_columns(df):
    """Elimina espacios en blanco en las columnas de tipo objeto"""
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip()
    return df

def preprocess_to_memory(df, columns_to_keep, integer_columns, float_columns=None):
    """Preprocesa y convierte tipos de datos según el esquema de la base de datos"""
    df = df.copy()  # Asegurarse de trabajar con una copia del DataFrame
    df.columns = df.columns.str.strip()
    df = clean_string_columns(df)
    
    missing_columns = [col for col in columns_to_keep if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Las siguientes columnas no se encontraron: {missing_columns}")
    
    df = df[columns_to_keep]
    
    for col in integer_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
    
    if float_columns:
        for col in float_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df.dropna(how='all', inplace=True)
    # Reemplazar NaN por None para mejor manejo
    df = df.where(pd.notnull(df), None)
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False, header=True, na_rep='\\N')
    csv_buffer.seek(0)
    return csv_buffer

def preprocess_user_score(df, columns_to_keep, integer_columns, valid_anime_ids):
    """Preprocesa el DataFrame de user_score y elimina filas con Anime ID no válido"""
    df = df.copy()  # Asegurarse de trabajar con una copia del DataFrame
    df.columns = df.columns.str.strip()
    df = clean_string_columns(df)
    
    missing_columns = [col for col in columns_to_keep if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Las siguientes columnas no se encontraron: {missing_columns}")
    
    df = df[columns_to_keep]
    
    df["anime_id"] = pd.to_numeric(df["anime_id"], errors="coerce")
    valid_set = set(valid_anime_ids)
    df = df[df["anime_id"].isin(valid_set)]
    
    for col in integer_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    
    df.dropna(how='all', inplace=True)
    # Reemplazar NaN por None para mejor manejo
    df = df.where(pd.notnull(df), None)
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False, header=True, na_rep='\\N')
    csv_buffer.seek(0)
    return csv_buffer

def main():
    # ==========================
    # Fase 1: Inicialización
    # ==========================
    # Inicializar configuraciones
    db_config = DBConfig()
    api_config = APIConfig()
    
    # Parámetros de reintento
    max_retries = 3
    retry_delay = 5  # en segundos

    # ==========================
    # Fase 2: Obtención de datos
    # ==========================
    for attempt in range(1, max_retries + 1):
        logger.info(f"🤔 Intento {attempt} de obtener datos base...")
        
        # 1. Obtener datos de anime
        anime_buffer = AnimeService(api_config).get_anime_data(1, 100)
        
        # 2. Generar lista de usuarios
        user_service = UserService(api_config)
        userlist_buffer = user_service.generate_userlist(start_id=1, end_id=100)
        
        # 3. Preparar datos para ScoreService
        userlist_df = pd.read_csv(userlist_buffer)
        userlist_df.rename(columns={'user_id': 'mal_id'}, inplace=True)
        modified_userlist_buffer = BytesIO()
        userlist_df.to_csv(modified_userlist_buffer, index=False)
        modified_userlist_buffer.seek(0)
        
        # 4. Obtener detalles de usuarios
        usernames = userlist_df['username'].dropna().tolist()
        details_service = DetailsService(api_config)
        details_buffer = details_service.get_user_details(usernames)
        
        # 5. Obtener puntuaciones
        score_service = ScoreService(api_config)
        scores_buffer = score_service.get_scores(modified_userlist_buffer)

        if all([anime_buffer, details_buffer, scores_buffer]):
            logger.info("✅ Obtención de datos base exitosa")
            break
        else:
            logger.warning(f"⚠️ Datos vacíos en intento {attempt}. Reintentando en {retry_delay} segundos...")
            time.sleep(retry_delay)
    else:
        logger.error("🚨 No se pudo obtener datos base después de varios intentos. Terminando el programa.")
        return

    # ==========================
    # Fase 3: Preprocesamiento y Carga de datos
    # ==========================
    # Inicializar conexión a la base de datos
    db = DatabaseManager(db_config)
    logger.info("✅ DatabaseManager cargado correctamente")

    try:
        with db.connection() as _:
            logger.info("📥 Iniciando carga de datos...")
            
            # Preprocesar y cargar anime_dataset
            anime_df = pd.read_csv(StringIO(anime_buffer.getvalue().decode('utf-8')))
            anime_buffer = preprocess_to_memory(
                anime_df,
                columns_to_keep=[
                    "anime_id", "score", "type", "episodes", "status",
                    "rank", "popularity", "favorites", "scored_by", "members"
                ],
                integer_columns=[
                    "anime_id", "episodes", "rank", "popularity",
                    "favorites", "scored_by", "members"
                ],
                float_columns=["score"]
            )
            db.copy_from_buffer(anime_buffer, "anime_dataset")

            # Preprocesar y cargar user_details
            details_df = pd.read_csv(StringIO(details_buffer.getvalue().decode('utf-8')))
            details_df.rename(columns={
                'Mal ID': 'mal_id',
                'Days Watched': 'days_watched',
                'Mean Score': 'mean_score',
                'Total Entries': 'total_entries',
                'Episodes Watched': 'episodes_watched',
                'Gender': 'gender',
                'Watching': 'watching',
                'Completed' : 'completed',
                'On Hold': 'on_hold',
                'Dropped': 'dropped',
                'Plan to Watch': 'plan_to_watch',
                'Rewatched': 'rewatched'
            }, inplace=True)
            
            details_buffer = preprocess_to_memory(
                details_df,
                columns_to_keep=[
                    "mal_id", "gender", "days_watched", "mean_score",
                    "watching", "completed", "on_hold", "dropped",
                    "plan_to_watch", "total_entries", "rewatched", "episodes_watched"
                ],
                integer_columns=[
                    "mal_id", "watching", "completed", "on_hold",
                    "dropped", "plan_to_watch", "total_entries",
                    "rewatched", "episodes_watched"
                ],
                float_columns=["days_watched", "mean_score"]
            )
            db.copy_from_buffer(details_buffer, "user_details")

            # Preprocesar y cargar user_score
            scores_df = pd.read_csv(StringIO(scores_buffer.getvalue().decode('utf-8')))
            scores_df.rename(columns={
                'User ID': 'user_id',
                'Anime ID': 'anime_id',
                'Score': 'rating'
            }, inplace=True)
            logging.info(scores_df)
            
            valid_anime_ids = anime_df["anime_id"].dropna().unique().tolist()
            scores_buffer = preprocess_user_score(
                scores_df,
                columns_to_keep=["user_id", "anime_id", "rating"],
                integer_columns=["user_id", "anime_id", "rating"],
                valid_anime_ids=valid_anime_ids
            )
            db.copy_from_buffer(scores_buffer, "user_score")
            
        logger.info("✅ Carga completada exitosamente")
    except Exception as e:
        logger.error(f"🚨 Error crítico: {e}")
        raise

if __name__ == "__main__":
    main()