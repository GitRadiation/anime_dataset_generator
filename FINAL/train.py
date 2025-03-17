import io
import logging
import random
import re
from functools import wraps
from typing import List, Set, Tuple

import pandas as pd
from config import DBConfig, LogConfig
from database import DatabaseManager
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine


def validate_dataframe(*columns):
    """Valida la presencia de columnas en un DataFrame"""
    def decorator(func):
        @wraps(func)
        def wrapper(df: pd.DataFrame, *args, **kwargs):
            missing = [col for col in columns if col not in df.columns]
            if missing:
                raise ValueError(f"Columnas faltantes: {missing}")
            return func(df, *args, **kwargs)
        return wrapper
    return decorator

# -------------------------------
# Configuración inicial de logging
# -------------------------------
LogConfig.setup(enable_colors=True)
logger = logging.getLogger(__name__)

# -------------------------------
# Data Manager
# -------------------------------
class DataManager:
    """SQLAlchemy y validación"""
    
    def __init__(self, db_config: DBConfig):
        self.db_config = db_config
        
    @DatabaseManager.error_handler
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Carga datos usando SQLAlchemy para compatibilidad."""
        engine = create_engine(
            f"postgresql://{self.db_config.user}:{self.db_config.password}@"
            f"{self.db_config.host}:{self.db_config.port}/{self.db_config.database}"
        )
        dfs = []
        for table in ["user_details", "anime_dataset", "user_score"]:
            dfs.append(pd.read_sql_table(table, engine))  # <-- Método compatible
        return tuple(dfs)

    @staticmethod
    def merge_data(*dfs: pd.DataFrame) -> pd.DataFrame:
        """Combina DataFrames con relaciones correctas."""
        if len(dfs) != 3:
            raise ValueError("Se requieren exactamente 3 DataFrames")
            
        # 1. Merge user_scores (N) <-> user_details (1)
        base = dfs[0].merge(  # user_scores
            dfs[1],           # user_details
            left_on="user_id", 
            right_on="mal_id", 
            how="inner",
            validate="many_to_one",  # <-- Corrección clave
            suffixes=("_score", "")
        )
        
        # 2. Merge resultante (N) <-> anime_dataset (1)
        merged = base.merge(
            dfs[2],  # anime_dataset
            on="anime_id", 
            how="inner",
            validate="many_to_one"
        )
        
        return merged.drop(columns=["user_id", "mal_id", "anime_id"], errors="ignore")

# -------------------------------
# Preprocesamiento optimizado
# -------------------------------
def clean_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza robusta con manejo de nulos."""
    str_cols = df.select_dtypes(include=["object", "string"]).columns
    df[str_cols] = df[str_cols].apply(lambda x: x.str.strip().replace(['\\N', 'nan'], pd.NA))
    return df.dropna(how='all', axis=1)

@LogConfig.log_execution
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline de preprocesamiento con validaciones y manejo de nombres de columnas."""
    # Convertir todos los nombres de columnas a strings y limpiarlos
    df.columns = [str(col).strip().replace(' ', '_').replace('-', '_').lower() for col in df.columns]
    df.columns = [
        re.sub(r'[^a-zA-Z0-9_]', '_', str(col)).lower()  # Reemplaza todo lo que no sea alfanumérico o _
        for col in df.columns
    ]
    # Validación de columnas necesarias (los ratings)
    required_cols = ["rating_x", "rating_y"]
    if not all(col in df.columns for col in required_cols):
        raise KeyError(f"Columnas requeridas faltantes: {required_cols}")

    # Discretización de ratings
    df['rating_class'] = pd.cut(
        df['rating_x'],
        bins=[0, 4.9, 6.9, 10],  # Rangos ajustados
        labels=['low', 'medium', 'high'],
        right=False
    ).cat.remove_unused_categories()  # Eliminar categorías no usadas

    # Eliminar filas sin clase de rating
    df = df.dropna(subset=['rating_class'])

    # One-hot encoding de la clase de rating
    rating_dummies = pd.get_dummies(df['rating_class'], prefix='rating', dtype=int)
    df = pd.concat([df.drop(columns=['rating_class']), rating_dummies], axis=1)

    # Procesamiento de columnas de texto (géneros y keywords)
    for col in ['keywords', 'genres']:
        if col in df.columns:
            # One-hot encoding con nombres limpios
            dummies = df[col].str.get_dummies(sep=',').add_prefix(f'{col}_')
            dummies = dummies.rename(columns=lambda x: x.replace(' ', '_').lower())
            
            # Filtrar columnas con baja frecuencia (<5%)
            dummies = dummies.loc[:, dummies.mean() > 0.05]
            
            # Concatenar con el DataFrame original
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)

    # Escalado numérico seguro
    numeric_cols = df.select_dtypes(include="number").columns.difference(rating_dummies.columns)
    if not numeric_cols.empty:
        # Asegurar que los nombres de columnas sean strings antes de escalar
        numeric_cols = [str(col) for col in numeric_cols]  # <-- Conversión explícita
        df[numeric_cols] = StandardScaler().fit_transform(df[numeric_cols])

    # Eliminar columnas con muchos nulos (>70%)
    df = df.dropna(axis=1, thresh=0.7 * len(df))

    # Eliminar filas con valores nulos
    df = df.dropna()

    return df

# -------------------------------
# Algoritmo Genético Corregido
# -------------------------------
class GeneticRuleMiner:
    """Manejo de edge cases"""
    
    def __init__(self, transactions: List[Tuple[Set[str], str]], target: str):
        self._validate_inputs(transactions, target)
        self.transactions = transactions
        self.target = target
        self.feature_map, self.tx_masks = self._precompute_features()
        
    def _precompute_features(self):
        features = sorted({f for antecedents, _ in self.transactions for f in antecedents})
        feature_map = {f: i for i, f in enumerate(features)}
        masks = [
            sum(1 << feature_map[f] for f in antecedents if f in feature_map)
            for antecedents, _ in self.transactions
        ]
        return feature_map, masks
    
    def _validate_inputs(self, transactions, target):
        if not transactions:
            raise ValueError("Lista de transacciones vacía")
        if not any(t[1] == target for t in transactions):
            raise ValueError(f"Target '{target}' no encontrado en transacciones")
        
    def calculate_fitness(self, individual: List[int]) -> float:
        individual_mask = sum(bit << idx for idx, bit in enumerate(individual))
        if not individual_mask:
            return 0.0
            
        matches = sum(
            (mask & individual_mask) == individual_mask and cons == self.target
            for mask, (_, cons) in zip(self.tx_masks, self.transactions)
        )
        
        total = sum((mask & individual_mask) == individual_mask for mask in self.tx_masks)
        if total == 0 or matches == 0:
            return 0.0  # <-- Evitar divisiones inválidas
            
        confidence = matches / total
        support = matches / len(self.transactions)
        
        # Balancear clases desequilibradas
        weight = 1.5 if self.target == 'rating_low' else 1.0
        return (0.6 * confidence + 0.4 * support * weight) * (0.95 ** sum(individual))

    @LogConfig.log_execution
    def run(self, generations: int = 500, pop_size: int = 300, mutation_rate: float = 0.25) -> Tuple[List[str], float]:
        """Ejecuta el algoritmo genético principal"""
        num_features = len(self.feature_map)
        population = self.initialize_population(pop_size, num_features)
        best_fitness = 0.0
        best_individual = []

        for gen in range(generations):
            fitness_scores = [self.calculate_fitness(ind) for ind in population]
            
            if not fitness_scores or all(score <= 0 for score in fitness_scores):
                logger.warning("Reinicializando población por fitness inválido")
                population = self.initialize_population(pop_size, num_features)
                continue

            current_best_idx = max(range(len(fitness_scores)), key=lambda i: fitness_scores[i])
            if fitness_scores[current_best_idx] > best_fitness:
                best_fitness = fitness_scores[current_best_idx]
                best_individual = population[current_best_idx]

            new_population = []
            # Implementación de elitismo (10% de la población)
            elite_size = int(pop_size * 0.1)
            elite = sorted(zip(population, fitness_scores), key=lambda x: x[1], reverse=True)[:elite_size]
            new_population.extend([ind for ind, _ in elite])
            
            for _ in range((pop_size - elite_size) // 2):
                parent1, parent2 = self.select_parents(population, fitness_scores)
                
                crossover_point = random.randint(1, num_features-1)
                child1 = parent1[:crossover_point] + parent2[crossover_point:]
                child2 = parent2[:crossover_point] + parent1[crossover_point:]
                
                for child in [child1, child2]:
                    for i in range(num_features):
                        if random.random() < mutation_rate:
                            child[i] ^= 1
                
                new_population.extend([child1, child2])
            
            population = new_population
            logger.info(f"Generación {gen}: Fitness={best_fitness:.4f}")

            # Reinicialización adaptativa si el fitness es bajo
            if gen % 20 == 0 and best_fitness < 0.3:
                logger.warning("Reinicialización adaptativa por bajo fitness")
                population = self.initialize_population(pop_size, num_features)

        best_antecedent = [
            feature for feature, idx in self.feature_map.items()
            if idx < len(best_individual) and best_individual[idx] == 1
        ]
        return best_antecedent, best_fitness

    @staticmethod
    def initialize_population(pop_size: int, num_features: int) -> List[List[int]]:
        return [[random.randint(0, 1) for _ in range(num_features)] for _ in range(pop_size)]

    def select_parents(self, population: List[List[int]], fitness_scores: List[float]) -> Tuple[List[int], List[int]]:
        """Selección por torneo binario"""
        parents = []
        for _ in range(2):
            candidates = random.sample(list(zip(population, fitness_scores)), 2)
            candidates.sort(key=lambda x: x[1], reverse=True)
            parents.append(candidates[0][0])
        return tuple(parents)


# -------------------------------
# Script Principal Actualizado
# -------------------------------
def main():
    db_config = DBConfig()
    
    try:
        data_manager = DataManager(db_config)
        user_details, anime_data, user_scores = data_manager.load_data()
        
        # Validación de datos cargados
        for df, name in [(user_details, "user_details"), (anime_data, "anime_dataset"), (user_scores, "user_score")]:
            if df.empty:
                raise ValueError(f"{name} está vacío")
            logger.info(f"Cargado {name}: {df.shape[0]} registros")
        
        merged_data = DataManager.merge_data(user_scores, user_details, anime_data)
        
        # Log de diagnóstico
        with io.StringIO() as buffer:
            merged_data.info(buf=buffer, show_counts=True)
            logger.debug(f"Metadata mergeada:\n{buffer.getvalue()}")
        
        df_clean = clean_string_columns(merged_data)
        df_preprocessed = preprocess_data(df_clean)
        
        # Generación de transacciones con logging
        transactions = []
        invalid_count = 0

        for _, row in df_preprocessed.iterrows():
            row_dict = row.to_dict()  # Convertir la fila a diccionario
            antecedents = {
                col for col in df_preprocessed.columns 
                if not col.startswith('rating_') and row_dict[col] != 0
            }
            
            consequent = next(
                (col for col in df_preprocessed.columns 
                if col.startswith('rating_') and row_dict[col] == 1),
                None
            )
            
            if consequent and antecedents:
                transactions.append((antecedents, consequent))
            else:
                invalid_count += 1
        
        # Balance de clases
        target_counts = pd.Series([t[1] for t in transactions]).value_counts()
        logger.info(f"Distribución de targets:\n{target_counts}")
        
        target = 'rating_high'
        miner = GeneticRuleMiner(transactions, target)
        best_rule, fitness = miner.run(mutation_rate=0.3, pop_size=500)
        
        logger.info(f"\n{' RESULTADO FINAL ':=^50}")
        logger.info(f"Regla: SI {', '.join(best_rule)} ENTONCES {target}")
        logger.info(f"Fitness: {fitness:.4f}")
        logger.info("=" * 50)

    except Exception as e:
        logger.critical(f"Error en ejecución: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()