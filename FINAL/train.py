import io
import logging
import random
from functools import wraps
from typing import List, Set, Tuple

import pandas as pd
from config import DBConfig, LogConfig
from database import DatabaseManager
from sklearn.preprocessing import StandardScaler


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
# Data Manager usando DatabaseManager
# -------------------------------
class DataManager:
    """Clase adaptada para usar DatabaseManager desde database.py"""
    
    def __init__(self, db_config: DBConfig):
        self.db_manager = DatabaseManager(config=db_config)
        
    @DatabaseManager.error_handler
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Carga datos usando el contexto de DatabaseManager."""
        dfs = []
        with self.db_manager.connection() as conn:
            for table in ["user_details", "anime_dataset", "user_score"]:
                dfs.append(pd.read_sql(f"SELECT * FROM {table}", conn))
        return tuple(dfs)

    @staticmethod
    def merge_data(*dfs: pd.DataFrame) -> pd.DataFrame:
        """Combina DataFrames con validación mejorada."""
        if len(dfs) != 3:
            raise ValueError("Se requieren exactamente 3 DataFrames")
            
        merged = dfs[0].merge(
            dfs[1], 
            left_on="user_id", 
            right_on="mal_id", 
            how="inner",
            suffixes=("_user", "")
        ).merge(
            dfs[2], 
            on="anime_id", 
            how="inner"
        )
        return merged.drop(columns=["user_id", "mal_id", "anime_id"], errors="ignore")

# -------------------------------
# Preprocesamiento mejorado
# -------------------------------
def clean_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza robusta de columnas de texto."""
    str_cols = df.select_dtypes(include=["object", "string"]).columns
    df[str_cols] = df[str_cols].apply(lambda x: x.str.strip().replace('\\N', pd.NA))
    return df

@LogConfig.log_execution
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline de preprocesamiento con validaciones."""
    required_cols = ["rating_x", "rating_y"]
    if not all(col in df.columns for col in required_cols):
        raise KeyError(f"Columnas requeridas faltantes: {required_cols}")
    
    # Discretización mejorada
    df['rating_class'] = pd.cut(
        df['rating_x'],
        bins=[0, 5, 7, 10],
        labels=['low', 'medium', 'high'],
        include_lowest=True
    ).cat.add_categories('unknown').fillna('unknown')
    
    # One-hot encoding seguro
    rating_dummies = pd.get_dummies(df['rating_class'], prefix='rating', dummy_na=True)
    df = pd.concat([df.drop(columns=['rating_class']), rating_dummies], axis=1)
    
    # Procesamiento de features
    for col in ['keywords', 'genres']:
        if col in df.columns:
            df = pd.concat([df.drop(columns=[col]), df[col].str.get_dummies(sep=',')], axis=1)
    
    # Escalado numérico seguro con filtrado de baja varianza
    variance_threshold = 0.01  # <-- Nuevo: Filtrar características con baja varianza
    numeric_cols = df.select_dtypes(include="number").columns.difference(['rating_low', 'rating_medium', 'rating_high'])
    numeric_cols = [col for col in numeric_cols if df[col].var() > variance_threshold]  # <--
    
    if not numeric_cols.empty:
        df[numeric_cols] = StandardScaler().fit_transform(df[numeric_cols])
    
    return df.dropna(how='all', axis=1).dropna()

# -------------------------------
# Algoritmo Genético Mejorado
# -------------------------------
class GeneticRuleMiner:
    """Versión optimizada con manejo de datos robusto"""
    
    def __init__(self, transactions: List[Tuple[Set[str], str]], target: str):
        self._validate_inputs(transactions, target)
        self.transactions = transactions
        self.target = target
        self.feature_map, self.tx_masks = self._precompute_features()
        
    def _validate_inputs(self, transactions, target):
        if not transactions or not target:
            raise ValueError("Datos de entrada inválidos")
        if not any(cons == target for _, cons in transactions):
            raise ValueError(f"Target no encontrado: {target}")

    def _precompute_features(self):
        features = sorted({f for antecedents, _ in self.transactions for f in antecedents})
        feature_map = {f: i for i, f in enumerate(features)}
        masks = [
            sum(1 << feature_map[f] for f in antecedents if f in feature_map)
            for antecedents, _ in self.transactions
        ]
        return feature_map, masks
    
    def calculate_fitness(self, individual: List[int]) -> float:
        individual_mask = sum(bit << idx for idx, bit in enumerate(individual))
        if not individual_mask:
            return 0.0
            
        matches = sum(
            (mask & individual_mask) == individual_mask and cons == self.target
            for mask, (_, cons) in zip(self.tx_masks, self.transactions)
        )
        total = sum((mask & individual_mask) == individual_mask for mask in self.tx_masks)
        
        confidence = matches / total if total else 0
        support = matches / len(self.transactions)
        
        # Penalización por reglas largas (nuevo)
        rule_length_penalty = 0.9 ** len([bit for bit in individual if bit == 1])  # <--
        return (0.5 * confidence + 0.5 * support) * rule_length_penalty  # <--

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

            # Reinicialización adaptativa si el fitness es bajo (nuevo)
            if gen % 20 == 0 and best_fitness < 0.3:  # <--
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
        
        for df, name in [(user_details, "user_details"), (anime_data, "anime_dataset"), (user_scores, "user_score")]:
            if df.empty:
                raise ValueError(f"{name} está vacío")
            logger.info(f"Cargado {name}: {df.shape[0]} registros")
        
        merged_data = DataManager.merge_data(user_scores, user_details, anime_data)
        
        with io.StringIO() as buffer:
            merged_data.info(buf=buffer, show_counts=True)
            logger.debug(f"Metadata mergeada:\n{buffer.getvalue()}")
        
        df_clean = clean_string_columns(merged_data)
        df_preprocessed = preprocess_data(df_clean)  # Eliminado el muestreo del 10%
        
        # Generación de transacciones con validación reforzada
        transactions = []
        for _, row in df_preprocessed.iterrows():
            antecedents = {col for col in df_preprocessed.columns 
                          if not col.startswith('rating_') and row[col] != 0}
            consequent = next((col for col in df_preprocessed.columns 
                              if col.startswith('rating_') and row[col] == 1), None)
            if consequent and antecedents:
                transactions.append((antecedents, consequent))
        
        # Validación de balance de clases
        target_counts = pd.Series([t[1] for t in transactions]).value_counts()
        logger.info(f"Distribución de targets:\n{target_counts}")
        
        target = 'rating_high'
        miner = GeneticRuleMiner(transactions, target)
        best_rule, fitness = miner.run()  # Usando nuevos hiperparámetros por defecto
        
        logger.info(f"\n{' RESULTADO FINAL ':=^50}")
        logger.info(f"Regla: SI {', '.join(best_rule)} ENTONCES {target}")
        logger.info(f"Fitness: {fitness:.4f} (Confianza*0.7 + Soporte*0.3)")
        logger.info("=" * 50)

    except Exception as e:
        logger.critical(f"Error en ejecución: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()