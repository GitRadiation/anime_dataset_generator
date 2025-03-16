# config.py
import logging
import os
from dataclasses import dataclass
from functools import wraps
from typing import Optional

# Códigos ANSI para colores
COLOR_CODES = {
    'DEBUG': '\033[94m',     # Azul
    'INFO': '\033[92m',      # Verde
    'WARNING': '\033[93m',   # Amarillo
    'ERROR': '\033[91m',     # Rojo
    'CRITICAL': '\033[91m',  # Rojo brillante
    'RESET': '\033[0m'       # Resetear color
}

@dataclass
class APIConfig:
    """Configuración para servicios de API externos"""
    base_url: str = "https://api.jikan.moe/v4/"
    max_retries: int = 3
    timeout: float = 10.0
    request_delay: float = 0.35
    rate_limit: int = 3

    def __post_init__(self):
        """Validación de valores básicos"""
        if self.timeout < 0:
            raise ValueError("El timeout no puede ser negativo")

@dataclass
class DBConfig:
    """Configuración para conexiones de base de datos"""
    host: str = os.getenv("DB_HOST", "localhost")
    port: int = int(os.getenv("DB_PORT", 5432))
    database: str = os.getenv("DB_NAME", "mydatabase")
    user: str = os.getenv("DB_USER", "postgres")
    password: str = os.getenv("DB_PASS", "postgres")
    sqlite_file: Optional[str] = None

class ColoredFormatter(logging.Formatter):
    """Formateador personalizado con colores ANSI"""
    def format(self, record):
        color = COLOR_CODES.get(record.levelname, COLOR_CODES['INFO'])
        message = super().format(record)
        return f"{color}{message}{COLOR_CODES['RESET']}"

class LogConfig:
    """Configuración centralizada de logging con colores"""
    @staticmethod
    def log_execution(func):
        """Decorador para registrar inicio/fin de ejecución con colores"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = logging.getLogger(func.__module__)
            logger.info(f"{COLOR_CODES['INFO']}Iniciando {func.__name__}...{COLOR_CODES['RESET']}")
            result = func(*args, **kwargs)
            logger.info(f"{COLOR_CODES['INFO']}{func.__name__} completado.{COLOR_CODES['RESET']}")
            return result
        return wrapper
    
    @staticmethod
    def setup(
        level: int = logging.INFO,
        fmt: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt: str = "%Y-%m-%d %H:%M:%S",
        enable_colors: bool = True
    ):
        """Configura el sistema de logging con colores opcionales"""
        
        # Configurar handlers
        handlers = []
        
        # Handler para consola con colores
        console_handler = logging.StreamHandler()
        if enable_colors:
            console_formatter = ColoredFormatter(
                f"%(asctime)s - %(name)s - {COLOR_CODES['RESET']}%(levelname)s - %(message)s",
                datefmt=datefmt
            )
        else:
            console_formatter = logging.Formatter(fmt, datefmt=datefmt)
            
        console_handler.setFormatter(console_formatter)
        handlers.append(console_handler)
        
        # Handler para archivo (sin colores)
        file_handler = logging.FileHandler("app.log", encoding="utf-8")
        file_formatter = logging.Formatter(fmt, datefmt=datefmt)
        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)
        
        # Configuración básica
        logging.basicConfig(
            level=level,
            handlers=handlers
        )
        
        # Configurar niveles para dependencias
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("requests").setLevel(logging.WARNING)
        logging.getLogger("sqlalchemy").setLevel(logging.WARNING)
        
        # Añadir colores a los niveles
        if enable_colors:
            logging.addLevelName(logging.DEBUG, f"{COLOR_CODES['DEBUG']}DEBUG{COLOR_CODES['RESET']}")
            logging.addLevelName(logging.INFO, f"{COLOR_CODES['INFO']}INFO{COLOR_CODES['RESET']}")
            logging.addLevelName(logging.WARNING, f"{COLOR_CODES['WARNING']}WARNING{COLOR_CODES['RESET']}")
            logging.addLevelName(logging.ERROR, f"{COLOR_CODES['ERROR']}ERROR{COLOR_CODES['RESET']}")
            logging.addLevelName(logging.CRITICAL, f"{COLOR_CODES['CRITICAL']}CRITICAL{COLOR_CODES['RESET']}")
