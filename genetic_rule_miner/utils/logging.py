"""Advanced logging configuration with color support and performance monitoring."""

import functools
import logging
import time
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, Optional

# ANSI color codes for terminal output
COLOR_CODES = {
    "DEBUG": "\033[94m",  # Blue
    "INFO": "\033[92m",  # Green
    "WARNING": "\033[93m",  # Yellow
    "ERROR": "\033[91m",  # Red
    "CRITICAL": "\033[1;91m",  # Bright Red
    "RESET": "\033[0m",  # Reset
}


@dataclass
class LogSettings:
    """Configuration settings for logging subsystem."""

    level: int = logging.INFO
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    enable_colors: bool = True
    log_file: Optional[str] = "application.log"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds color to log levels."""

    def __init__(self, fmt: str, datefmt: str, color_map: Dict[str, str]):
        super().__init__(fmt, datefmt)
        self.color_map = color_map

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with appropriate colors."""
        level_color = self.color_map.get(record.levelname, "")
        record.levelname = (
            f"{level_color}{record.levelname}{self.color_map['RESET']}"
        )
        return super().format(record)


class LogManager:
    """Centralized logging management with advanced features."""

    _configured: bool = False

    @classmethod
    def configure(cls, settings: Optional[LogSettings] = None) -> None:
        """Initialize logging system with specified settings."""
        if cls._configured:
            return

        settings = settings or LogSettings()

        # Configure handlers
        handlers = []

        # Console handler with colors
        console_handler = logging.StreamHandler()
        if settings.enable_colors:
            formatter = ColoredFormatter(
                settings.format, settings.date_format, COLOR_CODES
            )
        else:
            formatter = logging.Formatter(
                settings.format, settings.date_format
            )
        console_handler.setFormatter(formatter)
        handlers.append(console_handler)

        # File handler with rotation
        if settings.log_file:
            file_handler = RotatingFileHandler(
                settings.log_file,
                maxBytes=settings.max_file_size,
                backupCount=settings.backup_count,
                encoding="utf-8",
            )
            file_handler.setFormatter(
                logging.Formatter(settings.format, settings.date_format)
            )
            handlers.append(file_handler)

        # Basic configuration
        logging.basicConfig(level=settings.level, handlers=handlers)

        # Configure third-party loggers
        cls._configure_external_loggers()
        cls._configured = True

    @classmethod
    def _configure_external_loggers(cls) -> None:
        """Configure logging levels for common libraries."""
        for logger_name in ["urllib3", "requests", "sqlalchemy", "matplotlib"]:
            logging.getLogger(logger_name).setLevel(logging.WARNING)

    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """Get a configured logger instance."""
        if not cls._configured:
            cls.configure()
        return logging.getLogger(name)


def log_execution(func: Any) -> Any:
    """Decorator to log function entry and exit with timing."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        logger = LogManager.get_logger(func.__module__)
        start_time = time.perf_counter()

        logger.info(
            f"{COLOR_CODES['INFO']}Entering {func.__name__}..."
            f"{COLOR_CODES['RESET']}"
        )
        try:
            result = func(*args, **kwargs)
            duration = time.perf_counter() - start_time
            logger.info(
                f"{COLOR_CODES['INFO']}{func.__name__} completed in "
                f"{duration:.4f}s{COLOR_CODES['RESET']}"
            )
            return result
        except Exception as e:
            logger.error(
                f"{COLOR_CODES['ERROR']}{func.__name__} failed after "
                f"{time.perf_counter() - start_time:.4f}s: {e}"
                f"{COLOR_CODES['RESET']}",
                exc_info=True,
            )
            raise

    return wrapper
