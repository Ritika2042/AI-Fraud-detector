import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from src.config import LoggingConfig

def setup_logger(config: LoggingConfig) -> logging.Logger:
    """Sets up a structured logger with console and rotating file handlers."""
    log_file_path = Path(config.log_file)
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger("ai_scam_detector")
    # Prevent propagation to root logger to avoid duplicate console outputs
    logger.propagate = False
    
    # Set logging level
    level = getattr(logging, config.log_level.upper(), logging.INFO)
    logger.setLevel(level)
    
    # Avoid duplicate handlers if logger is already initialized
    if logger.handlers:
        return logger
        
    # Create formatter
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s [%(filename)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    logger.addHandler(console_handler)
    
    # Rotating File Handler
    file_handler = RotatingFileHandler(
        filename=str(log_file_path),
        maxBytes=config.max_bytes,
        backupCount=config.backup_count,
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    logger.addHandler(file_handler)
    
    return logger

def get_logger() -> logging.Logger:
    """Convenience function to get the logger if already configured, or default."""
    return logging.getLogger("ai_scam_detector")
