import os
import yaml
from pathlib import Path
from typing import List, Tuple
from pydantic import BaseModel

class DataConfig(BaseModel):
    raw_data_path: str
    processed_data_path: str
    test_size: float
    random_state: int

class TfidfConfig(BaseModel):
    max_features: int
    ngram_range: Tuple[int, int]
    lowercase: bool

class ClassifierConfig(BaseModel):
    C: float
    max_iter: int
    class_weight: str
    random_state: int

class ModelConfig(BaseModel):
    model_path: str
    metrics_path: str
    model_type: str
    tfidf: TfidfConfig
    classifier: ClassifierConfig

class ApiConfig(BaseModel):
    host: str
    port: int
    debug: bool
    cors_origins: List[str]

class LoggingConfig(BaseModel):
    log_level: str
    log_file: str
    backup_count: int
    max_bytes: int

class AppConfig(BaseModel):
    project_name: str
    data: DataConfig
    model: ModelConfig
    api: ApiConfig
    logging: LoggingConfig

def get_project_root() -> Path:
    """Returns the project root directory absolute path."""
    return Path(__file__).resolve().parents[1]

def load_config(config_path: str = None) -> AppConfig:
    """Loads and validates configuration from a YAML file."""
    root = get_project_root()
    if config_path is None:
        config_path = root / "config" / "config.yaml"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        # Fallback to absolute if it exists, or relative to root
        config_path = root / config_path if not config_path.is_absolute() else config_path
        
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
        
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)
        
    # Resolve relative paths in data and model configs relative to project root
    config = AppConfig(**config_dict)
    
    # Helper to resolve relative path strings
    def resolve_path(p: str) -> str:
        path = Path(p)
        if not path.is_absolute():
            return str((root / path).resolve())
        return str(path.resolve())

    config.data.raw_data_path = resolve_path(config.data.raw_data_path)
    config.data.processed_data_path = resolve_path(config.data.processed_data_path)
    config.model.model_path = resolve_path(config.model.model_path)
    config.model.metrics_path = resolve_path(config.model.metrics_path)
    config.logging.log_file = resolve_path(config.logging.log_file)
    
    return config
