"""
Configuration settings for the hotel recommendation system.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    host: str = "localhost"
    port: int = 5432
    database: str = "hotel_recommendation"
    username: str = "postgres"
    password: str = "password"
    
    @property
    def connection_string(self) -> str:
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"


@dataclass
class ModelConfig:
    """Machine learning model configuration."""
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"
    batch_size: int = 512
    embedding_dim: Optional[int] = None
    faiss_metric: str = "IP"
    normalize_embeddings: bool = True
    use_gpu: bool = True
    model_dir: str = "./models"


@dataclass
class ScrapingConfig:
    """Web scraping configuration."""
    chrome_driver_path: Optional[str] = None
    headless: bool = True
    timeout: int = 30
    max_retries: int = 3
    delay_between_requests: float = 1.0
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"


@dataclass
class ScoringConfig:
    """Scoring algorithm configuration."""
    default_weight: float = 1/16
    bayesian_constant: float = 100
    quality_weight: float = 0.8
    default_quality: float = 0.5
    global_score: float = 0.5
    quality_threshold: float = 0.0
    similarity_threshold: float = 0.90
    normalization_factor_base: float = 10.0


@dataclass
class PathConfig:
    """File and directory path configuration."""
    base_dir: Path = Path(__file__).parent.parent
    data_dir: Path = base_dir / "data"
    models_dir: Path = base_dir / "models"
    logs_dir: Path = base_dir / "logs"
    cache_dir: Path = base_dir / "cache"
    output_dir: Path = base_dir / "output"
    
    def __post_init__(self):
        """Create directories if they don't exist."""
        for directory in [self.data_dir, self.models_dir, self.logs_dir, 
                         self.cache_dir, self.output_dir]:
            directory.mkdir(parents=True, exist_ok=True)


class Settings:
    """Main settings class that combines all configuration."""
    
    def __init__(self):
        self.database = DatabaseConfig()
        self.model = ModelConfig()
        self.scraping = ScrapingConfig()
        self.scoring = ScoringConfig()
        self.paths = PathConfig()
        
        # Load environment variables
        self._load_env_vars()
    
    def _load_env_vars(self):
        """Load configuration from environment variables."""
        # Database settings
        self.database.host = os.getenv("DB_HOST", self.database.host)
        self.database.port = int(os.getenv("DB_PORT", self.database.port))
        self.database.database = os.getenv("DB_NAME", self.database.database)
        self.database.username = os.getenv("DB_USER", self.database.username)
        self.database.password = os.getenv("DB_PASSWORD", self.database.password)
        
        # Model settings
        self.model.model_name = os.getenv("MODEL_NAME", self.model.model_name)
        self.model.use_gpu = os.getenv("USE_GPU", "true").lower() == "true"
        self.model.model_dir = os.getenv("MODEL_DIR", self.model.model_dir)
        
        # Scraping settings
        self.scraping.headless = os.getenv("HEADLESS", "true").lower() == "true"
        self.scraping.timeout = int(os.getenv("SCRAPING_TIMEOUT", self.scraping.timeout))
    
    def get_path(self, filename: str) -> str:
        """Get full path for a file."""
        return str(self.paths.data_dir / filename)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary."""
        return {
            "database": {
                "host": self.database.host,
                "port": self.database.port,
                "database": self.database.database,
                "username": self.database.username,
                "password": "***"  # Hide password
            },
            "model": {
                "model_name": self.model.model_name,
                "batch_size": self.model.batch_size,
                "embedding_dim": self.model.embedding_dim,
                "faiss_metric": self.model.faiss_metric,
                "normalize_embeddings": self.model.normalize_embeddings,
                "use_gpu": self.model.use_gpu,
                "model_dir": self.model.model_dir
            },
            "scraping": {
                "headless": self.scraping.headless,
                "timeout": self.scraping.timeout,
                "max_retries": self.scraping.max_retries,
                "delay_between_requests": self.scraping.delay_between_requests,
                "user_agent": self.scraping.user_agent
            },
            "scoring": {
                "default_weight": self.scoring.default_weight,
                "bayesian_constant": self.scoring.bayesian_constant,
                "quality_weight": self.scoring.quality_weight,
                "default_quality": self.scoring.default_quality,
                "global_score": self.scoring.global_score,
                "quality_threshold": self.scoring.quality_threshold,
                "similarity_threshold": self.scoring.similarity_threshold,
                "normalization_factor_base": self.scoring.normalization_factor_base
            },
            "paths": {
                "base_dir": str(self.paths.base_dir),
                "data_dir": str(self.paths.data_dir),
                "models_dir": str(self.paths.models_dir),
                "logs_dir": str(self.paths.logs_dir),
                "cache_dir": str(self.paths.cache_dir),
                "output_dir": str(self.paths.output_dir)
            }
        }


# Global settings instance
settings = Settings()


# Legacy compatibility function
def get_path(filename: str) -> str:
    """Legacy function for backward compatibility."""
    return settings.get_path(filename)

