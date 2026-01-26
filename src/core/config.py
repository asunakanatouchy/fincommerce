"""Core configuration and initialization."""
import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DEFAULT_CONFIG = CONFIG_DIR / "settings.yaml"
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from YAML file with environment variable overrides.
    
    Args:
        config_path: Path to config file. Defaults to settings.yaml
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid
    """
    path = Path(config_path or DEFAULT_CONFIG)
    
    if not path.exists():
        logger.error(f"Config file not found: {path}")
        raise FileNotFoundError(f"Config file not found: {path}")
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Override with environment variables
        config = _apply_env_overrides(config)
        
        logger.info(f"Loaded configuration from {path}")
        return config
        
    except yaml.YAMLError as e:
        logger.error(f"Invalid YAML in config file: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        raise


def _apply_env_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """Apply environment variable overrides to config.
    
    Args:
        config: Base configuration
        
    Returns:
        Configuration with environment overrides
    """
    # Qdrant overrides
    if os.getenv('QDRANT_HOST'):
        config['qdrant']['host'] = os.getenv('QDRANT_HOST')
    if os.getenv('QDRANT_PORT'):
        config['qdrant']['port'] = int(os.getenv('QDRANT_PORT'))
    if os.getenv('QDRANT_API_KEY'):
        config['qdrant']['api_key'] = os.getenv('QDRANT_API_KEY')
    
    # Logging overrides
    if os.getenv('LOG_LEVEL'):
        config['logging']['level'] = os.getenv('LOG_LEVEL')
    
    # API overrides
    if os.getenv('API_HOST'):
        config['api']['host'] = os.getenv('API_HOST')
    if os.getenv('API_PORT'):
        config['api']['port'] = int(os.getenv('API_PORT'))
    
    return config


# Load config at import time
try:
    CONFIG = load_config()
except Exception as e:
    logger.warning(f"Failed to load config during import: {e}")
    CONFIG = {}


__all__ = ['CONFIG', 'load_config', 'PROJECT_ROOT', 'DATA_DIR', 'LOGS_DIR']
