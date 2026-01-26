"""Utility functions for FinCommerce."""
import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional


def setup_logging(log_dir: str = "./logs", level: str = "INFO", 
                 max_bytes: int = 10485760, backup_count: int = 5) -> None:
    """Configure logging with file rotation and console output.
    
    Args:
        log_dir: Directory for log files
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        max_bytes: Maximum log file size before rotation
        backup_count: Number of backup files to keep
    """
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler with rotation
    file_handler = RotatingFileHandler(
        log_path / 'fincomerce.log',
        maxBytes=max_bytes,
        backupCount=backup_count
    )
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    logger.info(f"Logging configured: level={level}, dir={log_dir}")


def format_price(price: float, currency: str = "€") -> str:
    """Format price with currency symbol.
    
    Args:
        price: Price value
        currency: Currency symbol
        
    Returns:
        Formatted price string
    """
    return f"{currency}{price:,.2f}"


def calculate_savings(price: float, msrp: float) -> float:
    """Calculate savings amount.
    
    Args:
        price: Current price
        msrp: Original price
        
    Returns:
        Savings amount
    """
    return max(0, msrp - price)


def calculate_discount_pct(price: float, msrp: float) -> float:
    """Calculate discount percentage.
    
    Args:
        price: Current price
        msrp: Original price
        
    Returns:
        Discount percentage
    """
    if msrp <= 0:
        return 0.0
    return round((msrp - price) / msrp * 100, 1)


__all__ = [
    'setup_logging',
    'format_price',
    'calculate_savings',
    'calculate_discount_pct'
]
