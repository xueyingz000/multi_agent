"""Logging utilities for IFC Semantic Agent."""

import sys
from pathlib import Path
from loguru import logger
from typing import Optional

from .config_loader import get_config


def setup_logger(config_override: Optional[dict] = None) -> None:
    """Setup logger with configuration.
    
    Args:
        config_override: Optional configuration override
    """
    config = get_config()
    
    # Get logging configuration
    log_config = config.get_section('logging')
    if config_override:
        log_config.update(config_override)
    
    # Remove default handler
    logger.remove()
    
    # Add console handler
    logger.add(
        sys.stderr,
        level=log_config.get('level', 'INFO'),
        format=log_config.get('format', '{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}'),
        colorize=True
    )
    
    # Add file handler if specified
    log_file = log_config.get('file')
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            log_file,
            level=log_config.get('level', 'INFO'),
            format=log_config.get('format', '{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}'),
            rotation=log_config.get('rotation', '1 day'),
            retention=log_config.get('retention', '30 days'),
            compression='zip'
        )


def get_logger(name: str = None):
    """Get logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    if name:
        return logger.bind(name=name)
    return logger


# Initialize logger on import
setup_logger()