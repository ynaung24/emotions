import logging
import sys
from pathlib import Path

from app.config.settings import LOG_SETTINGS

def setup_logger(name: str) -> logging.Logger:
    """
    Set up a logger with the specified name and configuration.
    
    Args:
        name (str): Name of the logger
        
    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(LOG_SETTINGS["level"])
        
        # Create formatters
        formatter = logging.Formatter(LOG_SETTINGS["format"])
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler
        log_file = Path(LOG_SETTINGS["log_file"])
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger
