import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from app.config.settings import MODEL_PATHS
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

def save_model(model: torch.nn.Module, path: Path, model_name: str) -> None:
    """
    Save a PyTorch model to disk.
    
    Args:
        model (torch.nn.Module): Model to save
        path (Path): Path to save the model
        model_name (str): Name of the model for logging
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), path)
        logger.info(f"Saved {model_name} model to {path}")
    except Exception as e:
        logger.error(f"Error saving {model_name} model: {str(e)}")
        raise

def load_model(model: torch.nn.Module, path: Path, model_name: str) -> torch.nn.Module:
    """
    Load a PyTorch model from disk.
    
    Args:
        model (torch.nn.Module): Model architecture to load weights into
        path (Path): Path to load the model from
        model_name (str): Name of the model for logging
        
    Returns:
        torch.nn.Module: Loaded model
    """
    try:
        if path.exists():
            model.load_state_dict(torch.load(path))
            logger.info(f"Loaded {model_name} model from {path}")
        else:
            logger.warning(f"No saved model found at {path} for {model_name}")
        return model
    except Exception as e:
        logger.error(f"Error loading {model_name} model: {str(e)}")
        raise

def save_json(data: Dict[str, Any], path: Path) -> None:
    """
    Save data to a JSON file.
    
    Args:
        data (Dict[str, Any]): Data to save
        path (Path): Path to save the JSON file
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)
        logger.info(f"Saved JSON data to {path}")
    except Exception as e:
        logger.error(f"Error saving JSON data: {str(e)}")
        raise

def load_json(path: Path) -> Optional[Dict[str, Any]]:
    """
    Load data from a JSON file.
    
    Args:
        path (Path): Path to load the JSON file from
        
    Returns:
        Optional[Dict[str, Any]]: Loaded data or None if file doesn't exist
    """
    try:
        if path.exists():
            with open(path, 'r') as f:
                data = json.load(f)
            logger.info(f"Loaded JSON data from {path}")
            return data
        else:
            logger.warning(f"No JSON file found at {path}")
            return None
    except Exception as e:
        logger.error(f"Error loading JSON data: {str(e)}")
        raise
