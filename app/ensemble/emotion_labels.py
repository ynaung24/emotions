from typing import Dict, List, Set

# Ekman's 6 basic emotions plus neutral
EMOTIONS = {
    "angry": {
        "description": "Strong feeling of displeasure and hostility",
        "synonyms": {"furious", "enraged", "irritated", "mad", "outraged"},
        "intensity_levels": ["mild", "moderate", "intense"]
    },
    "disgust": {
        "description": "Strong feeling of dislike or repulsion",
        "synonyms": {"revolted", "repulsed", "nauseated", "aversion", "distaste"},
        "intensity_levels": ["mild", "moderate", "intense"]
    },
    "fear": {
        "description": "Feeling of being afraid or worried",
        "synonyms": {"scared", "terrified", "anxious", "worried", "nervous"},
        "intensity_levels": ["mild", "moderate", "intense"]
    },
    "happy": {
        "description": "Feeling of joy and contentment",
        "synonyms": {"joyful", "delighted", "pleased", "content", "cheerful"},
        "intensity_levels": ["mild", "moderate", "intense"]
    },
    "sad": {
        "description": "Feeling of unhappiness and sorrow",
        "synonyms": {"unhappy", "depressed", "down", "melancholy", "gloomy"},
        "intensity_levels": ["mild", "moderate", "intense"]
    },
    "surprise": {
        "description": "Feeling of being startled or amazed",
        "synonyms": {"astonished", "amazed", "startled", "stunned", "shocked"},
        "intensity_levels": ["mild", "moderate", "intense"]
    },
    "neutral": {
        "description": "Absence of strong emotions",
        "synonyms": {"calm", "balanced", "composed", "unemotional", "detached"},
        "intensity_levels": ["mild"]  # Neutral typically doesn't have intensity levels
    }
}

def get_emotion_description(emotion: str) -> str:
    """
    Get the description of an emotion.
    
    Args:
        emotion (str): Emotion name
        
    Returns:
        str: Emotion description
    """
    return EMOTIONS.get(emotion, {}).get("description", "Unknown emotion")

def get_emotion_synonyms(emotion: str) -> Set[str]:
    """
    Get synonyms for an emotion.
    
    Args:
        emotion (str): Emotion name
        
    Returns:
        Set[str]: Set of synonyms
    """
    return EMOTIONS.get(emotion, {}).get("synonyms", set())

def get_emotion_intensity_levels(emotion: str) -> List[str]:
    """
    Get intensity levels for an emotion.
    
    Args:
        emotion (str): Emotion name
        
    Returns:
        List[str]: List of intensity levels
    """
    return EMOTIONS.get(emotion, {}).get("intensity_levels", [])

def get_all_emotions() -> List[str]:
    """
    Get list of all emotions.
    
    Returns:
        List[str]: List of emotion names
    """
    return list(EMOTIONS.keys())

def is_valid_emotion(emotion: str) -> bool:
    """
    Check if an emotion is valid.
    
    Args:
        emotion (str): Emotion name
        
    Returns:
        bool: True if emotion is valid, False otherwise
    """
    return emotion in EMOTIONS

def get_emotion_hierarchy() -> Dict[str, Dict]:
    """
    Get the complete emotion hierarchy.
    
    Returns:
        Dict[str, Dict]: Complete emotion hierarchy
    """
    return EMOTIONS
