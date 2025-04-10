import whisper
from typing import Optional

from app.config.settings import TEXT_SETTINGS
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class TranscriptionService:
    """Service for transcribing audio using Whisper."""
    
    def __init__(self):
        """Initialize the transcription service."""
        try:
            self.model = whisper.load_model("base")
            logger.info("Initialized Whisper transcription model")
        except Exception as e:
            logger.error(f"Error loading Whisper model: {str(e)}")
            raise
    
    def transcribe(self, audio_data: bytes) -> Optional[str]:
        """
        Transcribe audio data to text.
        
        Args:
            audio_data (bytes): Raw audio data
            
        Returns:
            Optional[str]: Transcribed text or None if transcription fails
        """
        try:
            # Transcribe audio
            result = self.model.transcribe(
                audio_data,
                language=TEXT_SETTINGS["language"],
                task="transcribe"
            )
            
            # Extract text from result
            text = result["text"].strip()
            
            if not text:
                logger.warning("Transcription produced empty text")
                return None
            
            logger.debug(f"Transcribed text: {text}")
            return text
            
        except Exception as e:
            logger.error(f"Error transcribing audio: {str(e)}")
            return None
    
    def transcribe_stream(self, audio_chunks: list) -> Optional[str]:
        """
        Transcribe a stream of audio chunks.
        
        Args:
            audio_chunks (list): List of audio chunks
            
        Returns:
            Optional[str]: Transcribed text or None if transcription fails
        """
        try:
            # Concatenate audio chunks
            audio_data = b"".join(audio_chunks)
            
            # Transcribe concatenated audio
            return self.transcribe(audio_data)
            
        except Exception as e:
            logger.error(f"Error transcribing audio stream: {str(e)}")
            return None
