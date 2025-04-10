import numpy as np
import sounddevice as sd
from typing import Generator, Optional, Tuple

from app.config.settings import AUDIO_SETTINGS
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class AudioStreamHandler:
    """Handler for audio stream capture from microphone."""
    
    def __init__(self):
        """Initialize audio stream handler."""
        self.stream = None
        self.is_recording = False
        self.audio_buffer = []
        logger.info("Initialized AudioStreamHandler")
    
    def start_stream(self) -> None:
        """Start the audio stream."""
        try:
            self.stream = sd.InputStream(
                samplerate=AUDIO_SETTINGS["sample_rate"],
                channels=AUDIO_SETTINGS["channels"],
                blocksize=AUDIO_SETTINGS["chunk_size"],
                callback=self._audio_callback
            )
            self.stream.start()
            self.is_recording = True
            logger.info("Started audio stream")
        except Exception as e:
            logger.error(f"Error starting audio stream: {str(e)}")
            raise
    
    def stop_stream(self) -> None:
        """Stop the audio stream."""
        if self.stream is not None:
            self.is_recording = False
            self.stream.stop()
            self.stream.close()
            self.stream = None
            logger.info("Stopped audio stream")
    
    def _audio_callback(self, indata: np.ndarray, frames: int,
                       time_info: dict, status: sd.CallbackFlags) -> None:
        """
        Callback function for audio stream.
        
        Args:
            indata (np.ndarray): Input audio data
            frames (int): Number of frames
            time_info (dict): Time information
            status (sd.CallbackFlags): Status flags
        """
        if status:
            logger.warning(f"Audio callback status: {status}")
        
        if self.is_recording:
            self.audio_buffer.append(indata.copy())
    
    def get_audio_chunk(self) -> Optional[np.ndarray]:
        """
        Get the latest audio chunk from the buffer.
        
        Returns:
            Optional[np.ndarray]: Audio chunk or None if buffer is empty
        """
        if not self.audio_buffer:
            return None
        
        # Concatenate all chunks in buffer
        audio_data = np.concatenate(self.audio_buffer, axis=0)
        self.audio_buffer = []  # Clear buffer
        
        return audio_data
    
    def stream_generator(self) -> Generator[np.ndarray, None, None]:
        """
        Generator for streaming audio chunks.
        
        Yields:
            np.ndarray: Audio chunk
        """
        while self.is_recording:
            audio_chunk = self.get_audio_chunk()
            if audio_chunk is not None:
                yield audio_chunk
    
    def __enter__(self):
        """Context manager entry."""
        self.start_stream()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_stream()
