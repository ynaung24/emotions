import librosa
import numpy as np
from typing import Tuple

from app.config.settings import AUDIO_SETTINGS
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class AudioProcessor:
    """Processor for audio feature extraction."""
    
    def __init__(self):
        """Initialize audio processor."""
        self.sample_rate = AUDIO_SETTINGS["sample_rate"]
        logger.info("Initialized AudioProcessor")
    
    def extract_mfcc(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Extract Mel-frequency cepstral coefficients (MFCCs).
        
        Args:
            audio_data (np.ndarray): Input audio data
            
        Returns:
            np.ndarray: MFCC features
        """
        try:
            # Ensure audio data is 1D
            if len(audio_data.shape) > 1:
                audio_data = audio_data.squeeze()
            
            # Extract MFCCs
            mfccs = librosa.feature.mfcc(
                y=audio_data,
                sr=self.sample_rate,
                n_mfcc=13,
                hop_length=512
            )
            
            # Normalize MFCCs
            mfccs = (mfccs - np.mean(mfccs, axis=1, keepdims=True)) / \
                   (np.std(mfccs, axis=1, keepdims=True) + 1e-8)
            
            return mfccs
        except Exception as e:
            logger.error(f"Error extracting MFCCs: {str(e)}")
            raise
    
    def extract_mel_spectrogram(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Extract Mel spectrogram.
        
        Args:
            audio_data (np.ndarray): Input audio data
            
        Returns:
            np.ndarray: Mel spectrogram
        """
        try:
            # Ensure audio data is 1D
            if len(audio_data.shape) > 1:
                audio_data = audio_data.squeeze()
            
            # Extract Mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio_data,
                sr=self.sample_rate,
                n_mels=128,
                hop_length=512
            )
            
            # Convert to log scale
            mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Normalize
            mel_spec = (mel_spec - np.mean(mel_spec)) / (np.std(mel_spec) + 1e-8)
            
            return mel_spec
        except Exception as e:
            logger.error(f"Error extracting Mel spectrogram: {str(e)}")
            raise
    
    def extract_features(self, audio_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract both MFCC and Mel spectrogram features.
        
        Args:
            audio_data (np.ndarray): Input audio data
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (MFCC features, Mel spectrogram)
        """
        mfccs = self.extract_mfcc(audio_data)
        mel_spec = self.extract_mel_spectrogram(audio_data)
        return mfccs, mel_spec
    
    def pad_features(self, features: np.ndarray, target_length: int) -> np.ndarray:
        """
        Pad or truncate features to target length.
        
        Args:
            features (np.ndarray): Input features
            target_length (int): Target length
            
        Returns:
            np.ndarray: Padded/truncated features
        """
        current_length = features.shape[1]
        
        if current_length > target_length:
            # Truncate
            return features[:, :target_length]
        elif current_length < target_length:
            # Pad with zeros
            pad_width = ((0, 0), (0, target_length - current_length))
            return np.pad(features, pad_width, mode='constant')
        else:
            return features
