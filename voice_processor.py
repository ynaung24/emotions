import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import librosa
import soundfile as sf
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Tuple
import os
import json

class EmotionRNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, num_classes: int):
        super(EmotionRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate RNN
        out, _ = self.rnn(x, h0)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

class VoiceProcessor:
    def __init__(self, model_path: str = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.text_model = AutoModel.from_pretrained('bert-base-uncased').to(self.device)
        
        # Initialize RNN model
        self.rnn = EmotionRNN(
            input_size=768,  # BERT hidden size
            hidden_size=256,
            num_layers=2,
            num_classes=29  # Updated to match the number of emotion classes in the data
        ).to(self.device)
        
        if model_path and os.path.exists(model_path):
            self.rnn.load_state_dict(torch.load(model_path))
        
        self.rnn.eval()
        
        # Load emotion labels
        self.emotion_labels = [
            'admiration', 'amusement', 'anger', 'annoyance', 'apprehension', 'confusion',
            'contempt', 'contentment', 'desire', 'disappointment', 'disapproval', 'disgust',
            'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love',
            'nervousness', 'optimism', 'pride', 'realization', 'relief', 'remorse',
            'sadness', 'surprise', 'neutral'
        ]
    
    def preprocess_audio(self, audio_path: str) -> torch.Tensor:
        """Convert audio file to spectrogram features."""
        # Load audio file
        audio, sr = librosa.load(audio_path, sr=16000)
        
        # Extract mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Convert to tensor and normalize
        features = torch.FloatTensor(mel_spec_db).unsqueeze(0)  # [1, n_mels, time]
        features = (features - features.mean()) / features.std()
        
        # Ensure consistent size by padding or truncating
        target_size = 64  # Fixed size for audio features
        if features.size(2) > target_size:
            features = features[:, :, :target_size]
        else:
            padding = torch.zeros(features.size(0), features.size(1), target_size - features.size(2))
            features = torch.cat([features, padding], dim=2)
        
        # Project to match text feature dimension (768)
        features = features.permute(0, 2, 1)  # [1, time, n_mels]
        features = torch.nn.functional.linear(
            features,
            torch.randn(768, features.size(2), device=features.device)
        )  # [1, time, 768]
        
        return features.to(self.device)
    
    def preprocess_text(self, text: str) -> torch.Tensor:
        """Convert text to BERT embeddings."""
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=64)  # Changed to match audio length
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.text_model(**inputs)
        
        # Ensure consistent size
        features = outputs.last_hidden_state  # [1, seq_len, 768]
        target_size = 64  # Fixed size for text features
        if features.size(1) > target_size:
            features = features[:, :target_size, :]
        else:
            padding = torch.zeros(features.size(0), target_size - features.size(1), features.size(2))
            features = torch.cat([features, padding], dim=1)
        
        return features
    
    def detect_emotions(self, audio_path: str, text: str) -> Dict[str, float]:
        """Detect emotions from both audio and text."""
        # Process audio
        audio_features = self.preprocess_audio(audio_path)
        
        # Process text
        text_features = self.preprocess_text(text)
        
        # Combine features
        combined_features = torch.cat([audio_features, text_features], dim=1)
        
        # Get emotion predictions
        with torch.no_grad():
            outputs = self.rnn(combined_features)
            probabilities = torch.softmax(outputs, dim=1)
        
        # Convert to dictionary
        emotion_scores = {
            label: float(score)
            for label, score in zip(self.emotion_labels, probabilities[0])
        }
        
        return emotion_scores
    
    def save_model(self, path: str):
        """Save the trained model."""
        torch.save(self.rnn.state_dict(), path)
    
    @staticmethod
    def train_model(train_data: List[dict], 
                   val_data: List[dict],
                   model_path: str,
                   num_epochs: int = 10,
                   batch_size: int = 32,
                   learning_rate: float = 0.001):
        """Train the RNN model on the Go Emotions dataset."""
        processor = VoiceProcessor()
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(processor.rnn.parameters(), lr=learning_rate)
        
        # Training loop
        for epoch in range(num_epochs):
            processor.rnn.train()
            total_loss = 0
            
            for i in range(0, len(train_data), batch_size):
                batch = train_data[i:i+batch_size]
                
                # Process batch
                audio_features = []
                text_features = []
                labels = []
                
                for sample in batch:
                    try:
                        audio_feat = processor.preprocess_audio(sample['audio_path'])
                        text_feat = processor.preprocess_text(sample['text'])
                        
                        audio_features.append(audio_feat)
                        text_features.append(text_feat)
                        labels.append(sample['emotions'])
                    except Exception as e:
                        print(f"Error processing sample: {str(e)}")
                        continue
                
                if not audio_features:  # Skip batch if all samples failed
                    continue
                
                # Combine features
                audio_features = torch.cat(audio_features, dim=0)  # [batch_size, time, 768]
                text_features = torch.cat(text_features, dim=0)  # [batch_size, time, 768]
                
                # Ensure both features have the same sequence length
                assert audio_features.size(1) == text_features.size(1), f"Audio size {audio_features.size(1)} != Text size {text_features.size(1)}"
                
                combined_features = torch.cat([audio_features, text_features], dim=1)  # [batch_size, 2*time, 768]
                
                # Convert labels to tensor
                labels = torch.FloatTensor(labels).to(processor.device)
                
                # Forward pass
                outputs = processor.rnn(combined_features)
                loss = criterion(outputs, labels)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # Print epoch statistics
            avg_loss = total_loss / (len(train_data) // batch_size)
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
            
            # Save model
            processor.save_model(model_path) 