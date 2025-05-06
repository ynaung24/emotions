import os
import json
import pandas as pd
from typing import List, Tuple
import soundfile as sf
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
import tempfile
import random
import torch
from voice_processor import VoiceProcessor
import time
from tenacity import retry, stop_after_attempt, wait_exponential
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI()

def download_go_emotions(output_dir: str) -> str:
    """Download the Go Emotions dataset."""
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, 'goemotions_1.csv')
    
    if not os.path.exists(csv_path):
        print("Downloading Go Emotions dataset...")
        url = "https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/full_dataset/goemotions_1.csv"
        df = pd.read_csv(url)
        df.to_csv(csv_path, index=False)
        print(f"Dataset downloaded to {csv_path}")
    else:
        print(f"Dataset already exists at {csv_path}")
    
    return csv_path

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def generate_synthetic_audio(text: str, output_dir: str) -> str:
    """Generate synthetic audio from text using OpenAI's TTS API with retry logic."""
    try:
        # Add a small delay to avoid rate limits
        time.sleep(random.uniform(1, 2))
        
        # Create a unique filename
        filename = f"audio_{hash(text) % 10000}.mp3"
        output_path = os.path.join(output_dir, filename)
        
        # Generate audio using OpenAI's TTS API
        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text
        )
        
        # Save the audio file
        response.stream_to_file(output_path)
        
        return output_path
    except Exception as e:
        print(f"Error generating audio for text: {text[:50]}... Error: {str(e)}")
        raise

def process_dataset(csv_path: str, output_dir: str, use_tts: bool = True, max_samples: int = None) -> tuple:
    """Process the dataset and generate audio files."""
    print("Processing dataset...")
    
    # Create output directories
    temp_dir = os.path.join(output_dir, 'temp_audio')
    os.makedirs(temp_dir, exist_ok=True)
    
    # Read the dataset
    df = pd.read_csv(csv_path)
    
    # Limit samples if specified
    if max_samples:
        df = df.sample(n=min(max_samples, len(df)), random_state=42)
    
    # Split into train and validation sets (90/10)
    train_size = int(0.9 * len(df))
    train_df = df[:train_size]
    val_df = df[train_size:]
    
    train_data = []
    val_data = []
    
    # Process training data
    print("Processing training data...")
    for _, row in tqdm(train_df.iterrows(), total=len(train_df)):
        text = row['text']
        emotions = [1 if float(row[col]) > 0 else 0 for col in df.columns if col not in ['text', 'id', 'author', 'subreddit', 'link_id', 'parent_id', 'created_utc', 'rater_id']]
        
        if use_tts:
            try:
                audio_path = generate_synthetic_audio(text, temp_dir)
                train_data.append({
                    'text': text,
                    'audio_path': audio_path,
                    'emotions': emotions
                })
            except Exception as e:
                print(f"Failed to generate audio for sample {row.name} after retries. Using fallback method.")
                train_data.append({
                    'text': text,
                    'audio_path': None,
                    'emotions': emotions
                })
        else:
            train_data.append({
                'text': text,
                'audio_path': None,
                'emotions': emotions
            })
    
    # Process validation data
    print("Processing validation data...")
    for _, row in tqdm(val_df.iterrows(), total=len(val_df)):
        text = row['text']
        emotions = [1 if float(row[col]) > 0 else 0 for col in df.columns if col not in ['text', 'id', 'author', 'subreddit', 'link_id', 'parent_id', 'created_utc', 'rater_id']]
        
        if use_tts:
            try:
                audio_path = generate_synthetic_audio(text, temp_dir)
                val_data.append({
                    'text': text,
                    'audio_path': audio_path,
                    'emotions': emotions
                })
            except Exception as e:
                print(f"Failed to generate audio for sample {row.name} after retries. Using fallback method.")
                val_data.append({
                    'text': text,
                    'audio_path': None,
                    'emotions': emotions
                })
        else:
            val_data.append({
                'text': text,
                'audio_path': None,
                'emotions': emotions
            })
    
    return train_data, val_data

def generate_sine_wave(emotions: List[int], output_dir: str, idx: int) -> str:
    """Generate a sine wave audio file based on emotion intensity."""
    duration = 3.0  # seconds
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Generate audio based on emotion intensity
    base_freq = 220  # Base frequency
    emotion_intensity = sum(emotions) / len(emotions)
    frequency = base_freq * (1 + emotion_intensity)
    audio = 0.5 * np.sin(2 * np.pi * frequency * t)
    
    # Save audio file
    audio_path = os.path.join(output_dir, f"sample_{idx}.wav")
    sf.write(audio_path, audio, sample_rate)
    
    return audio_path

def train_model(train_data: List[dict], 
                val_data: List[dict],
                model_path: str,
                num_epochs: int = 10,
                batch_size: int = 32,
                learning_rate: float = 0.001):
    """Train the emotion detection model."""
    # Initialize voice processor
    processor = VoiceProcessor()
    
    # Filter out samples without audio paths
    train_data = [d for d in train_data if d['audio_path'] is not None]
    val_data = [d for d in val_data if d['audio_path'] is not None]
    
    if not train_data or not val_data:
        raise ValueError("No valid audio samples found for training")
    
    print(f"Training with {len(train_data)} samples")
    print(f"Validating with {len(val_data)} samples")
    
    # Train model
    print("Training emotion detection model...")
    processor.train_model(
        train_data=train_data,
        val_data=val_data,
        model_path=model_path,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate
    )

def main():
    # Set up directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    output_dir = os.path.join(data_dir, "go_emotions")
    models_dir = os.path.join(base_dir, "models")
    model_path = os.path.join(models_dir, "emotion_model.pt")
    
    # Create necessary directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # Check if model already exists
    if os.path.exists(model_path):
        print(f"Found existing model at {model_path}")
        print("Skipping training as model already exists.")
        return
    
    # Check for existing processed data
    train_data_path = os.path.join(output_dir, "train_data.json")
    val_data_path = os.path.join(output_dir, "val_data.json")
    
    if os.path.exists(train_data_path) and os.path.exists(val_data_path):
        print("Loading existing processed data...")
        with open(train_data_path, 'r') as f:
            train_data = json.load(f)
        with open(val_data_path, 'r') as f:
            val_data = json.load(f)
        print(f"Loaded {len(train_data)} training samples and {len(val_data)} validation samples")
    else:
        # Download and process dataset
        print("Downloading and processing dataset...")
        csv_path = download_go_emotions(output_dir)
        train_data, val_data = process_dataset(csv_path, output_dir, use_tts=True, max_samples=100)
        
        # Save processed data
        print("Saving processed data...")
        with open(train_data_path, "w") as f:
            json.dump(train_data, f)
        with open(val_data_path, "w") as f:
            json.dump(val_data, f)
    
    # Train model
    print("Training model...")
    try:
        train_model(
            train_data=train_data,
            val_data=val_data,
            model_path=model_path,
            num_epochs=10,
            batch_size=32,
            learning_rate=0.001
        )
        print(f"Model saved to {model_path}")
    except Exception as e:
        print(f"Error during training: {str(e)}")
        return
    
    print(f"Dataset processed and saved to {output_dir}")
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")

if __name__ == "__main__":
    main() 