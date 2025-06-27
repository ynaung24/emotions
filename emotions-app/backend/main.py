from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Union
import os
import numpy as np
import tempfile
import speech_recognition as sr
import soundfile as sf
from pydub import AudioSegment
import io
import numpy as np
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import evaluation modules
from evaluate.evaluate import evaluate_response
# from voice_processor import VoiceProcessor  # Uncomment if needed

app = FastAPI(
    title="Emotion Detection API",
    description="API for detecting emotions from text and voice inputs",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class TextRequest(BaseModel):
    text: str
    question: str

class AudioRequest(BaseModel):
    audio: bytes
    question: str

class EvaluationResponse(BaseModel):
    score: float
    metrics: Dict[str, float]
    feedback: Union[str, List[str]]
    emotions: Optional[Dict[str, float]] = None
    
    class Config:
        json_encoders = {
            np.float32: lambda v: float(v)  # Convert numpy float32 to Python float
        }

# Import the model initialization function
from evaluate.evaluate import initialize_models, load_corpus

# Initialize models and corpus
voice_processor = None  # Initialize if voice processing is needed
corpus = load_corpus()

# Pre-load models to improve first-request performance
print("Loading evaluation models...")
model, nlp = initialize_models('default')
print("Models loaded successfully.")

@app.get("/")
async def root():
    return {"message": "Emotion Detection API is running"}

@app.get("/questions")
async def get_questions():
    try:
        # Return the list of questions from the corpus
        if not corpus or "questions" not in corpus:
            raise ValueError("Corpus not properly loaded or missing questions")
            
        questions = [q["text"] for q in corpus["questions"]]
        logger.info(f"Successfully loaded {len(questions)} questions from corpus")
        return {"questions": questions}
    except Exception as e:
        logger.error(f"Error loading questions from corpus: {str(e)}")
        # Fallback to default questions if there's an error
        default_questions = [
            "Tell me about yourself",
            "What are your strengths and weaknesses?",
            "Why do you want to work here?",
            "Where do you see yourself in 5 years?",
            "Tell me about a challenge you faced and how you overcame it"
        ]
        logger.info("Using default questions due to error")
        return {"questions": default_questions}

@app.post("/evaluate/text", response_model=EvaluationResponse)
async def evaluate_text(request: TextRequest):
    try:
        # Log the evaluation request
        logger.info(f"=== New Evaluation Request ===")
        logger.info(f"Question: {request.question}")
        logger.info(f"Response length: {len(request.text)} characters")
        logger.info(f"Response preview: {request.text[:200]}...")
        
        # Call the evaluation function
        evaluation_result = evaluate_response(
            response=request.text,
            question=request.question,
            role_context={"role": "software engineer"}  # Default role, can be customized
        )
        
        logger.info("=== Raw Evaluation Result ===")
        logger.info(f"Score: {evaluation_result.get('score', 'N/A')}")
        logger.info(f"Metrics: {evaluation_result.get('metrics', {})}")
        logger.info(f"Feedback: {evaluation_result.get('feedback', 'No feedback')[:200]}...")
        
        # Get the feedback text and ensure it's a string
        feedback_text = evaluation_result.get("feedback", "No feedback available.")
        
        # If feedback is a list, join it into a single string
        if isinstance(feedback_text, list):
            feedback_text = " ".join(str(item) for item in feedback_text)
        
        # Convert numpy types to Python native types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_numpy_types(item) for item in obj]
            return obj
            
        # Convert the evaluation result to the expected response format
        evaluation = convert_numpy_types({
            "score": evaluation_result.get("score", 0),  # Already in 0-100 scale
            "metrics": {
                "relevance": evaluation_result.get("metrics", {}).get("relevance", 0),
                "clarity": evaluation_result.get("metrics", {}).get("clarity", 0),
                "completeness": evaluation_result.get("metrics", {}).get("completeness", 0),
                "confidence": 0,  # Not currently calculated in evaluate_response
                "conciseness": 75  # Default value since it's not calculated yet
            },
            "feedback": feedback_text  # Keep as a single string
            # Removed emotions field as it's not relevant for text evaluation
        })
        
        logger.info("=== Final Evaluation Response ===")
        logger.info(f"Final score: {evaluation['score']}")
        logger.info(f"Metrics: {evaluation['metrics']}")
        
        return evaluation
    except Exception as e:
        logger.error(f"Error in evaluate_text: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/evaluate/audio", response_model=EvaluationResponse)
async def evaluate_audio(file: UploadFile = File(...), question: str = Form(...)):
    temp_audio = None
    try:
        logger.info(f"Received audio evaluation request for question: {question}")
        
        # Read the file content
        file_content = await file.read()
        logger.info(f"Received audio file size: {len(file_content)} bytes")
        
        # Save uploaded file to a temporary file with explicit WAV format
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
            temp_audio_path = temp_audio.name
            try:
                # First try to read as is (in case it's already WAV)
                temp_audio.write(file_content)
                temp_audio.flush()
                
                # Try to open with pydub to check/convert format
                try:
                    audio = AudioSegment.from_file(temp_audio_path)
                    # Ensure it's in the right format (16-bit PCM, mono, 16kHz is typical for speech recognition)
                    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
                    audio.export(temp_audio_path, format='wav')
                    logger.info("Successfully converted audio to 16kHz 16-bit mono WAV")
                except Exception as conv_error:
                    logger.warning(f"Could not convert audio, trying raw: {str(conv_error)}")
                    # If conversion fails, try with the original file
                    temp_audio.seek(0)
                    temp_audio.truncate()
                    temp_audio.write(file_content)
                    temp_audio.flush()
                    
            except Exception as e:
                logger.error(f"Error processing audio file: {str(e)}", exc_info=True)
                raise HTTPException(status_code=400, detail=f"Invalid audio file: {str(e)}")
        
        # Verify the file exists and has content
        if not os.path.exists(temp_audio_path) or os.path.getsize(temp_audio_path) == 0:
            raise HTTPException(status_code=400, detail="Failed to process audio file")
            
        logger.info(f"Audio file saved to {temp_audio_path}, size: {os.path.getsize(temp_audio_path)} bytes")
        
        # Transcribe audio
        recognizer = sr.Recognizer()
        temp_audio_path = temp_audio.name if hasattr(temp_audio, 'name') else None
        
        try:
            logger.info("Attempting to transcribe audio...")
            
            # Read the audio file with a timeout
            with sr.AudioFile(temp_audio_path) as source:
                # Adjust for ambient noise for better recognition
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio_data = recognizer.record(source)
                
                logger.info("Sending audio to Google Speech Recognition...")
                try:
                    # Use recognize_google with explicit language and show_all=False for better error handling
                    text = recognizer.recognize_google(
                        audio_data,
                        language="en-US",
                        show_all=False
                    )
                    logger.info(f"Successfully transcribed text: {text}")
                    
                except sr.UnknownValueError:
                    logger.error("Google Speech Recognition could not understand the audio")
                    raise HTTPException(
                        status_code=400,
                        detail="Could not understand the audio. Please ensure you're speaking clearly and try again."
                    )
                except sr.RequestError as e:
                    logger.error(f"Could not request results from Google Speech Recognition service; {e}")
                    raise HTTPException(
                        status_code=503,
                        detail=f"Speech recognition service is currently unavailable. Please try again later. Error: {str(e)}"
                    )
                
                # Get evaluation using the same function as text evaluation
                evaluation_result = evaluate_response(
                    response=text,
                    question=question,
                    role_context={"role": "software engineer"}
                )
                
                # Add emotion scores from voice analysis (placeholder)
                emotion_scores = {
                    "confidence": 0.7,
                    "enthusiasm": 0.6,
                    "professionalism": 0.8,
                    "clarity": 0.75
                }
                
                # Convert the evaluation result to the expected response format
                feedback_text = evaluation_result.get("feedback", "No feedback available.")
                if isinstance(feedback_text, list):
                    feedback_text = " ".join(str(item) for item in feedback_text)
                
                evaluation = {
                    "score": evaluation_result.get("score", 0),
                    "metrics": {
                        "relevance": evaluation_result.get("metrics", {}).get("relevance", 0),
                        "clarity": evaluation_result.get("metrics", {}).get("clarity", 0),
                        "completeness": evaluation_result.get("metrics", {}).get("completeness", 0),
                        "confidence": 0.7,  # From voice analysis
                        "conciseness": 75  # Default value
                    },
                    "feedback": feedback_text,
                    "emotions": emotion_scores,  # Include voice emotion analysis
                    "transcribedText": text  # Include the transcribed text
                }
                
                logger.info(f"Evaluation completed. Score: {evaluation['score']}")
                return evaluation
                
        except sr.UnknownValueError as e:
            logger.error(f"Speech Recognition could not understand audio: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail="Could not understand the audio. Please speak more clearly and try again."
            )
        except sr.RequestError as e:
            logger.error(f"Speech recognition service error: {str(e)}")
            raise HTTPException(
                status_code=503,
                detail="Speech recognition service is currently unavailable. Please try again later."
            )
        except Exception as e:
            logger.error(f"Error during audio transcription: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"An error occurred during audio processing: {str(e)}"
            )
        
    except HTTPException as he:
        logger.error(f"HTTP Exception: {str(he.detail)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in evaluate_audio: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while processing your request. Please try again."
        )
    finally:
        # Clean up temporary file
        if 'temp_audio_path' in locals() and temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.unlink(temp_audio_path)
                logger.info(f"Successfully cleaned up temporary file: {temp_audio_path}")
            except Exception as e:
                logger.error(f"Error cleaning up temporary file {temp_audio_path}: {str(e)}")
        elif 'temp_audio' in locals() and hasattr(temp_audio, 'name') and os.path.exists(temp_audio.name):
            try:
                os.unlink(temp_audio.name)
                logger.info(f"Successfully cleaned up temporary file: {temp_audio.name}")
            except Exception as e:
                logger.error(f"Error cleaning up temporary file {temp_audio.name}: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
