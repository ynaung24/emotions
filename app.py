import streamlit as st
from evaluate import evaluate_response, load_corpus
from voice_processor import VoiceProcessor
import tempfile
import soundfile as sf
import base64
import os
import speech_recognition as sr
import numpy as np
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
from pydub import AudioSegment
import io

# Suppress tokenizers warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def process_audio_file(audio_path: str):
    """Process audio file for transcription and emotion detection."""
    try:
        # Debug: Print initial state
        st.write("Debug - Initial state:")
        st.write(f"Session state user_answer: {st.session_state.user_answer}")
        st.write(f"Session state recording_complete: {st.session_state.recording_complete}")
        
        # Verify file exists and has content
        if not os.path.exists(audio_path):
            st.error("Audio file not found!")
            return
        if os.path.getsize(audio_path) == 0:
            st.error("Audio file is empty!")
            return
            
        # Play back the recording
        st.audio(audio_path, format='audio/wav')
        
        # Convert speech to text
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
            try:
                text = recognizer.recognize_google(audio_data)
                st.write("Transcribed text:", text)
                
                # Store the transcribed text in session state
                st.session_state.user_answer = text
                st.session_state.recording_complete = True
                
                # Debug: Print state after successful transcription
                st.write("Debug - After transcription:")
                st.write(f"Session state user_answer: {st.session_state.user_answer}")
                st.write(f"Session state recording_complete: {st.session_state.recording_complete}")
                
                # Get emotion scores from audio
                try:
                    emotion_scores = voice_processor.detect_emotions(audio_path, text)
                    
                    # Display emotion analysis
                    st.subheader("🎭 Emotion Analysis")
                    for emotion, score in emotion_scores.items():
                        if score > 0.1:  # Only show significant emotions
                            st.progress(score, text=f"{emotion.capitalize()}: {score:.2f}")
                except Exception as e:
                    st.warning(f"Could not analyze emotions: {str(e)}")
                
            except sr.UnknownValueError:
                st.error("Could not understand audio")
                st.session_state.user_answer = ""  # Clear any previous answer
                st.session_state.recording_complete = False
            except sr.RequestError as e:
                st.error(f"Could not request results; {e}")
                st.session_state.user_answer = ""  # Clear any previous answer
                st.session_state.recording_complete = False
    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")
        st.session_state.user_answer = ""  # Clear any previous answer
        st.session_state.recording_complete = False
    finally:
        # Debug: Print final state
        st.write("Debug - Final state:")
        st.write(f"Session state user_answer: {st.session_state.user_answer}")
        st.write(f"Session state recording_complete: {st.session_state.recording_complete}")

# Set up the page
st.set_page_config(page_title="Interview Evaluator", layout="centered")

# Initialize voice processor with proper model path
model_path = os.path.join('models', 'emotion_model.pt')
if os.path.exists(model_path):
    try:
        voice_processor = VoiceProcessor(model_path=model_path)
        st.success("Emotion detection model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        voice_processor = VoiceProcessor()  # Fallback to default model
        st.warning("Using default model due to loading error.")
else:
    voice_processor = VoiceProcessor()  # Initialize without pre-trained model
    st.warning("No pre-trained emotion model found. Using default model.")

# Load corpus and get questions
try:
    corpus = load_corpus()
    questions = [q["text"] for q in corpus["questions"]]
except Exception as e:
    st.error(f"Error loading corpus: {str(e)}")
    st.stop()

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose Input Method:", ["Text Input", "Voice Input"])

# Common elements
st.title("💬 Interview Response Evaluator")
question = st.selectbox("Choose a question:", questions)

if page == "Text Input":
    st.header("📝 Text Response")
    user_answer = st.text_area("Your response:")
    
    if st.button("Evaluate Text Response"):
        if not user_answer.strip():
            st.warning("Please enter a response first.")
        else:
            try:
                evaluation = evaluate_response(user_answer, question)
                
                # Display results
                st.subheader(f"🧠 Overall Score: {evaluation['score']}/100")
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Relevance", f"{evaluation['metrics']['relevance']}/100")
                with col2:
                    st.metric("Clarity", f"{evaluation['metrics']['clarity']}/100")
                with col3:
                    st.metric("Completeness", f"{evaluation['metrics']['completeness']}/100")
                
                # Display feedback
                st.info(f"📋 Feedback: {evaluation['feedback']}")
            except Exception as e:
                st.error(f"Error during evaluation: {str(e)}")

else:  # Voice Input page
    st.header("🎤 Voice Response")
    
    # Initialize session state
    if 'user_answer' not in st.session_state:
        st.session_state.user_answer = ""
    if 'audio_data' not in st.session_state:
        st.session_state.audio_data = None
    if 'recorder' not in st.session_state:
        st.session_state.recorder = sr.Recognizer()
    if 'temp_audio_path' not in st.session_state:
        st.session_state.temp_audio_path = None
    if 'recording_complete' not in st.session_state:
        st.session_state.recording_complete = False
    if 'transcribed_text' not in st.session_state:
        st.session_state.transcribed_text = ""
    
    # Input method selection
    input_method = st.radio(
        "Choose input method:",
        ["Live Recording", "Upload Audio File"]
    )
    
    if input_method == "Live Recording":
        st.write("Click 'Start Recording' to begin and 'Stop Recording' when finished:")
        
        # Add recording controls
        col1, col2 = st.columns(2)
        with col1:
            start_recording = st.button("Start Recording")
        with col2:
            stop_recording = st.button("Stop Recording")
        
        if start_recording:
            st.session_state.recording = True
            st.info("Recording... Speak now!")
            
            try:
                # Start recording
                with sr.Microphone() as source:
                    st.session_state.recorder.adjust_for_ambient_noise(source, duration=1)
                    st.info("Listening... (speak for at least 3 seconds)")
                    audio = st.session_state.recorder.listen(source, timeout=120, phrase_time_limit=120)
                    
                    # Create data directory if it doesn't exist
                    os.makedirs('data', exist_ok=True)
                    
                    # Save to a fixed location in the data directory
                    temp_audio_path = os.path.join('data', 'temp_recording.wav')
                    with open(temp_audio_path, 'wb') as f:
                        f.write(audio.get_wav_data())
                    
                    # Store the audio path in session state
                    st.session_state.temp_audio_path = temp_audio_path
                    
                    # Process the audio immediately
                    try:
                        # Convert speech to text
                        text = st.session_state.recorder.recognize_google(audio)
                        if text:  # Only update if we got text
                            st.session_state.transcribed_text = text
                            st.session_state.user_answer = text
                            st.session_state.recording_complete = True
                            
                            # Get emotion scores
                            try:
                                emotion_scores = voice_processor.detect_emotions(temp_audio_path, text)
                                
                                # Display emotion analysis
                                st.subheader("🎭 Emotion Analysis")
                                for emotion, score in emotion_scores.items():
                                    if score > 0.1:  # Only show significant emotions
                                        st.progress(score, text=f"{emotion.capitalize()}: {score:.2f}")
                            except Exception as e:
                                st.warning(f"Could not analyze emotions: {str(e)}")
                        else:
                            st.error("No speech detected")
                            st.session_state.transcribed_text = ""
                            st.session_state.user_answer = ""
                            st.session_state.recording_complete = False
                            
                    except sr.UnknownValueError:
                        st.error("Could not understand audio")
                        st.session_state.transcribed_text = ""
                        st.session_state.user_answer = ""
                        st.session_state.recording_complete = False
                    except sr.RequestError as e:
                        st.error(f"Could not request results; {e}")
                        st.session_state.transcribed_text = ""
                        st.session_state.user_answer = ""
                        st.session_state.recording_complete = False
                    
            except sr.WaitTimeoutError:
                st.error("No speech detected within timeout period. Please try again.")
                st.session_state.recording = False
            except Exception as e:
                st.error(f"Error during recording: {str(e)}")
                st.session_state.recording = False
        
        if stop_recording:
            st.session_state.recording = False
            st.success("Recording stopped!")
            
            # Display current state
            if st.session_state.transcribed_text:
                st.subheader("📝 Transcribed Text")
                st.write(st.session_state.transcribed_text)
            else:
                st.warning("No transcription available. Please try recording again.")
    
    else:  # Upload Audio File
        st.write("Upload your voice response (supported formats: WAV, MP3, M4A, OGG):")
        
        # Add file uploader for audio
        audio_file = st.file_uploader(
            "Upload your voice response",
            type=["wav", "mp3", "m4a", "ogg"]
        )
        
        if audio_file:
            # Create data directory if it doesn't exist
            os.makedirs('data', exist_ok=True)
            
            # Convert to WAV if needed
            audio_segment = AudioSegment.from_file(audio_file)
            
            # Save to a fixed location
            temp_audio_path = os.path.join('data', 'temp_recording.wav')
            audio_segment.export(temp_audio_path, format="wav")
            st.session_state.temp_audio_path = temp_audio_path
            
            try:
                # Process audio
                with sr.AudioFile(temp_audio_path) as source:
                    # Adjust for ambient noise
                    st.session_state.recorder.adjust_for_ambient_noise(source, duration=0.5)
                    
                    # Record the entire audio file
                    audio_data = st.session_state.recorder.record(source)
                    
                    try:
                        # Use Google's speech recognition with longer timeout
                        text = st.session_state.recorder.recognize_google(
                            audio_data,
                            language="en-US",
                            show_all=False  # Get the most likely transcription
                        )
                        
                        if text:
                            st.session_state.transcribed_text = text
                            st.session_state.user_answer = text
                            st.session_state.recording_complete = True
                            
                            # Display transcription
                            st.subheader("📝 Transcribed Text")
                            st.write(text)
                            
                            # Get emotion scores
                            try:
                                emotion_scores = voice_processor.detect_emotions(temp_audio_path, text)
                                
                                # Display emotion analysis
                                st.subheader("🎭 Emotion Analysis")
                                for emotion, score in emotion_scores.items():
                                    if score > 0.1:  # Only show significant emotions
                                        st.progress(score, text=f"{emotion.capitalize()}: {score:.2f}")
                            except Exception as e:
                                st.warning(f"Could not analyze emotions: {str(e)}")
                        else:
                            st.error("No speech detected in the audio file")
                            st.session_state.transcribed_text = ""
                            st.session_state.user_answer = ""
                            st.session_state.recording_complete = False
                            
                    except sr.UnknownValueError:
                        st.error("Could not understand audio. Please ensure the audio is clear and in English.")
                        st.session_state.transcribed_text = ""
                        st.session_state.user_answer = ""
                        st.session_state.recording_complete = False
                    except sr.RequestError as e:
                        st.error(f"Could not request results from speech recognition service; {e}")
                        st.session_state.transcribed_text = ""
                        st.session_state.user_answer = ""
                        st.session_state.recording_complete = False
            except Exception as e:
                st.error(f"Error processing audio file: {str(e)}")
                st.session_state.transcribed_text = ""
                st.session_state.user_answer = ""
                st.session_state.recording_complete = False
    
    # Add evaluate button
    if st.button("Evaluate Voice Response"):
        if not st.session_state.transcribed_text:
            st.warning("No transcribed text found. Please record your response first.")
        else:
            try:
                evaluation = evaluate_response(st.session_state.transcribed_text, question)
                
                # Display results
                st.subheader(f"🧠 Overall Score: {evaluation['score']}/100")
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Relevance", f"{evaluation['metrics']['relevance']}/100")
                with col2:
                    st.metric("Clarity", f"{evaluation['metrics']['clarity']}/100")
                with col3:
                    st.metric("Completeness", f"{evaluation['metrics']['completeness']}/100")
                
                # Display feedback
                st.info(f"📋 Feedback: {evaluation['feedback']}")
            except Exception as e:
                st.error(f"Error during evaluation: {str(e)}")