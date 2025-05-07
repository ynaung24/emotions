import streamlit as st
from evaluate import evaluate_response, load_corpus
from voice_processor import VoiceProcessor
import tempfile
import soundfile as sf
import base64
import os
import speech_recognition as sr
import numpy as np
import sounddevice as sd
from pydub import AudioSegment
import io
import time
import librosa

# Suppress tokenizers warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def process_audio_file(audio_path: str):
    """Process audio file for transcription and emotion detection."""
    try:
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
                
                # Get emotion scores from audio
                try:
                    emotion_scores = voice_processor.detect_emotions('data/temp_recording.wav', text)
                    
                    # Display emotion analysis
                    st.subheader("🎭 Emotion Analysis")
                    if emotion_scores:
                        # Sort emotions by score
                        sorted_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)
                        
                        # Display top emotions
                        st.write("Top emotions detected:")
                        for emotion, score in sorted_emotions:
                            if score > 0.05:  # Only show emotions with score > 5%
                                st.write(f"• {emotion.capitalize()}: {score:.1%}")
                                st.progress(score, text=f"{emotion.capitalize()}")
                        
                        # Show all emotions in expander
                        with st.expander("View all emotion scores"):
                            for emotion, score in sorted_emotions:
                                st.write(f"{emotion}: {score:.4f}")
                    else:
                        st.warning("No emotion scores were returned")
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

def record_audio():
    """Simple audio recording function."""
    try:
        # Record 30 seconds of audio
        duration = 30
        st.write(f"Recording for {duration} seconds...")
        
        # Record audio
        recording = sd.rec(int(duration * 16000), samplerate=16000, channels=1, blocking=True)
        
        # Save to data folder
        os.makedirs('data', exist_ok=True)
        output_path = os.path.join('data', 'temp_recording.wav')
        
        # Save the recording
        sf.write(output_path, recording, 16000)
        
        st.write(f"Recording saved to {output_path}")
        return output_path
        
    except Exception as e:
        st.error(f"Error during recording: {str(e)}")
        return None

# Set up the page
st.set_page_config(page_title="Interview Science", layout="centered")

# Initialize voice processor with proper model path
model_path = os.path.join('models', 'emotion_model.pt')
if os.path.exists(model_path):
    try:
        voice_processor = VoiceProcessor(model_path=model_path)
        st.success("Emotion detection model loaded successfully!")
    except Exception as e:
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
    
    if st.button("Evaluate Text Response", key="evaluate_text"):
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
        st.write("Click 'Start Recording' to begin:")
        
        # Add recording controls
        if st.button("Start Recording", key="start_recording"):
            try:
                # Record and save audio
                audio_path = record_audio()
                
                if audio_path and os.path.exists(audio_path):
                    # Play back the recording
                    st.audio(audio_path)
                else:
                    st.error("Recording failed. Please try again.")
            except Exception as e:
                st.error(f"Error during recording: {str(e)}")
        
        # Add evaluate button that handles both transcription and evaluation
        if st.button("Evaluate Voice Response", key="evaluate_voice"):
            try:
                # Check if recording exists
                if not os.path.exists('data/temp_recording.wav'):
                    st.warning("No recording found. Please record your response first.")
                else:
                    st.write("Processing recording...")
                    
                    # Convert speech to text using Google's speech recognition
                    with sr.AudioFile('data/temp_recording.wav') as source:
                        audio_data = st.session_state.recorder.record(source)
                        text = st.session_state.recorder.recognize_google(
                            audio_data,
                            language="en-US",
                            show_all=False
                        )
                    
                    if text:  # Only proceed if we got text
                        # Display transcription
                        st.subheader("📝 Transcribed Text")
                        st.write(text)
                        
                        # Get emotion scores
                        try:
                            emotion_scores = voice_processor.detect_emotions('data/temp_recording.wav', text)
                            
                            # Display emotion analysis
                            st.subheader("🎭 Emotion Analysis")
                            if emotion_scores:
                                # Sort emotions by score
                                sorted_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)
                                
                                # Display top emotions
                                st.write("Top emotions detected:")
                                for emotion, score in sorted_emotions:
                                    if score > 0.05:  # Only show emotions with score > 5%
                                        st.write(f"• {emotion.capitalize()}: {score:.1%}")
                                        st.progress(score, text=f"{emotion.capitalize()}")
                                
                                # Show all emotions in expander
                                with st.expander("View all emotion scores"):
                                    for emotion, score in sorted_emotions:
                                        st.write(f"{emotion}: {score:.4f}")
                            else:
                                st.warning("No emotion scores were returned")
                        except Exception as e:
                            st.warning(f"Could not analyze emotions: {str(e)}")
                        
                        # Evaluate the response
                        try:
                            # Get base evaluation
                            evaluation = evaluate_response(text, question)
                            
                            # Define positive emotions and their weights
                            positive_emotions = {
                                'pride': 1.2,
                                'confidence': 1.2,
                                'enthusiasm': 1.15,
                                'excitement': 1.15,
                                'optimism': 1.1,
                                'joy': 1.1,
                                'gratitude': 1.1,
                                'admiration': 1.05,
                                'contentment': 1.05
                            }
                            
                            # Calculate emotion boost
                            emotion_boost = 0
                            emotion_impacts = []  # Store emotion impacts for feedback
                            if emotion_scores:
                                for emotion, score in emotion_scores.items():
                                    if emotion.lower() in positive_emotions and score > 0.05:
                                        impact = score * positive_emotions[emotion.lower()]
                                        emotion_boost += impact
                                        emotion_impacts.append({
                                            'emotion': emotion,
                                            'score': score,
                                            'impact': impact,
                                            'weight': positive_emotions[emotion.lower()]
                                        })
                            
                            # Apply boost to scores (capped at 100)
                            if emotion_boost > 0:
                                boost_factor = min(1 + emotion_boost, 1.3)  # Cap at 30% boost
                                original_score = evaluation['score']
                                evaluation['score'] = min(int(evaluation['score'] * boost_factor), 100)
                                evaluation['metrics']['relevance'] = min(int(evaluation['metrics']['relevance'] * boost_factor), 100)
                                evaluation['metrics']['clarity'] = min(int(evaluation['metrics']['clarity'] * boost_factor), 100)
                                evaluation['metrics']['completeness'] = min(int(evaluation['metrics']['completeness'] * boost_factor), 100)
                                
                                # Add emotion impact information to feedback
                                if emotion_impacts:
                                    evaluation['feedback'] += "\n\nEmotional Impact Analysis:"
                                    for impact in sorted(emotion_impacts, key=lambda x: x['impact'], reverse=True):
                                        evaluation['feedback'] += f"\n• {impact['emotion'].capitalize()}: {impact['score']:.1%} detected (weight: {impact['weight']:.2f}x) → {impact['impact']:.1%} boost"
                                    evaluation['feedback'] += f"\n\nTotal emotional boost: {emotion_boost:.1%} (capped at 30%)"
                                    evaluation['feedback'] += f"\nScore adjusted from {original_score} to {evaluation['score']}"
                            
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
                    else:
                        st.error("No speech detected in the recording")
                        
            except sr.UnknownValueError:
                st.error("Could not understand audio. Please speak clearly and try again.")
            except sr.RequestError as e:
                st.error(f"Could not request results from speech recognition service; {e}")
            except Exception as e:
                st.error(f"Error processing audio: {str(e)}")

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
                                if emotion_scores:
                                    # Sort emotions by score
                                    sorted_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)
                                    
                                    # Display top emotions
                                    st.write("Top emotions detected:")
                                    for emotion, score in sorted_emotions:
                                        if score > 0.05:  # Only show emotions with score > 5%
                                            st.write(f"• {emotion.capitalize()}: {score:.1%}")
                                            st.progress(score, text=f"{emotion.capitalize()}")
                                    
                                    # Show all emotions in expander
                                    with st.expander("View all emotion scores"):
                                        for emotion, score in sorted_emotions:
                                            st.write(f"{emotion}: {score:.4f}")
                                else:
                                    st.warning("No emotion scores were returned")
                            except Exception as e:
                                st.warning(f"Could not analyze emotions: {str(e)}")
                            
                            # Evaluate the response
                            try:
                                # Get base evaluation
                                evaluation = evaluate_response(text, question)
                                
                                # Define positive emotions and their weights
                                positive_emotions = {
                                    'pride': 1.2,
                                    'confidence': 1.2,
                                    'enthusiasm': 1.15,
                                    'excitement': 1.15,
                                    'optimism': 1.1,
                                    'joy': 1.1,
                                    'gratitude': 1.1,
                                    'admiration': 1.05,
                                    'contentment': 1.05
                                }
                                
                                # Calculate emotion boost
                                emotion_boost = 0
                                emotion_impacts = []  # Store emotion impacts for feedback
                                if emotion_scores:
                                    for emotion, score in emotion_scores.items():
                                        if emotion.lower() in positive_emotions and score > 0.05:
                                            impact = score * positive_emotions[emotion.lower()]
                                            emotion_boost += impact
                                            emotion_impacts.append({
                                                'emotion': emotion,
                                                'score': score,
                                                'impact': impact,
                                                'weight': positive_emotions[emotion.lower()]
                                            })
                                
                                # Apply boost to scores (capped at 100)
                                if emotion_boost > 0:
                                    boost_factor = min(1 + emotion_boost, 1.3)  # Cap at 30% boost
                                    original_score = evaluation['score']
                                    evaluation['score'] = min(int(evaluation['score'] * boost_factor), 100)
                                    evaluation['metrics']['relevance'] = min(int(evaluation['metrics']['relevance'] * boost_factor), 100)
                                    evaluation['metrics']['clarity'] = min(int(evaluation['metrics']['clarity'] * boost_factor), 100)
                                    evaluation['metrics']['completeness'] = min(int(evaluation['metrics']['completeness'] * boost_factor), 100)
                                    
                                    # Add emotion impact information to feedback
                                    if emotion_impacts:
                                        evaluation['feedback'] += "\n\nEmotional Impact Analysis:"
                                        for impact in sorted(emotion_impacts, key=lambda x: x['impact'], reverse=True):
                                            evaluation['feedback'] += f"\n• {impact['emotion'].capitalize()}: {impact['score']:.1%} detected (weight: {impact['weight']:.2f}x) → {impact['impact']:.1%} boost"
                                        evaluation['feedback'] += f"\n\nTotal emotional boost: {emotion_boost:.1%} (capped at 30%)"
                                        evaluation['feedback'] += f"\nScore adjusted from {original_score} to {evaluation['score']}"
                                
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