import sounddevice as sd
import soundfile as sf
import numpy as np
import time
import os

def test_recording():
    print("Testing audio recording...")
    
    # Test 1: Basic recording
    print("\nTest 1: Basic recording")
    try:
        # Record 5 seconds of audio
        duration = 5
        print(f"Recording for {duration} seconds...")
        recording = sd.rec(int(duration * 16000), samplerate=16000, channels=1, blocking=True)
        print("Recording completed")
        
        # Save the recording
        sf.write('test_recording.wav', recording, 16000)
        print("Recording saved to test_recording.wav")
        
        # Verify the file exists and has content
        if os.path.exists('test_recording.wav'):
            size = os.path.getsize('test_recording.wav')
            print(f"File size: {size} bytes")
            if size > 0:
                print("Test 1: PASSED")
            else:
                print("Test 1: FAILED - File is empty")
        else:
            print("Test 1: FAILED - File not created")
            
    except Exception as e:
        print(f"Test 1: FAILED - Error: {str(e)}")
    
    # Test 2: Check audio levels
    print("\nTest 2: Check audio levels")
    try:
        if recording is not None and len(recording) > 0:
            max_level = np.max(np.abs(recording))
            print(f"Maximum audio level: {max_level}")
            if max_level > 0:
                print("Test 2: PASSED - Audio detected")
            else:
                print("Test 2: FAILED - No audio detected")
        else:
            print("Test 2: FAILED - No recording data")
    except Exception as e:
        print(f"Test 2: FAILED - Error: {str(e)}")

if __name__ == "__main__":
    test_recording() 