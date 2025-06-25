import { useState, useRef, useEffect } from 'react';
import { useQuery } from 'react-query';
import { useNavigate, Link as RouterLink } from 'react-router-dom';
import axios from 'axios';
import { 
  Box, 
  Button, 
  FormControl, 
  FormLabel, 
  Select, 
  VStack, 
  Heading, 
  useToast,
  Spinner,
  Container,
  Text,
  HStack,
  IconButton,
  Progress,
  useColorModeValue,
  CircularProgress,
  CircularProgressLabel,
  Flex
} from '@chakra-ui/react';
import { FaArrowLeft, FaMicrophone, FaStop, FaRedo } from 'react-icons/fa';
import { fetchQuestions, evaluateAudio } from '../api/client';
import { EvaluationResponse } from '../types/evaluation';

// Helper function to convert AudioBuffer to WAV format
const audioBufferToWav = (buffer: AudioBuffer): Blob => {
  const numChannels = buffer.numberOfChannels;
  const sampleRate = buffer.sampleRate;
  const format = 1; // Float32, but we'll convert to 16-bit
  const bitDepth = 16;

  // Get the maximum number of samples across all channels
  const maxSamples = buffer.length;
  
  // Create a buffer for the WAV file
  const bytesPerSample = bitDepth / 8;
  const blockAlign = numChannels * bytesPerSample;
  const dataSize = maxSamples * blockAlign;
  
  // Create buffer for the WAV file
  const bufferSize = 44 + dataSize;
  const arrayBuffer = new ArrayBuffer(bufferSize);
  const view = new DataView(arrayBuffer);

  // Helper to write string to DataView
  const writeString = (view: DataView, offset: number, string: string) => {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  };

  // Helper to write 16-bit PCM samples
  const floatTo16BitPCM = (output: DataView, offset: number, input: Float32Array) => {
    for (let i = 0; i < input.length; i++, offset += 2) {
      const s = Math.max(-1, Math.min(1, input[i]));
      output.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }
  };

  // Write WAV header
  writeString(view, 0, 'RIFF');
  view.setUint32(4, 36 + dataSize, true); // File length
  writeString(view, 8, 'WAVE');
  writeString(view, 12, 'fmt ');
  view.setUint32(16, 16, true); // Subchunk1Size (16 for PCM)
  view.setUint16(20, format, true);
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * blockAlign, true); // Byte rate
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bitDepth, true);
  writeString(view, 36, 'data');
  view.setUint32(40, dataSize, true);

  // Write the audio data
  const offset = 44;
  for (let channel = 0; channel < numChannels; channel++) {
    const channelData = buffer.getChannelData(channel);
    floatTo16BitPCM(view, offset + (channel * bytesPerSample), channelData);
  }

  // Create a Blob from the ArrayBuffer
  return new Blob([view], { type: 'audio/wav' });
};
const VoiceEvaluationPage = () => {
  const [selectedQuestion, setSelectedQuestion] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const [audioUrl, setAudioUrl] = useState<string>('');
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<number>();
  const toast = useToast();
  const cardBg = useColorModeValue('white', 'gray.700');
  const borderColor = useColorModeValue('gray.200', 'gray.600');
  const navigate = useNavigate();

  // Fetch questions from the API
  const { data: questionsData, isLoading: isLoadingQuestions } = useQuery('questions', fetchQuestions);
  const questions = questionsData?.questions || [];

  // Clean up on unmount
  useEffect(() => {
    return () => {
      if (mediaRecorderRef.current) {
        mediaRecorderRef.current.stream.getTracks().forEach(track => track.stop());
      }
      if (timerRef.current) {
        window.clearInterval(timerRef.current);
      }
    };
  }, []);

  const startRecording = async () => {
    if (!selectedQuestion) {
      toast({
        title: 'Error',
        description: 'Please select a question first',
        status: 'error',
        duration: 3000,
        isClosable: true,
      });
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];
      
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };
      
      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
        const audioUrl = URL.createObjectURL(audioBlob);
        setAudioBlob(audioBlob);
        setAudioUrl(audioUrl);
      };
      
      mediaRecorder.start();
      setIsRecording(true);
      setRecordingTime(0);
      
      // Start timer
      timerRef.current = window.setInterval(() => {
        setRecordingTime((time) => time + 1);
      }, 1000);
      
    } catch (error) {
      console.error('Error accessing microphone:', error);
      toast({
        title: 'Microphone Access Error',
        description: 'Could not access your microphone. Please check your permissions.',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
      mediaRecorderRef.current.stream.getTracks().forEach(track => track.stop());
      setIsRecording(false);
      
      if (timerRef.current) {
        window.clearInterval(timerRef.current);
      }
    }
  };

  const resetRecording = () => {
    setAudioBlob(null);
    setAudioUrl('');
    setRecordingTime(0);
  };

  const submitRecording = async () => {
    if (!audioBlob || !selectedQuestion) return;
    
    setIsProcessing(true);
    let toastId: string | number | undefined;
    
    try {
      // Show loading toast
      toastId = toast({
        title: 'Processing your recording',
        description: 'This may take a moment...',
        status: 'info',
        duration: null,
        isClosable: false,
        position: 'top',
      });
      
      // Ensure we have a WAV file with the correct format
      const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
      const arrayBuffer = await audioBlob.arrayBuffer();
      const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
      
      // Convert to mono 16-bit PCM WAV
      const wavBlob = await audioBufferToWav(audioBuffer);
      
      // Create a proper WAV file with correct headers
      const wavFile = new File([wavBlob], 'recording.wav', { 
        type: 'audio/wav'
      });
      
      // Call the API with the properly formatted WAV file
      const result = await evaluateAudio(wavFile, selectedQuestion);
      
      // Close loading toast
      if (toastId) {
        toast.close(toastId);
      }
      
      // Navigate to results page with the evaluation
      navigate('/results', { 
        state: { 
          evaluation: result,
          text: result.feedback, // Use the transcribed text if available
          question: selectedQuestion,
          timestamp: new Date().toISOString(),
          isVoiceEvaluation: true
        } 
      });
      
    } catch (error) {
      console.error('Error submitting recording:', error);
      
      // Close loading toast if it's still open
      if (toastId) {
        toast.close(toastId);
      }
      
      let errorMessage = 'Failed to process your recording. Please try again.';
      
      if (axios.isAxiosError(error)) {
        const responseData = error.response?.data as { detail?: string } | undefined;
        const errorDetail = responseData?.detail || '';
        
        if (error.response?.status === 400) {
          errorMessage = 'Invalid audio. Please ensure your recording is clear and try again.';
        } else if (error.response?.status === 500) {
          errorMessage = 'Server error. Please try again later.';
        } else if (error.code === 'ECONNABORTED') {
          errorMessage = 'Request timed out. Please check your connection and try again.';
        }
        
        if (errorDetail) {
          errorMessage += ` (${errorDetail})`;
        }
      }
      
      toast({
        title: 'Error',
        description: errorMessage,
        status: 'error',
        duration: 5000,
        isClosable: true,
        position: 'top',
      });
    } finally {
      setIsProcessing(false);
    }
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  if (isLoadingQuestions) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minH="50vh">
        <Spinner size="xl" />
      </Box>
    );
  }

  return (
    <Container maxW="container.lg" py={8}>
      <Button
        as={RouterLink}
        to="/"
        leftIcon={<FaArrowLeft />}
        variant="ghost"
        mb={6}
        colorScheme="brand"
      >
        Back to Home
      </Button>

      <Box
        bg={cardBg}
        borderRadius="lg"
        boxShadow="lg"
        p={8}
        borderWidth="1px"
        borderColor={borderColor}
      >
        <VStack spacing={6}>
          <Heading as="h1" size="xl" textAlign="center" color="brand.500">
            Voice Response Evaluation
          </Heading>
          
          <Text textAlign="center" color="gray.600" mb={6}>
            Record your response to the interview question and get instant feedback on your tone and emotions
          </Text>
          
          <FormControl isRequired>
            <FormLabel>Select a question</FormLabel>
            <Select
              placeholder="Choose a question"
              value={selectedQuestion}
              onChange={(e) => setSelectedQuestion(e.target.value)}
              size="lg"
              mb={6}
              isDisabled={isRecording || !!audioUrl}
            >
              {questions.map((question, index) => (
                <option key={index} value={question}>
                  {question}
                </option>
              ))}
            </Select>
          </FormControl>
          
          <Box w="100%" textAlign="center" mb={6}>
            {!audioUrl ? (
              <VStack spacing={6}>
                <CircularProgress
                  value={isRecording ? (recordingTime % 100) : 0}
                  color="brand.500"
                  size="160px"
                  thickness="4px"
                  trackColor="gray.200"
                >
                  <CircularProgressLabel>
                    {isRecording ? formatTime(recordingTime) : '00:00'}
                  </CircularProgressLabel>
                </CircularProgress>
                
                <Button
                  leftIcon={isRecording ? <FaStop /> : <FaMicrophone />}
                  colorScheme={isRecording ? 'red' : 'brand'}
                  size="lg"
                  onClick={isRecording ? stopRecording : startRecording}
                  isDisabled={!selectedQuestion}
                  px={8}
                  py={6}
                >
                  {isRecording ? 'Stop Recording' : 'Start Recording'}
                </Button>
                
                {!isRecording && recordingTime > 0 && (
                  <Button
                    leftIcon={<FaRedo />}
                    variant="outline"
                    onClick={resetRecording}
                    size="sm"
                  >
                    Start Over
                  </Button>
                )}
              </VStack>
            ) : (
              <VStack spacing={6}>
                <Box w="100%" p={4} borderWidth="1px" borderRadius="md">
                  <audio src={audioUrl} controls style={{ width: '100%' }} />
                </Box>
                
                <HStack spacing={4} w="100%" justify="center">
                  <Button
                    leftIcon={<FaRedo />}
                    onClick={resetRecording}
                    variant="outline"
                  >
                    Re-record
                  </Button>
                  
                  <Button
                    colorScheme="brand"
                    onClick={submitRecording}
                    isLoading={isProcessing}
                    loadingText="Analyzing..."
                    rightIcon={<FaMicrophone />}
                  >
                    Analyze Recording
                  </Button>
                </HStack>
              </VStack>
            )}
          </Box>
          
          {isRecording && (
            <Box w="100%" mt={4}>
              <Text fontSize="sm" color="gray.600" mb={2} textAlign="center">
                Recording in progress...
              </Text>
              <Progress size="xs" isIndeterminate colorScheme="brand" />
            </Box>
          )}
        </VStack>
      </Box>
      
      <Box mt={8} textAlign="center">
        <Text color="gray.500" fontSize="sm">
          Your voice will be analyzed for tone, pace, and emotional expression.
        </Text>
      </Box>
    </Container>
  );
};

export default VoiceEvaluationPage;
