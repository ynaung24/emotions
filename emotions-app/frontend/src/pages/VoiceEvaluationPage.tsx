import { useState, useRef, useEffect } from 'react';
import { useQuery } from 'react-query';
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
import { Link as RouterLink } from 'react-router-dom';
import { fetchQuestions } from '../api/client';

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
    
    try {
      // In a real app, you would call the API here
      // const result = await evaluateAudio(audioBlob, selectedQuestion);
      // Then navigate to results page with the evaluation
      
      // For now, we'll simulate a successful submission
      setTimeout(() => {
        setIsProcessing(false);
        // Navigate to results page with mock data
        window.location.href = '/results';
      }, 2000);
      
    } catch (error) {
      console.error('Error submitting recording:', error);
      toast({
        title: 'Error',
        description: 'Failed to process your recording. Please try again.',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
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
