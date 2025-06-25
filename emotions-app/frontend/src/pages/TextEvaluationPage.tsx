import { useState, useEffect } from 'react';
import { useQuery } from 'react-query';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import { 
  Box, 
  Button, 
  FormControl, 
  FormLabel, 
  Select, 
  Textarea, 
  VStack, 
  Heading, 
  useToast,
  Spinner,
  Container,
  Text,
  HStack,
  IconButton,
  useColorModeValue,
} from '@chakra-ui/react';
import { FaArrowLeft, FaPaperPlane } from 'react-icons/fa';
import { Link as RouterLink } from 'react-router-dom';
import { fetchQuestions, evaluateText } from '../api/client';

const TextEvaluationPage = () => {
  const [selectedQuestion, setSelectedQuestion] = useState('');
  const [response, setResponse] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const navigate = useNavigate();
  const toast = useToast();
  const cardBg = useColorModeValue('white', 'gray.700');
  const borderColor = useColorModeValue('gray.200', 'gray.600');

  // Fetch questions from the API with error handling
  const { 
    data: questionsData, 
    isLoading: isLoadingQuestions, 
    error: questionsError,
    refetch: refetchQuestions
  } = useQuery('questions', fetchQuestions, {
    retry: 2,
    refetchOnWindowFocus: false
  });
  
  const questions = questionsData?.questions || [];
  
  // Show error toast if questions fail to load
  useEffect(() => {
    if (questionsError) {
      toast({
        title: 'Error loading questions',
        description: 'Failed to load interview questions. Please try again.',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    }
  }, [questionsError, toast]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    // Validate inputs
    if (!selectedQuestion) {
      toast({
        title: 'Error',
        description: 'Please select a question',
        status: 'error',
        duration: 3000,
        isClosable: true,
        position: 'top',
      });
      return;
    }

    const trimmedResponse = response.trim();
    if (!trimmedResponse) {
      toast({
        title: 'Error',
        description: 'Please enter your response',
        status: 'error',
        duration: 3000,
        isClosable: true,
        position: 'top',
      });
      return;
    }

    // Check if response is too short
    if (trimmedResponse.length < 10) {
      toast({
        title: 'Response too short',
        description: 'Please provide a more detailed response for better evaluation.',
        status: 'warning',
        duration: 4000,
        isClosable: true,
        position: 'top',
      });
    }

    setIsSubmitting(true);
    let toastId: string | number | undefined;
    
    try {
      // Show loading toast
      toastId = toast({
        title: 'Evaluating your response',
        description: 'This may take a moment...',
        status: 'info',
        duration: null, // Don't auto-dismiss
        isClosable: false,
        position: 'top',
      });
      
      // Call the evaluation API
      const evaluation = await evaluateText(trimmedResponse, selectedQuestion);
      
      // Close loading toast
      if (toastId) {
        toast.close(toastId);
      }
      
      // Show success toast
      toast({
        title: 'Evaluation complete!',
        status: 'success',
        duration: 2000,
        isClosable: true,
        position: 'top',
      });
      
      // Navigate to results page with the evaluation data
      navigate('/results', { 
        state: { 
          evaluation,
          text: trimmedResponse,
          question: selectedQuestion,
          timestamp: new Date().toISOString()
        } 
      });
      
    } catch (error) {
      console.error('Error evaluating response:', error);
      
      // Close loading toast if it's still open
      if (toastId) {
        toast.close(toastId);
      }
      
      let errorMessage = 'Failed to evaluate response. Please try again.';
      let errorDetails = '';
      
      if (axios.isAxiosError(error)) {
        // Safely access error response data
        const responseData = error.response?.data as { detail?: string } | undefined;
        errorDetails = responseData?.detail || '';
        
        // Handle different HTTP status codes
        if (error.response?.status === 400) {
          errorMessage = 'Invalid request. Please check your input.';
        } else if (error.response?.status === 500) {
          errorMessage = 'Server error. Please try again later.';
        } else if (error.code === 'ECONNABORTED') {
          errorMessage = 'Request timed out. Please check your connection and try again.';
        }
      }
      
      toast({
        title: 'Evaluation Error',
        description: (
          <Box>
            <Text>{errorMessage}</Text>
            {errorDetails && <Text fontSize="sm" mt={2} fontStyle="italic">{errorDetails}</Text>}
          </Box>
        ),
        status: 'error',
        duration: 5000, // Reduced from 8000 to 5000
        isClosable: true,
        position: 'top',
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  if (isLoadingQuestions) {
    return (
      <Container maxW="container.lg" py={8}>
        <Box display="flex" flexDirection="column" alignItems="center" justifyContent="center" minH="50vh">
          <Spinner size="xl" mb={4} />
          <Text>Loading questions...</Text>
        </Box>
      </Container>
    );
  }
  
  if (questionsError) {
    return (
      <Container maxW="container.lg" py={8}>
        <Box textAlign="center" py={10} px={6}>
          <Heading as="h2" size="xl" mt={6} mb={2}>
            Failed to load questions
          </Heading>
          <Text color={'gray.500'} mb={6}>
            We couldn't load the interview questions. Please try again.
          </Text>
          <Button
            colorScheme="brand"
            onClick={() => refetchQuestions()}
            leftIcon={<FaArrowLeft />}
          >
            Retry
          </Button>
        </Box>
      </Container>
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
        <VStack spacing={6} as="form" onSubmit={handleSubmit}>
          <Heading as="h1" size="xl" textAlign="center" color="brand.500">
            Text Response Evaluation
          </Heading>
          
          <Text textAlign="center" color="gray.600" mb={6}>
            Type your response to the interview question below and get instant feedback
          </Text>
          
          <FormControl isRequired>
            <FormLabel>Select a question</FormLabel>
            <Select
              placeholder="Choose a question"
              value={selectedQuestion}
              onChange={(e) => setSelectedQuestion(e.target.value)}
              size="lg"
              mb={6}
            >
              {questions.map((question, index) => (
                <option key={index} value={question}>
                  {question}
                </option>
              ))}
            </Select>
          </FormControl>
          
          <FormControl isRequired>
            <FormLabel>Your Response</FormLabel>
            <Textarea
              value={response}
              onChange={(e) => setResponse(e.target.value)}
              placeholder="Type your response here..."
              size="lg"
              minH="200px"
              resize="vertical"
            />
          </FormControl>
          
          <HStack spacing={4} w="100%" justifyContent="flex-end" mt={4}>
            <Button
              type="submit"
              colorScheme="brand"
              size="lg"
              rightIcon={<FaPaperPlane />}
              isLoading={isSubmitting}
              loadingText="Evaluating..."
              isDisabled={!selectedQuestion || !response.trim()}
            >
              Evaluate Response
            </Button>
          </HStack>
        </VStack>
      </Box>
      
      <Box mt={8} textAlign="center">
        <Text color="gray.500" fontSize="sm">
          Your response will be analyzed for content, clarity, and emotional tone.
        </Text>
      </Box>
    </Container>
  );
};

export default TextEvaluationPage;
