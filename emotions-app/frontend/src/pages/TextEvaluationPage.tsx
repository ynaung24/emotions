import { useState } from 'react';
import { useQuery } from 'react-query';
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
  const toast = useToast();
  const cardBg = useColorModeValue('white', 'gray.700');
  const borderColor = useColorModeValue('gray.200', 'gray.600');

  // Fetch questions from the API
  const { data: questionsData, isLoading: isLoadingQuestions } = useQuery('questions', fetchQuestions);
  const questions = questionsData?.questions || [];

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!selectedQuestion) {
      toast({
        title: 'Error',
        description: 'Please select a question',
        status: 'error',
        duration: 3000,
        isClosable: true,
      });
      return;
    }

    if (!response.trim()) {
      toast({
        title: 'Error',
        description: 'Please enter your response',
        status: 'error',
        duration: 3000,
        isClosable: true,
      });
      return;
    }

    setIsSubmitting(true);
    
    try {
      // In a real app, you would call the API here
      // const result = await evaluateText(response, selectedQuestion);
      // Then navigate to results page with the evaluation
      
      // For now, we'll simulate a successful submission
      setTimeout(() => {
        setIsSubmitting(false);
        // Navigate to results page with mock data
        window.location.href = '/results';
      }, 1500);
      
    } catch (error) {
      console.error('Error submitting response:', error);
      toast({
        title: 'Error',
        description: 'Failed to evaluate response. Please try again.',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
      setIsSubmitting(false);
    }
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
