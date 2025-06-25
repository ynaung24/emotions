import { useState, useEffect } from 'react';
import { 
  Box, 
  VStack, 
  HStack, 
  Text, 
  Heading, 
  Progress, 
  Container, 
  Button,
  SimpleGrid,
  useToast
} from '@chakra-ui/react';
import { FaArrowLeft } from 'react-icons/fa';
import { useNavigate, useLocation } from 'react-router-dom';
import { evaluateText, EvaluationResponse } from '../api/client';

// Define our UI-specific types
interface EmotionScore {
  name: string;
  value: number;
  color: string;
}

interface UIEvaluation {
  score: number;
  metrics: {
    relevance: number;
    clarity: number;
    confidence: number;
    conciseness: number;
    [key: string]: number; // Allow additional metrics
  };
  emotions: EmotionScore[];
  feedback: string[];
}

// Map emotion names to colors
const emotionColors: Record<string, string> = {
  'happy': 'green',
  'sad': 'blue',
  'angry': 'red',
  'neutral': 'gray',
  'fear': 'purple',
  'surprise': 'yellow',
  'disgust': 'orange',
  'confident': 'teal',
  'nervous': 'yellow',
  'uncertain': 'orange',
  'excited': 'pink'
};

const ResultsPage = () => {
  const [isLoading, setIsLoading] = useState(true);
  const [evaluation, setEvaluation] = useState<UIEvaluation | null>(null);
  const [error, setError] = useState<string | null>(null);
  const navigate = useNavigate();
  const location = useLocation();
  const toast = useToast();

  // Process API response into UI-friendly format
  const processApiResponse = (data: EvaluationResponse): UIEvaluation => {
    // Ensure we have a valid response
    if (!data) {
      throw new Error('No evaluation data received');
    }

    // Process emotions
    const emotions: EmotionScore[] = [];
    if (data.emotions && typeof data.emotions === 'object') {
      Object.entries(data.emotions).forEach(([name, value]) => {
        const numValue = typeof value === 'number' ? value : 0;
        emotions.push({
          name,
          value: Math.round(numValue * 100), // Convert to percentage
          color: emotionColors[name.toLowerCase()] || 'gray'
        });
      });
    }

    // Process feedback
    let feedback: string[] = [];
    if (typeof data.feedback === 'string') {
      feedback = [data.feedback];
    } else if (Array.isArray(data.feedback)) {
      // Type assertion to handle the filter on potentially unknown array type
      feedback = (data.feedback as any[]).filter((f): f is string => typeof f === 'string');
    } else {
      feedback = ['No feedback available.'];
    }

    // Process metrics with defaults, ensuring no property overwrites
    const defaultMetrics = {
      relevance: 0,
      clarity: 0,
      confidence: 0,
      conciseness: 0
    };
    
    // Merge with provided metrics, using default values as fallback
    const metrics = {
      ...defaultMetrics,
      ...(data.metrics || {})
    };
    
    // Ensure all required metrics are numbers
    metrics.relevance = typeof metrics.relevance === 'number' ? metrics.relevance : 0;
    metrics.clarity = typeof metrics.clarity === 'number' ? metrics.clarity : 0;
    metrics.confidence = typeof metrics.confidence === 'number' ? metrics.confidence : 0;
    metrics.conciseness = typeof metrics.conciseness === 'number' ? metrics.conciseness : 0;

    return {
      score: typeof data.score === 'number' ? data.score : 0,
      metrics,
      emotions,
      feedback
    };
  };

  // Fetch evaluation data
  useEffect(() => {
    const fetchEvaluation = async () => {
      try {
        setIsLoading(true);
        
        // Check if we have evaluation data from navigation state
        if (location.state?.evaluation) {
          const processed = processApiResponse(location.state.evaluation);
          setEvaluation(processed);
        } 
        // Check if we have text and question to evaluate
        else if (location.state?.text && location.state?.question) {
          const { text, question } = location.state;
          const response = await evaluateText(text, question);
          const processed = processApiResponse(response);
          setEvaluation(processed);
        } 
        // No valid data provided
        else {
          throw new Error('No evaluation data or input provided');
        }
      } catch (err) {
        console.error('Error in fetchEvaluation:', err);
        setError('Failed to load evaluation. Please try again.');
        toast({
          title: 'Error',
          description: 'Failed to load evaluation results.',
          status: 'error',
          duration: 5000,
          isClosable: true,
        });
      } finally {
        setIsLoading(false);
      }
    };

    fetchEvaluation();
  }, [location.state, toast]);

  const handleRetry = () => {
    navigate('/');
  };

  if (isLoading) {
    return (
      <Container maxW="container.md" py={10}>
        <Box display="flex" justifyContent="center" alignItems="center" minH="70vh">
          <VStack spacing={4}>
            <Progress size="xs" isIndeterminate w="200px" />
            <Text>Loading evaluation results...</Text>
          </VStack>
        </Box>
      </Container>
    );
  }

  if (!evaluation) {
    return (
      <Container maxW="container.md" py={10}>
        <Text>No evaluation data available.</Text>
        <Button mt={4} leftIcon={<FaArrowLeft />} onClick={handleRetry}>
          Back to Home
        </Button>
      </Container>
    );
  }

  return (
    <Container maxW="container.lg" py={10}>
      <Button 
        leftIcon={<FaArrowLeft />} 
        variant="ghost" 
        mb={6}
        onClick={() => navigate(-1)}
      >
        Back
      </Button>

      <VStack spacing={8} align="stretch">
        {/* Overall Score */}
        <Box 
          p={6} 
          borderWidth="1px" 
          borderRadius="lg" 
          bg="white"
          boxShadow="sm"
        >
          <Heading size="lg" mb={4}>Interview Evaluation Results</Heading>
          
          <VStack spacing={6} align="stretch">
            <Box textAlign="center">
              <Text fontSize="sm" color="gray.500" mb={2}>Overall Score</Text>
              <Heading size="2xl" color="brand.500">
                {evaluation.score}/100
              </Heading>
            </Box>

            {/* Metrics */}
            <SimpleGrid columns={{ base: 1, md: 2, lg: 4 }} spacing={4}>
              {Object.entries(evaluation.metrics).map(([key, value]) => (
                <Box key={key} p={4} borderWidth="1px" borderRadius="md">
                  <Text fontSize="sm" color="gray.500" textTransform="capitalize">
                    {key}
                  </Text>
                  <Text fontSize="xl" fontWeight="bold">
                    {value}
                  </Text>
                </Box>
              ))}
            </SimpleGrid>
          </VStack>
        </Box>

        {/* Emotion Analysis - Only show if we have emotions data */}
        {evaluation.emotions && evaluation.emotions.length > 0 && (
          <Box 
            p={6} 
            borderWidth="1px" 
            borderRadius="lg" 
            bg="white"
            boxShadow="sm"
          >
            <Heading size="md" mb={4}>Emotion Analysis</Heading>
            <VStack spacing={4} align="stretch">
              {evaluation.emotions.map((emotion) => (
                <Box key={emotion.name}>
                  <HStack justify="space-between" mb={1}>
                    <Text>{emotion.name}</Text>
                    <Text>{emotion.value}%</Text>
                  </HStack>
                  <Progress 
                    value={emotion.value} 
                    size="sm" 
                    colorScheme={emotion.color}
                    borderRadius="full"
                  />
                </Box>
              ))}
            </VStack>
          </Box>
        )}

        {/* Feedback */}
        <Box 
          p={6} 
          borderWidth="1px" 
          borderRadius="lg" 
          bg="white"
          boxShadow="sm"
        >
          <Heading size="md" mb={4}>Feedback</Heading>
          <VStack spacing={4} align="stretch">
            {evaluation.feedback.map((item, index) => (
              <Box key={index} p={3} bg="gray.50" borderRadius="md">
                <Text>{item}</Text>
              </Box>
            ))}
          </VStack>
        </Box>
      </VStack>
    </Container>
  );
};

export default ResultsPage;
