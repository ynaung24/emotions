import { Box, Button, Container, Heading, Stack, Text, VStack, useColorModeValue } from '@chakra-ui/react';
import { FaMicrophone, FaKeyboard } from 'react-icons/fa';
import { Link as RouterLink } from 'react-router-dom';

const HomePage = () => {


  return (
    <VStack spacing={12} py={12}>
      <Box textAlign="center" py={10} px={6}>
        <Heading as="h1" size="2xl" mb={4} color="brand.500">
          Interview Emotion Analyzer
        </Heading>
        <Text fontSize="xl" color="gray.600">
          Get instant feedback on your interview responses with AI-powered emotion analysis
        </Text>
      </Box>

      <Container maxW="container.lg">
        <VStack spacing={8}>
          <Box
            p={8}
            borderRadius="lg"
            boxShadow="lg"
            bg={useColorModeValue('white', 'gray.700')}
            w="100%"
          >
            <VStack spacing={6}>
              <Heading as="h2" size="lg">Get Started</Heading>
              <Text textAlign="center" color="gray.600">
                Choose how you'd like to practice your interview responses:
              </Text>
              <Stack direction={{ base: 'column', md: 'row' }} spacing={6} w="100%" justify="center">
                <Button
                  as={RouterLink}
                  to="/evaluate/text"
                  leftIcon={<FaKeyboard />}
                  size="lg"
                  colorScheme="brand"
                  variant="outline"
                  px={8}
                  py={6}
                  height="auto"
                  whiteSpace="normal"
                >
                  <VStack spacing={2}>
                    <Text>Text Response</Text>
                    <Text fontSize="sm" fontWeight="normal" color="gray.600">
                      Type your response and get feedback
                    </Text>
                  </VStack>
                </Button>
                <Button
                  as={RouterLink}
                  to="/evaluate/voice"
                  leftIcon={<FaMicrophone />}
                  size="lg"
                  colorScheme="brand"
                  variant="solid"
                  px={8}
                  py={6}
                  height="auto"
                  whiteSpace="normal"
                >
                  <VStack spacing={2}>
                    <Text>Voice Response</Text>
                    <Text fontSize="sm" fontWeight="normal" color="whiteAlpha.900">
                      Record your voice and analyze emotions
                    </Text>
                  </VStack>
                </Button>
              </Stack>
            </VStack>
          </Box>

          <Box
            p={8}
            borderRadius="lg"
            boxShadow="lg"
            bg={useColorModeValue('white', 'gray.700')}
            w="100%"
          >
            <VStack spacing={4}>
              <Heading as="h3" size="md">How It Works</Heading>
              <Stack direction={{ base: 'column', md: 'row' }} spacing={8} w="100%">
                <FeatureCard
                  title="1. Choose a Question"
                  description="Select from common interview questions or use your own."
                />
                <FeatureCard
                  title="2. Record or Type"
                  description="Record your voice response or type it out."
                />
                <FeatureCard
                  title="3. Get Feedback"
                  description="Receive detailed analysis and emotion detection."
                />
              </Stack>
            </VStack>
          </Box>
        </VStack>
      </Container>
    </VStack>
  );
};

const FeatureCard = ({ title, description }: { title: string; description: string }) => (
  <Box
    p={4}
    borderWidth="1px"
    borderRadius="md"
    flex={1}
    _hover={{
      transform: 'translateY(-4px)',
      boxShadow: 'lg',
      transition: 'all 0.2s',
    }}
  >
    <VStack spacing={2} align="start">
      <Text fontWeight="bold">{title}</Text>
      <Text color="gray.600" fontSize="sm">
        {description}
      </Text>
    </VStack>
  </Box>
);

export default HomePage;
