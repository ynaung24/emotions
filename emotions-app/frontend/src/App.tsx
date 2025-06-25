import { ChakraProvider, Box, Container, Heading, VStack } from '@chakra-ui/react';
import { QueryClient, QueryClientProvider } from 'react-query';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import theme from './theme';
import HomePage from './pages/HomePage';
import TextEvaluationPage from './pages/TextEvaluationPage';
import VoiceEvaluationPage from './pages/VoiceEvaluationPage';
import ResultsPage from './pages/ResultsPage';
import Navbar from './components/Navbar';

const queryClient = new QueryClient();

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <ChakraProvider theme={theme}>
        <Router>
          <Box minH="100vh" bg="gray.50">
            <Navbar />
            <Container maxW="container.lg" py={8}>
              <Routes>
                <Route path="/" element={<HomePage />} />
                <Route path="/evaluate/text" element={<TextEvaluationPage />} />
                <Route path="/evaluate/voice" element={<VoiceEvaluationPage />} />
                <Route path="/results" element={<ResultsPage />} />
              </Routes>
            </Container>
          </Box>
        </Router>
      </ChakraProvider>
    </QueryClientProvider>
  );
}

export default App;
