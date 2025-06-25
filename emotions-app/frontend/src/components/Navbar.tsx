import { Box, Flex, Button, useColorModeValue, Heading, HStack } from '@chakra-ui/react';
import { Link as RouterLink } from 'react-router-dom';
import { FaHome, FaTextWidth, FaMicrophone } from 'react-icons/fa';

const Navbar = () => {
  const bg = useColorModeValue('white', 'gray.800');
  
  return (
    <Box boxShadow="sm" bg={bg} position="sticky" top={0} zIndex={10}>
      <Flex h={16} alignItems="center" justifyContent="space-between" maxW="container.lg" mx="auto" px={4}>
        <HStack spacing={4}>
          <Heading size="md" color="brand.500">Interview Emotion Analyzer</Heading>
        </HStack>
        <HStack spacing={4}>
          <Button
            as={RouterLink}
            to="/"
            leftIcon={<FaHome />}
            variant="ghost"
            colorScheme="brand"
          >
            Home
          </Button>
          <Button
            as={RouterLink}
            to="/evaluate/text"
            leftIcon={<FaTextWidth />}
            variant="ghost"
            colorScheme="brand"
          >
            Text Evaluation
          </Button>
          <Button
            as={RouterLink}
            to="/evaluate/voice"
            leftIcon={<FaMicrophone />}
            variant="ghost"
            colorScheme="brand"
          >
            Voice Evaluation
          </Button>
        </HStack>
      </Flex>
    </Box>
  );
};

export default Navbar;
