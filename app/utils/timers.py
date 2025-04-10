import time
from contextlib import contextmanager
from typing import Generator

from app.utils.logger import setup_logger

logger = setup_logger(__name__)

@contextmanager
def timer(name: str) -> Generator[None, None, None]:
    """
    Context manager for timing code execution.
    
    Args:
        name (str): Name of the operation being timed
        
    Yields:
        None
    """
    start = time.time()
    try:
        yield
    finally:
        end = time.time()
        duration = end - start
        logger.debug(f"{name} took {duration:.2f} seconds")

class PerformanceMonitor:
    """Class for monitoring performance metrics."""
    
    def __init__(self):
        self.metrics = {}
    
    def start_operation(self, name: str) -> None:
        """Start timing an operation."""
        self.metrics[name] = {"start": time.time()}
    
    def end_operation(self, name: str) -> None:
        """End timing an operation and log the duration."""
        if name in self.metrics:
            end = time.time()
            duration = end - self.metrics[name]["start"]
            logger.info(f"Operation {name} completed in {duration:.2f} seconds")
            del self.metrics[name]
    
    def get_duration(self, name: str) -> float:
        """Get the duration of a completed operation."""
        if name in self.metrics:
            return time.time() - self.metrics[name]["start"]
        return 0.0
