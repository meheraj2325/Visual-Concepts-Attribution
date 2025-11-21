from abc import ABC, abstractmethod
from typing import Optional
from PIL import Image
import torch

class VLMInterface(ABC):
    """Abstract interface for Vision-Language Models"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
    
    @abstractmethod
    def load_model(self):
        """Load the model and processor"""
        pass
    
    @abstractmethod
    def generate_response(self, prompt: str, image: Optional[Image.Image] = None) -> str:
        """Generate response from the model"""
        pass
    
    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'processor'):
            del self.processor
        torch.cuda.empty_cache()
