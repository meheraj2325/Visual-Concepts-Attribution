from typing import List
from .base import VLMInterface
from .llama_interface import LlamaVLMInterface
from .internvl_interface import InternVLInterface

class VLMFactory:
    """Factory for creating VLM interfaces"""
    
    _models = {
        "llama-3.2-11b-vision": LlamaVLMInterface,
        "internvl-2-8b": InternVLInterface,
    }
    
    @classmethod
    def create_model(cls, model_name: str) -> VLMInterface:
        """Create a VLM interface instance"""
        if model_name not in cls._models:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(cls._models.keys())}")
        
        interface = cls._models[model_name](model_name)
        interface.load_model()
        return interface
    
    @classmethod
    def available_models(cls) -> List[str]:
        """Get list of available models"""
        return list(cls._models.keys())