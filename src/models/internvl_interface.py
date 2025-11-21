import torch
from typing import Optional
from PIL import Image
from .base import VLMInterface

class InternVLInterface(VLMInterface):
    """Interface for InternVL models"""
    
    def load_model(self):
        """Load InternVL model"""
        try:
            from transformers import AutoModel, AutoTokenizer
            
            model_id = "OpenGVLab/InternVL2-8B"
            
            self.model = AutoModel.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=True
            )
            print(f"✓ Loaded {self.model_name}")
            
        except Exception as e:
            print(f"✗ Error loading {self.model_name}: {e}")
            raise
    
    def generate_response(self, prompt: str, image: Optional[Image.Image] = None) -> str:
        """Generate response using InternVL"""
        response = self.model.chat(
            self.tokenizer,
            pixel_values=None if image is None else image,
            question=prompt,
            generation_config=dict(max_new_tokens=512, do_sample=False)
        )
        
        return response
