import torch
from typing import Optional
from PIL import Image
from .base import VLMInterface

class LlamaVLMInterface(VLMInterface):
    """Interface for Llama 3.2 Vision models"""
    
    def load_model(self):
        """Load Llama 3.2 Vision model"""
        try:
            from transformers import MllamaForConditionalGeneration, AutoProcessor
            
            model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
            
            self.model = MllamaForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            self.processor = AutoProcessor.from_pretrained(model_id)
            print(f"✓ Loaded {self.model_name}")
            
        except Exception as e:
            print(f"✗ Error loading {self.model_name}: {e}")
            raise
    
    def generate_response(self, prompt: str, image: Optional[Image.Image] = None) -> str:
        """Generate response using Llama 3.2 Vision"""
        messages = [
            {"role": "user", "content": [
                {"type": "image"} if image else None,
                {"type": "text", "text": prompt}
            ]}
        ]
        # Filter out None values
        messages[0]["content"] = [c for c in messages[0]["content"] if c is not None]
        
        input_text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        
        inputs = self.processor(
            image,
            input_text,
            return_tensors="pt"
        ).to(self.model.device)
        
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False
            )
        
        response = self.processor.decode(output[0], skip_special_tokens=True)
        # Extract only the assistant's response
        response = response.split("assistant\n")[-1].strip()
        
        return response