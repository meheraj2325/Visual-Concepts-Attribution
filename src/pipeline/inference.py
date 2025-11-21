import os
import json
from typing import List, Optional
from PIL import Image
from tqdm import tqdm
from ..models.base import VLMInterface
from ..data.structures import VQAExample, ModelResponse
from .prompts import PromptTemplates

class InferencePipeline:
    """Main inference pipeline for VLM analysis"""
    
    def __init__(self, model_interface: VLMInterface):
        self.model = model_interface
        self.prompt_templates = PromptTemplates()
    
    def run_single_example(self, example: VQAExample) -> ModelResponse:
        """Run inference on a single example"""
        
        # Load image
        image = Image.open(example.image_path).convert('RGB')
        
        # 1. Answer with explanation (image + text)
        prompt1 = self.prompt_templates.answer_with_explanation(example.question)
        answer_with_explanation = self.model.generate_response(prompt1, image)
        
        # 2. Extract visual concepts (image only)
        prompt2 = self.prompt_templates.extract_visual_concepts()
        visual_concepts = self.model.generate_response(prompt2, image)
        
        # 3. Extract textual keypoints (text only)
        prompt3 = self.prompt_templates.extract_textual_keypoints(example.question)
        textual_keypoints = self.model.generate_response(prompt3, None)
        
        return ModelResponse(
            example_id=example.example_id or "unknown",
            answer_with_explanation=answer_with_explanation,
            visual_concepts=visual_concepts,
            textual_keypoints=textual_keypoints,
            model_name=self.model.model_name
        )
    
    def run_batch(self, examples: List[VQAExample], save_dir: Optional[str] = None) -> List[ModelResponse]:
        """Run inference on a batch of examples"""
        responses = []
        
        for example in tqdm(examples, desc=f"Processing with {self.model.model_name}"):
            try:
                response = self.run_single_example(example)
                responses.append(response)
                
                # Optionally save intermediate results
                if save_dir:
                    self._save_response(response, save_dir)
                    
            except Exception as e:
                print(f"✗ Error processing {example.example_id}: {e}")
                continue
        
        return responses
    
    def _save_response(self, response: ModelResponse, save_dir: str):
        """Save individual response to file"""
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, f"{response.example_id}.json")
        with open(filepath, 'w') as f:
            json.dump(response.to_dict(), f, indent=2)