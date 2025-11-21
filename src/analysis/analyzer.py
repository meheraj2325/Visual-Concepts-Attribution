from typing import Dict, List, Any
from tqdm import tqdm
from ..data.structures import ModelResponse, AnalysisResult

class ExplanationAnalyzer:
    """Analyzes the relationship between visual concepts, textual keypoints, and explanations"""
    
    def __init__(self):
        pass
    
    def parse_concepts(self, text: str) -> List[str]:
        """Parse concepts from model output"""
        # Simple parsing - extract lines or comma-separated items
        concepts = []
        
        # Try line-by-line
        lines = text.strip().split('\n')
        for line in lines:
            line = line.strip()
            # Remove bullet points, numbers, dashes
            line = line.lstrip('•-*123456789.): ')
            if line and len(line) > 2:
                concepts.append(line)
        
        # If no concepts found, try comma separation
        if not concepts:
            concepts = [c.strip() for c in text.split(',') if c.strip()]
        
        return concepts
    
    def calculate_overlap(self, 
                         explanation: str,
                         visual_concepts: List[str],
                         textual_keypoints: List[str]) -> Dict[str, Any]:
        """Calculate overlap between explanation and extracted concepts"""
        
        explanation_lower = explanation.lower()
        
        # Find which visual concepts appear in explanation
        visual_matches = [
            concept for concept in visual_concepts
            if any(word.lower() in explanation_lower for word in concept.split())
        ]
        
        # Find which textual keypoints appear in explanation
        textual_matches = [
            keypoint for keypoint in textual_keypoints
            if any(word.lower() in explanation_lower for word in keypoint.split())
        ]
        
        return {
            "visual_concept_matches": visual_matches,
            "textual_keypoint_matches": textual_matches,
            "visual_coverage": len(visual_matches) / len(visual_concepts) if visual_concepts else 0,
            "textual_coverage": len(textual_matches) / len(textual_keypoints) if textual_keypoints else 0,
            "total_concepts": len(visual_concepts) + len(textual_keypoints),
            "total_matches": len(visual_matches) + len(textual_matches)
        }
    
    def analyze_response(self, response: ModelResponse) -> AnalysisResult:
        """Analyze a single model response"""
        
        # Parse visual concepts and textual keypoints
        visual_concepts_list = self.parse_concepts(response.visual_concepts)
        textual_keypoints_list = self.parse_concepts(response.textual_keypoints)
        
        # Extract answer and explanation (assuming format: answer followed by explanation)
        full_text = response.answer_with_explanation
        parts = full_text.split('\n', 1)
        extracted_answer = parts[0].strip() if parts else full_text[:100]
        explanation = parts[1].strip() if len(parts) > 1 else full_text
        
        # Calculate overlaps
        concept_overlap = self.calculate_overlap(
            explanation,
            visual_concepts_list,
            textual_keypoints_list
        )
        
        return AnalysisResult(
            example_id=response.example_id,
            model_name=response.model_name,
            extracted_answer=extracted_answer,
            explanation=explanation,
            visual_concepts_list=visual_concepts_list,
            textual_keypoints_list=textual_keypoints_list,
            concept_overlap=concept_overlap
        )
    
    def analyze_batch(self, responses: List[ModelResponse]) -> List[AnalysisResult]:
        """Analyze a batch of responses"""
        return [self.analyze_response(resp) for resp in tqdm(responses, desc="Analyzing")]
