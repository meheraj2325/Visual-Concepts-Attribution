from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional

@dataclass
class VQAExample:
    """Represents a single VQA example"""
    image_path: str
    question: str
    ground_truth: Optional[str] = None
    example_id: Optional[str] = None
    metadata: Optional[Dict] = None


@dataclass
class ModelResponse:
    """Stores model responses for a single example"""
    example_id: str
    answer_with_explanation: str
    visual_concepts: str
    textual_keypoints: str
    model_name: str
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class AnalysisResult:
    """Stores analysis results"""
    example_id: str
    model_name: str
    extracted_answer: str
    explanation: str
    visual_concepts_list: List[str]
    textual_keypoints_list: List[str]
    concept_overlap: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        return asdict(self)