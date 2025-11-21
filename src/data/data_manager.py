import os
import json
from typing import List
import pandas as pd
from .structures import VQAExample, AnalysisResult, ModelResponse

class DataManager:
    """Manages data loading and saving"""
    
    @staticmethod
    def load_vqa_dataset(dataset_path: str, dataset_type: str = "custom") -> List[VQAExample]:
        """Load VQA dataset from various formats"""
        
        if dataset_type == "custom_json":
            with open(dataset_path, 'r') as f:
                data = json.load(f)
            
            examples = []
            for item in data:
                examples.append(VQAExample(
                    image_path=item['image_path'],
                    question=item['question'],
                    ground_truth=item.get('answer'),
                    example_id=item.get('id', str(len(examples))),
                    metadata=item.get('metadata')
                ))
            return examples
        
        elif dataset_type == "custom_csv":
            df = pd.read_csv(dataset_path)
            examples = []
            for idx, row in df.iterrows():
                examples.append(VQAExample(
                    image_path=row['image_path'],
                    question=row['question'],
                    ground_truth=row.get('answer'),
                    example_id=str(row.get('id', idx)),
                    metadata=None
                ))
            return examples
        
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")
    
    @staticmethod
    def save_results(results: List[AnalysisResult], output_path: str, format: str = "json"):
        """Save analysis results"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if format == "json":
            with open(output_path, 'w') as f:
                json.dump([r.to_dict() for r in results], f, indent=2)
        
        elif format == "csv":
            df = pd.DataFrame([r.to_dict() for r in results])
            df.to_csv(output_path, index=False)
        
        print(f"✓ Saved results to {output_path}")
    
    @staticmethod
    def save_responses(responses: List[ModelResponse], output_path: str):
        """Save raw model responses"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump([r.to_dict() for r in responses], f, indent=2)
        
        print(f"✓ Saved responses to {output_path}")