import os
from typing import List
from ..models.factory import VLMFactory
from ..data.structures import VQAExample
from ..data.data_manager import DataManager
from ..analysis.analyzer import ExplanationAnalyzer
from .inference import InferencePipeline

class VLMAnalysisPipeline:
    """End-to-end pipeline orchestrator"""
    
    def __init__(self, output_dir: str = "./outputs"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def run(self, 
            dataset_path: str,
            model_names: List[str],
            dataset_type: str = "custom_json",
            save_intermediate: bool = True):
        """Run the complete pipeline"""
        
        print("=" * 80)
        print("VLM Explanation Analysis Pipeline")
        print("=" * 80)
        
        # Load dataset
        print("\n[1/4] Loading dataset...")
        examples = DataManager.load_vqa_dataset(dataset_path, dataset_type)
        print(f"✓ Loaded {len(examples)} examples")
        
        # Process each model
        all_results = {}
        
        for model_name in model_names:
            print(f"\n[2/4] Processing with {model_name}...")
            
            try:
                # Create model interface
                model = VLMFactory.create_model(model_name)
                pipeline = InferencePipeline(model)
                
                # Run inference
                responses = pipeline.run_batch(
                    examples,
                    save_dir=os.path.join(self.output_dir, f"{model_name}_raw") if save_intermediate else None
                )
                
                # Save responses
                if save_intermediate:
                    DataManager.save_responses(
                        responses,
                        os.path.join(self.output_dir, f"{model_name}_responses.json")
                    )
                
                # Analyze responses
                print(f"\n[3/4] Analyzing responses from {model_name}...")
                analyzer = ExplanationAnalyzer()
                results = analyzer.analyze_batch(responses)
                
                # Save analysis results
                print(f"\n[4/4] Saving analysis results for {model_name}...")
                DataManager.save_results(
                    results,
                    os.path.join(self.output_dir, f"{model_name}_analysis.json"),
                    format="json"
                )
                DataManager.save_results(
                    results,
                    os.path.join(self.output_dir, f"{model_name}_analysis.csv"),
                    format="csv"
                )
                
                all_results[model_name] = results
                
                # Cleanup
                model.cleanup()
                
            except Exception as e:
                print(f"✗ Error processing {model_name}: {e}")
                continue
        
        print("\n" + "=" * 80)
        print("Pipeline Complete!")
        print(f"Results saved to: {self.output_dir}")
        print("=" * 80)
        
        return all_results