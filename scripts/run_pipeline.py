#!/usr/bin/env python3
"""
Main script to run the VLM analysis pipeline
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipeline.orchestrator import VLMAnalysisPipeline
import yaml


def main():
    parser = argparse.ArgumentParser(description="Run VLM Explanation Analysis Pipeline")
    parser.add_argument("--config", type=str, default="config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--dataset", type=str, 
                       help="Path to dataset (overrides config)")
    parser.add_argument("--models", nargs="+",
                       help="Models to use (overrides config)")
    parser.add_argument("--output-dir", type=str,
                       help="Output directory (overrides config)")
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with command line arguments
    dataset_path = args.dataset or config['dataset']['path']
    output_dir = args.output_dir or config['output']['dir']
    
    if args.models:
        model_names = args.models
    else:
        model_names = [m['name'] for m in config['models'] if m['enabled']]
    
    # Run pipeline
    pipeline = VLMAnalysisPipeline(output_dir=output_dir)
    
    results = pipeline.run(
        dataset_path=dataset_path,
        model_names=model_names,
        dataset_type=config['dataset']['type'],
        save_intermediate=config['output']['save_intermediate']
    )
    
    print(f"\n✓ Pipeline completed successfully!")
    print(f"✓ Processed {len(model_names)} models")
    print(f"✓ Results saved to: {output_dir}")


if __name__ == "__main__":
    main()