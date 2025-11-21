#!/bin/bash

# Simple setup script for VLM Analysis Pipeline
# Creates only the essential files and directories

echo "Creating VLM Analysis project structure..."


# Create source directories
mkdir -p src/data
mkdir -p src/models
mkdir -p src/pipeline
mkdir -p src/analysis

# Create essential files with empty content
touch src/__init__.py
touch src/data/__init__.py
touch src/data/structures.py
touch src/data/data_manager.py

touch src/models/__init__.py
touch src/models/base.py
touch src/models/llama_interface.py
touch src/models/internvl_interface.py
touch src/models/factory.py

touch src/pipeline/__init__.py
touch src/pipeline/prompts.py
touch src/pipeline/inference.py
touch src/pipeline/orchestrator.py

touch src/analysis/__init__.py
touch src/analysis/analyzer.py

# Create scripts directory
mkdir -p scripts
touch scripts/run_pipeline.py

# Create config and setup files
touch requirements.txt
touch config.yaml
touch .gitignore

# Create data and output directories
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/images
mkdir -p outputs

# Create .gitkeep files
touch data/raw/.gitkeep
touch data/images/.gitkeep
touch outputs/.gitkeep

echo "✓ Project structure created!"
echo ""
echo "Created directories:"
echo "  - src/data/"
echo "  - src/models/"
echo "  - src/pipeline/"
echo "  - src/analysis/"
echo "  - scripts/"
echo "  - data/"
echo "  - outputs/"
echo ""
echo "Now copy your code into these files:"
echo "  1. src/data/structures.py - VQAExample, ModelResponse, AnalysisResult"
echo "  2. src/data/data_manager.py - DataManager class"
echo "  3. src/models/base.py - VLMInterface"
echo "  4. src/models/llama_interface.py - LlamaVLMInterface"
echo "  5. src/models/internvl_interface.py - InternVLInterface"
echo "  6. src/models/factory.py - VLMFactory"
echo "  7. src/pipeline/prompts.py - PromptTemplates"
echo "  8. src/pipeline/inference.py - InferencePipeline"
echo "  9. src/pipeline/orchestrator.py - VLMAnalysisPipeline"
echo "  10. src/analysis/analyzer.py - ExplanationAnalyzer"
echo "  11. scripts/run_pipeline.py - Main execution script"
echo "  12. requirements.txt, setup.py, config.yaml, .gitignore"
echo ""
echo "Next: cd vlm-explanation-analysis"