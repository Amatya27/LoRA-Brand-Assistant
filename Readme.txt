LoRA Brand Assistant

This submission contains the runnable code for the LoRA + RAG brand assistant.

Code structure
- run_brand_assistant.py: main command-line entry point
- app.py: FastAPI server for the browser app
- run_app.sh: launcher for the web app
- brand_assistant_rag.py: web retrieval, evidence extraction, and prompt building
- brand_assistant_keras_backend.py: Gemma/Keras LoRA inference
- brand_assistant_checkpoint.py: checkpoint discovery and selection
- brand_assistant_service.py: app service layer
- brand_assistant_postprocess.py: app-ready formatting of results
- brand_assistant_schema.py: structured-output parsing and validation
- brand_assistant_openai_backend.py: optional OpenAI-compatible local backend
- prepare_mlx_dataset.py: optional helper for MLX-LM dataset conversion
- webapp/: frontend files

Checkpoint used by default
- The code uses lora_weights_fixed.weights.h5 by default.
- In brand_assistant_checkpoint.py, the code first checks for lora_weights_fixed.weights.h5 and only falls back to lora_weights.weights.h5 if the fixed file is not present.
- Because of that, this submission includes lora_weights_fixed.weights.h5 as the required checkpoint file.

Setup
1. Open a terminal in this folder.
2. Create and activate a Python environment, for example:

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

3. Install dependencies:

pip install -r requirements-local.txt
pip install -r requirements-keras.txt
pip install -r requirements-app.txt

4. Set Kaggle credentials before using the Keras backend. The Gemma preset is downloaded by KerasHub on first run.

Option A:
Place kaggle.json in ~/.kaggle/kaggle.json

Option B:
export KAGGLE_USERNAME="your_kaggle_username"
export KAGGLE_KEY="your_kaggle_key"

Run commands

Part 1: Retrieval only
python run_brand_assistant.py --backend rag-only --company "Nike" --person "Director of Brand Marketing"

Part 2: Full CLI run with the LoRA checkpoint
python run_brand_assistant.py --backend keras --company "Nike" --person "Director of Brand Marketing"

Part 3: Web app
./run_app.sh

Then open:
http://127.0.0.1:8010

If you do not want to use the launcher script, you can run:
python -m uvicorn app:app --host 127.0.0.1 --port 8010

Part 4: Optional OpenAI-compatible local backend
python run_brand_assistant.py --backend openai-compatible --company "Nike" --person "Director of Brand Marketing" --api-base http://localhost:8080/v1 --model mlx-community/gemma-3-1b-it-qat-4bit

Outputs
- CLI outputs are written under outputs/
- App outputs are written under outputs/app/

Notes
- Internet access is required for the DDGS web search and for downloading the Gemma preset on first use.
- This zip is focused on the runnable project files and the default checkpoint used by the current code path.
