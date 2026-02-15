import os
from huggingface_hub import HfApi

# Configuration
REPO_ID = "akskhare/vehicle-breakdown-model"
MODEL_CARD_PATH = "artifacts/README.md"

# Create a basic model card if it doesn't exist locally
if not os.path.exists("artifacts"):
    os.makedirs("artifacts", exist_ok=True)

if not os.path.exists(MODEL_CARD_PATH):
    with open(MODEL_CARD_PATH, "w") as f:
        f.write("# Vehicle Breakdown Prediction\nPredictive maintenance model for engine health.")

api = HfApi(token=os.getenv("HF_TOKEN"))

try:
    api.upload_file(
        path_or_fileobj=MODEL_CARD_PATH,
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="model"
    )
    print("✅ Model card updated on Hugging Face Hub.")
except Exception as e:
    print(f"❌ Error updating model card: {e}")
