from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import os

# Define missing variables
MODEL_NAME = "gradient_boosting_vehicle_breakdown_v1.joblib"
model_path = os.path.join("artifacts", MODEL_NAME)

api = HfApi(token=os.getenv("HF_TOKEN"))

repo_id = "akskhare/vehicle-breakdown-model"
repo_type = "model"

# Create repo if it does not exist
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print("Model repo already exists.")
except RepositoryNotFoundError:
    create_repo(repo_id=repo_id, repo_type=repo_type)
    print("Model repo created.")

# Ensure the model file exists before uploading
if os.path.exists(model_path):
    # Upload model
    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=MODEL_NAME,
        repo_id=repo_id,
        repo_type=repo_type,
    )
    print("🚀 Final model uploaded to Hugging Face")
else:
    print(f"❌ Error: {model_path} not found. Ensure training and evaluation completed successfully.")
