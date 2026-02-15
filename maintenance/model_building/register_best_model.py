import os
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# Configuration
REPO_ID = "akskhare/vehicle-breakdown-model"
MODEL_FILENAME = "gradient_boosting_vehicle_breakdown_v1.joblib"
LOCAL_MODEL_PATH = os.path.join("artifacts", MODEL_FILENAME)

api = HfApi(token=os.getenv("HF_TOKEN"))

try:
    api.repo_info(repo_id=REPO_ID, repo_type="model")
    print(f"Model repository {REPO_ID} already exists.")
except RepositoryNotFoundError:
    print(f"Creating model repository {REPO_ID}...")
    create_repo(repo_id=REPO_ID, repo_type="model", private=False)

if os.path.exists(LOCAL_MODEL_PATH):
    print(f"Uploading {MODEL_FILENAME} to Hugging Face Hub...")
    api.upload_file(
        path_or_fileobj=LOCAL_MODEL_PATH,
        path_in_repo=MODEL_FILENAME,
        repo_id=REPO_ID,
        repo_type="model"
    )
    print("✅ Model registration successful.")
else:
    print(f"❌ Error: Model file not found at {LOCAL_MODEL_PATH}")
