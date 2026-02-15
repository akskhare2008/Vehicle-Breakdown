from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import os

api = HfApi(token=os.getenv("HF_TOKEN"))
repo_id = "akskhare/vehicle-breakdown-model"

# Check if Space exists, create if not
try:
    api.repo_info(repo_id=repo_id, repo_type="space")
    print(f"Space {repo_id} already exists.")
except (RepositoryNotFoundError, Exception):
    print(f"Creating space {repo_id}...")
    create_repo(repo_id=repo_id, repo_type="space", space_sdk="streamlit", private=False, token=os.getenv("HF_TOKEN"))

# Upload the deployment folder
api.upload_folder(
    folder_path="maintenance/deployment",
    repo_id=repo_id,
    repo_type="space",
    path_in_repo="",
)
print("🚀 Deployment to Hugging Face Spaces complete")
