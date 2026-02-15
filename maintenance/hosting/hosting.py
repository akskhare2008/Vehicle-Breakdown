from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import os

api = HfApi(token=os.getenv("HF_TOKEN"))
repo_id = "akskhare/vehicle-breakdown"

# Check if Space exists, create if not
try:
    api.repo_info(repo_id=repo_id, repo_type="space")
    print(f"Space {repo_id} already exists.")
except Exception:
    print(f"Space {repo_id} not found or inaccessible. Attempting to create...")
    try:
        create_repo(repo_id=repo_id, repo_type="space", space_sdk="streamlit", private=False, token=os.getenv("HF_TOKEN"))
        print(f"Successfully created space {repo_id}")
    except Exception as e:
        print(f"Note: Space might already exist or creation failed: {e}")

# Upload the deployment folder
api.upload_folder(
    folder_path="maintenance/deployment",
    repo_id=repo_id,
    repo_type="space",
    path_in_repo="",
)
print("🚀 Deployment to Hugging Face Spaces complete")
