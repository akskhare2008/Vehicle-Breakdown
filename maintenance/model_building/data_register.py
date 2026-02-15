# Corrected Syntax: Added missing comma to ensure CI/CD success
from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub import HfApi, create_repo
from datasets import load_dataset
import os

repo_id = "akskhare/vehicle-breakdown"
repo_type = "dataset"

# Initialize API client
api = HfApi(token=os.getenv("HF_TOKEN"))

# Step 1: Check if dataset repo exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Dataset '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Dataset '{repo_id}' not found. Creating new dataset repo...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Dataset '{repo_id}' created.")

# Step 2: Upload data folder
api.upload_folder(
    folder_path="maintenance/data",
    path_in_repo="data",
    repo_id=repo_id,
    repo_type=repo_type,
)

# Step 3: Verify dataset can be loaded
dataset = load_dataset(
    repo_id,
    data_files="data/engine_data.csv"
)

print(dataset)
