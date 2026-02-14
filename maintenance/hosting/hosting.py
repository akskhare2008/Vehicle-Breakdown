from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="maintenance/deployment",     
    repo_id="akskhare/vehicle-breakdown-model",         
    repo_type="space",                     
    path_in_repo="",                         
)
