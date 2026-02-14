from huggingface_hub import HfApi

api = HfApi()

repo_id = "akskhare/vehicle-breakdown"

files_to_upload = [
    "X_train_scaled.csv",
    "X_test_scaled.csv",
    "y_train.csv",
    "y_test.csv"
]

for file in files_to_upload:
    api.upload_file(
        path_or_fileobj=file,
        path_in_repo=file,
        repo_id=repo_id,
        repo_type="dataset"
    )

print("✅ All split datasets uploaded")
