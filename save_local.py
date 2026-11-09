import kagglehub
import shutil
import os

# 1. Locate the hidden dataset path
cache_path = kagglehub.dataset_download("get2jawa/toxic-comments-train")
full_source_file = os.path.join(cache_path, "train.csv")

# 2. Define the new destination right in your current project folder
local_destination = "./train.csv"

# 3. Copy the full file over
shutil.copy(full_source_file, local_destination)

print("="*60)
print("SUCCESS! The full 'train.csv' dataset has been copied.")
print("Look at the left sidebar of your VS Code workspace to see it.")
print("="*60)
