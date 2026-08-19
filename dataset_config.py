# dataset_config.py
"""
LeafSense AI - Dataset Configuration Module
Defines global hyper-parameters, target dimensions, and dataset paths.
"""

import os
from pathlib import Path

# Hyperparameters and Data Dimensions
BATCH_SIZE = 32
TARGET_SIZE = (224, 224)

# Primary Kaggle path and local fallbacks
KAGGLE_DATASET = "/kaggle/input/omdena-mango-leaf/MangoLeafBD_Without_Testset_Augmentation"
# dataset_config.py
LOCAL_DATASET = r"C:\path\to\your\MangoLeafBD_Without_Testset_Augmentation"

# Determine active dataset directory
if os.path.exists(KAGGLE_DATASET):
    DATASET_PATH = KAGGLE_DATASET
elif os.path.exists(LOCAL_DATASET):
    DATASET_PATH = LOCAL_DATASET
else:
    DATASET_PATH = LOCAL_DATASET  # Will be created or validated dynamically

def walk_through_dir(dir_path: str):
    """
    Walks through dir_path and prints its contents (number of subdirectories and files).
    """
    if not os.path.exists(dir_path):
        print(f"Directory '{dir_path}' does not exist.")
        return

    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

if __name__ == "__main__":
    walk_through_dir(DATASET_PATH)


