# data_loader.py
"""
LeafSense AI - Data Loader Module
Discovers image paths, extracts class labels, builds Pandas DataFrames for training and testing,
and provides a fallback dataset generator for standalone testing.
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
from dataset_config import DATASET_PATH

def create_image_dataframe(dataset_path: str) -> pd.DataFrame:
    """
    Scans dataset_path for JPG, PNG, JPEG images and constructs a DataFrame containing
    file paths and label names.
    """
    image_dir = Path(dataset_path)

    filepaths = (
        list(image_dir.glob(r'**/*.JPG')) +
        list(image_dir.glob(r'**/*.jpg')) +
        list(image_dir.glob(r'**/*.png')) +
        list(image_dir.glob(r'**/*.PNG')) +
        list(image_dir.glob(r'**/*.jpeg')) +
        list(image_dir.glob(r'**/*.JPEG'))
    )

    if not filepaths:
        return pd.DataFrame(columns=['Filepath', 'Label'])

    labels = [os.path.basename(os.path.dirname(fp)) for fp in filepaths]

    filepaths_series = pd.Series([str(fp) for fp in filepaths], name='Filepath')
    labels_series = pd.Series(labels, name='Label')

    image_df = pd.concat([filepaths_series, labels_series], axis=1)
    return image_df

def create_synthetic_dataset(target_dir: str, num_classes=8, images_per_class=10):
    """
    Generates a small synthetic dataset for testing when actual images are missing.
    """
    print(f"[INFO] Creating synthetic fallback dataset at '{target_dir}'...")
    classes = [f"Disease_{i+1}" for i in range(num_classes)]
    for split in ["Train", "Test"]:
        for cls in classes:
            cls_dir = os.path.join(target_dir, split, cls)
            os.makedirs(cls_dir, exist_ok=True)
            for img_idx in range(images_per_class):
                img_path = os.path.join(cls_dir, f"img_{img_idx}.jpg")
                if not os.path.exists(img_path):
                    random_arr = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
                    img = Image.fromarray(random_arr)
                    img.save(img_path)

def load_data(dataset_path: str = None):
    """
    Loads train and test DataFrames. Falls back to synthetic dataset if path is invalid or empty.
    Returns:
        (train_df, test_df) tuple
    """
    if dataset_path is None:
        dataset_path = DATASET_PATH

    train_path = os.path.join(dataset_path, "Train")
    test_path = os.path.join(dataset_path, "Test")

    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        if not os.path.exists(dataset_path) or len(os.listdir(dataset_path)) == 0:
            create_synthetic_dataset(dataset_path)
            train_path = os.path.join(dataset_path, "Train")
            test_path = os.path.join(dataset_path, "Test")

    train_df = create_image_dataframe(train_path if os.path.exists(train_path) else dataset_path)
    test_df = create_image_dataframe(test_path if os.path.exists(test_path) else dataset_path)

    if test_df.empty and not train_df.empty:
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df['Label'])
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

    print(f"[INFO] Loaded DataFrames - Train samples: {len(train_df)}, Test samples: {len(test_df)}")
    return train_df, test_df

if __name__ == "__main__":
    tr_df, te_df = load_data()
    print(tr_df.head())

# def create_image_dataframe(dataset_path: str) -> pd.DataFrame:
#     image_dir = Path(dataset_path)
#
#     # Get filepaths for various extensions
#     filepaths = list(image_dir.glob(r'**/*.JPG')) + \
#                 list(image_dir.glob(r'**/*.jpg')) + \
#                 list(image_dir.glob(r'**/*.png')) + \
#                 list(image_dir.glob(r'**/*.PNG'))
#
#     # Extract labels from filepaths
#     labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))
#
#     # Convert lists to Series
#     filepaths_series = pd.Series(filepaths, name='Filepath').astype(str)
#     labels_series = pd.Series(labels, name='Label')
#
#     # Concatenate filepaths and labels into a DataFrame
#     image_df = pd.concat([filepaths_series, labels_series], axis=1)
#
#     return image_df
#
#
# # Usage
# test_dataset = "/kaggle/input/omdena-mango-leaf/MangoLeafBD_Without_Testset_Augmentation/Test"
# train_dataset = "/kaggle/input/omdena-mango-leaf/MangoLeafBD_Without_Testset_Augmentation/Train"
#
# test_df = create_image_dataframe(test_dataset)
# train_df = create_image_dataframe(train_dataset)
#
# test_df
#
# plt.style.use('fivethirtyeight')
#
# # Get the labels
# label_counts = train_df['Label'].value_counts()[:]
#
# # Create a cycler object using the desired colors
# color_cycler = (cycler(color=["#3B240B"]))  # Darker coffee brown color
#
# # Set the property cycle of the axes to the created cycler object
# plt.rc('axes', prop_cycle=color_cycler)