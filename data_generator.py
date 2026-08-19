# data_generator.py
"""
LeafSense AI - Data Generator Module
Builds Keras ImageDataGenerator streams for training, validation, and testing with stratified splits.
"""

from collections import Counter
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from dataset_config import BATCH_SIZE, TARGET_SIZE


def create_generators(train_df: pd.DataFrame, test_df: pd.DataFrame, target_size=TARGET_SIZE, batch_size=BATCH_SIZE):
    """
    Creates ImageDataGenerators for training, validation, and test datasets.

    Returns:
        (train_images, val_images, test_images)
    """
    train_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.resnet50.preprocess_input,
    )
    test_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.resnet50.preprocess_input,
    )

    train_df_shuffled = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

    if len(train_df_shuffled) >= 2 and len(train_df_shuffled['Label'].unique()) > 1:
        train_sub_df, val_df = train_test_split(
            train_df_shuffled,
            test_size=0.3,
            stratify=train_df_shuffled['Label'],
            random_state=42
        )
    else:
        train_sub_df = train_df_shuffled
        val_df = train_df_shuffled

    train_images = train_datagen.flow_from_dataframe(
        dataframe=train_sub_df,
        x_col='Filepath',
        y_col='Label',
        target_size=target_size,
        color_mode='rgb',
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=True,
        seed=42
    )

    val_images = train_datagen.flow_from_dataframe(
        dataframe=val_df,
        x_col='Filepath',
        y_col='Label',
        target_size=target_size,
        color_mode='rgb',
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=True,
        seed=42
    )

    test_images = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col='Filepath',
        y_col='Label',
        target_size=target_size,
        color_mode='rgb',
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=False
    )

    return train_images, val_images, test_images


def get_distribution(generator):
    """
    Computes class distribution from an ImageDataGenerator instance.
    """
    counter = Counter(generator.classes)
    max_val = float(max(counter.values())) if counter else 1.0
    return {k: v / max_val for k, v in counter.items()}

# # Separate in train and test data
# train_generator = ImageDataGenerator(
#     preprocessing_function=tf.keras.applications.resnet50.preprocess_input,
# )
#
# test_generator = ImageDataGenerator(
#     preprocessing_function=tf.keras.applications.resnet50.preprocess_input,
# )
#
# from sklearn.model_selection import train_test_split
#
# # Shuffle your dataframe for randomness
# train_df = train_df.sample(frac=1).reset_index(drop=True)
#
# # Stratified Split
# train_df, val_df = train_test_split(train_df, test_size=0.3, stratify=train_df['Label'])
#
# # Now use train_df with train_images generator and val_df with val_images generator without the subset parameter
# train_images = train_generator.flow_from_dataframe(
#     dataframe=train_df,
#     x_col='Filepath',
#     y_col='Label',
#     target_size=TARGET_SIZE,
#     color_mode='rgb',
#     class_mode='categorical',
#     batch_size=BATCH_SIZE,
#     shuffle=True,
#     seed=42
# )
#
# val_images = train_generator.flow_from_dataframe(
#     dataframe=val_df,
#     x_col='Filepath',
#     y_col='Label',
#     target_size=TARGET_SIZE,
#     color_mode='rgb',
#     class_mode='categorical',
#     batch_size=BATCH_SIZE,
#     shuffle=True,
#     seed=42
# )
#
# test_images = test_generator.flow_from_dataframe(
#     dataframe=test_df,
#     x_col='Filepath',
#     y_col='Label',
#     target_size=TARGET_SIZE,
#     color_mode='rgb',
#     class_mode='categorical',
#     batch_size=BATCH_SIZE,
#     shuffle=False
# )
#
# # Function to get distribution from the generator
# def get_distribution(generator):
#     # Counting occurrences for each class
#     counter = Counter(generator.classes)
#     max_val = float(max(counter.values()))
#     return {k: v/max_val for k, v in counter.items()}
#
# # Print distributions
# print(get_distribution(train_images))
# print(get_distribution(val_images))