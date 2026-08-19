# main.py
"""
LeafSense AI - Main Pipeline Entry Point
Coordinates data loading, data generator construction, model training, evaluation, inference,
Grad-CAM heatmaps, and advanced metrics visualization.
"""

import os
import sys
import warnings
import logging

# Suppress all C++ and Python warnings / TensorFlow loggers before any other imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.filterwarnings('ignore')

logging.getLogger('tensorflow').setLevel(logging.ERROR)

from seed_everything import seed_everything
from data_loader import load_data
from data_generator import create_generators
from model_training import train_model
from model_evaluation import evaluate_model
from prediction import predict
from gradcam_visualization import visualize_gradcam_grid
from advanced_evaluation import run_advanced_evaluation

def main():
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')

    print("==================================================")
    print("           LeafSense AI Pipeline Starting         ")
    print("==================================================")

    # 1. Set seed for deterministic reproducibility
    seed_everything(42)

    # 2. Load dataset DataFrames
    train_df, test_df = load_data()

    # 3. Build Keras image generators
    train_images, val_images, test_images = create_generators(train_df, test_df)

    # 4. Build and train ResNet50 model
    model, history = train_model(train_images, val_images, epochs=10)

    # 5. Basic evaluation and loss curve visualization
    evaluate_model(model, history, test_images)

    # 6. Predict on test set and plot random predictions
    pred_labels, _ = predict(model, test_images, test_df)

    # 7. Generate Grad-CAM heatmaps
    visualize_gradcam_grid(model, test_df, pred_labels=pred_labels)

    # 8. Run advanced evaluation metrics (PR curves, heatmap report, confusion matrix)
    run_advanced_evaluation(model, test_images, test_df)

    print("==================================================")
    print("       LeafSense AI Pipeline Completed Successfully ")
    print("==================================================")

if __name__ == "__main__":
    main()



