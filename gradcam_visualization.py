# gradcam_visualization.py
"""
LeafSense AI - Grad-CAM Visualizer Module
Generates and displays visual explanation heatmaps across test image samples using ResNet50 feature maps.
"""

import textwrap
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

from gradcam import get_img_array, make_gradcam_heatmap, save_and_display_gradcam

preprocess_input = tf.keras.applications.resnet50.preprocess_input

def visualize_gradcam_grid(model, test_df: pd.DataFrame, pred_labels: list = None, last_conv_layer_name: str = "conv5_block3_out", img_size: tuple = (224, 224), num_samples: int = 15):
    """
    Renders a grid of Grad-CAM heatmaps showing which leaf regions influenced neural network classification decisions.
    """
    if test_df.empty or 'Filepath' not in test_df.columns:
        print("[WARNING] Cannot plot Grad-CAM: DataFrame is empty or missing 'Filepath'.")
        return

    sample_count = min(num_samples, len(test_df))
    random_indices = np.random.choice(len(test_df), size=sample_count, replace=False)

    fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(18, 10), subplot_kw={'xticks': [], 'yticks': []})
    axes_flat = axes.flat

    for i, ax in enumerate(axes_flat):
        if i < sample_count:
            idx = random_indices[i]
            img_path = test_df.iloc[idx]['Filepath']
            true_label = test_df.iloc[idx]['Label']
            pred_label = pred_labels[idx] if (pred_labels and idx < len(pred_labels)) else "N/A"

            try:
                img_array = preprocess_input(get_img_array(img_path, size=img_size))
                heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name=last_conv_layer_name)
                cam_path = save_and_display_gradcam(img_path, heatmap, cam_path=f"temp_cam_{i}.jpg")
                ax.imshow(plt.imread(cam_path))

                title_text = f"True: {true_label}\nPred: {pred_label}"
                wrapped_title = '\n'.join(textwrap.wrap(title_text, 22))
                ax.set_title(wrapped_title, fontsize=9)
            except Exception as e:
                ax.set_title(f"Grad-CAM Error:\n{str(e)[:25]}", fontsize=8)
        else:
            ax.axis('off')

    plt.tight_layout()
    plt.show()

# preprocess_input = tf.keras.applications.efficientnet.preprocess_input
# decode_predictions = tf.keras.applications.efficientnet.decode_predictions
#
# last_conv_layer_name = "conv5_block3_out"
#
# img_size = (224, 224, 3)
#
# # Remove last layer's softmax
# model.layers[-1].activation = None
# # %%
# import textwrap
#
# # Display the part of the pictures used by the neural network to classify the pictures
# fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(15, 10),
#                          subplot_kw={'xticks': [], 'yticks': []})
#
# for i, ax in enumerate(axes.flat):
#     img_path = test_df.Filepath.iloc[random_index[i]]
#     img_array = preprocess_input(get_img_array(img_path, size=img_size))
#     heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
#     cam_path = save_and_display_gradcam(img_path, heatmap)
#     ax.imshow(plt.imread(cam_path))
#
#     # Wrap the title text
#     title_text = f"True: {test_df.Label.iloc[random_index[i]]}\nPredicted: {pred[random_index[i]]}"
#     wrapped_title = '\n'.join(textwrap.wrap(title_text, 20))  # Adjust the number as needed
#     ax.set_title(wrapped_title)
#
# plt.tight_layout()
# plt.show()