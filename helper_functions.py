# helper_functions.py
"""
LeafSense AI - Helper Functions Module
Provides utility functions for TensorBoard logging, plotting loss curves,
unzipping datasets, comparing training histories, directory walking, and single image prediction.
"""

import os
import datetime
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def create_tensorboard_callback(dir_name: str, experiment_name: str):
    """
    Creates a TensorBoard callback instance to store log files.
    """
    log_dir = os.path.join(dir_name, experiment_name, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    print(f"[INFO] Saving TensorBoard log files to: '{log_dir}'")
    return tensorboard_callback

def plot_loss_curves(history):
    """
    Returns separate loss and accuracy curves for training and validation metrics.
    """
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    accuracy = history.history.get('accuracy', history.history.get('acc'))
    val_accuracy = history.history.get('val_accuracy', history.history.get('val_acc'))

    epochs = range(len(history.history['loss']))

    plt.figure(figsize=(14, 5))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='Training Loss', color='blue')
    plt.plot(epochs, val_loss, label='Validation Loss', color='red')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label='Training Accuracy', color='blue')
    plt.plot(epochs, val_accuracy, label='Validation Accuracy', color='red')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.tight_layout()
    plt.show()

def unzip_data(filename: str):
    """
    Unzips filename into the current working directory.
    """
    zip_ref = zipfile.ZipFile(filename, 'r')
    zip_ref.extractall()
    zip_ref.close()
    print(f"[INFO] Unzipped '{filename}' successfully.")

def compare_historys(original_history, new_history, initial_epochs=5):
    """
    Compares two TensorFlow History objects.
    """
    acc = original_history.history['accuracy']
    loss = original_history.history['loss']

    val_acc = original_history.history['val_accuracy']
    val_loss = original_history.history['val_loss']

    total_acc = acc + new_history.history['accuracy']
    total_loss = loss + new_history.history['loss']

    total_val_acc = val_acc + new_history.history['val_accuracy']
    total_val_loss = val_loss + new_history.history['val_loss']

    plt.figure(figsize=(14, 8))
    plt.subplot(2, 1, 1)
    plt.plot(total_acc, label='Training Accuracy')
    plt.plot(total_val_acc, label='Validation Accuracy')
    plt.plot([initial_epochs-1, initial_epochs-1], plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(total_loss, label='Training Loss')
    plt.plot(total_val_loss, label='Validation Loss')
    plt.plot([initial_epochs-1, initial_epochs-1], plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.tight_layout()
    plt.show()

def walk_through_dir(dir_path: str):
    """
    Walks through dir_path printing number of subdirectories and files.
    """
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

def pred_and_plot(model, filename: str, class_names: list, target_size=(224, 224)):
    """
    Imports an image located at filename, makes a prediction on it with
    a trained model, and plots the image with the predicted class as the title.
    """
    img = tf.keras.preprocessing.image.load_img(filename, target_size=target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    pred = model.predict(img_array)

    if len(pred[0]) > 1:
        pred_class = class_names[tf.argmax(pred[0])]
    else:
        pred_class = class_names[int(tf.round(pred[0]))]

    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.title(f"Prediction: {pred_class}")
    plt.axis(False)
    plt.show()