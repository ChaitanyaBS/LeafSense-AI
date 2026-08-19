# imports.py
"""
LeafSense AI - Global Imports Module
"""

# Suppress TensorFlow C++ startup logs (silences red text in PyCharm)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Import Data Science Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split

# TensorFlow / Keras Libraries
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2, ResNet50
from tensorflow.keras import Model

# System & Utility Libraries
from pathlib import Path
import os.path
import random
from collections import Counter
import textwrap

# Image Processing & Visualization Libraries
import matplotlib.cm as cm
import cv2
import seaborn as sns
from cycler import cycler
from PIL import Image, ImageChops, ImageEnhance

# Metrics & Evaluation
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, average_precision_score
from sklearn.preprocessing import LabelBinarizer
import itertools

# Set default styling
sns.set_style('darkgrid')