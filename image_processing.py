# image_processing.py
"""
LeafSense AI - Image Processing & Error Level Analysis (ELA) Module
Performs ELA detection to highlight compression differences and image integrity.
"""

import os
import random
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageChops, ImageEnhance

def compute_ela_cv(path: str, quality: int = 90, scale: float = 15.0):
    """
    Computes Error Level Analysis (ELA) image matrix using OpenCV.
    """
    temp_filename = 'temp_file_name.jpeg'
    orig_img = cv2.imread(path)
    if orig_img is None:
        raise FileNotFoundError(f"Could not load image at path: {path}")

    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(temp_filename, cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])

    compressed_img = cv2.imread(temp_filename)
    compressed_img = cv2.cvtColor(compressed_img, cv2.COLOR_BGR2RGB)

    diff = scale * cv2.absdiff(orig_img, compressed_img)
    if os.path.exists(temp_filename):
        os.remove(temp_filename)

    return diff

def convert_to_ela_image(path: str, quality: int = 90):
    """
    Converts image to Error Level Analysis (ELA) format using PIL.
    """
    temp_filename = 'temp_file_name.jpeg'
    image = Image.open(path).convert('RGB')
    image.save(temp_filename, 'JPEG', quality=quality)
    temp_image = Image.open(temp_filename)

    ela_image = ImageChops.difference(image, temp_image)

    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1

    scale = 255.0 / max_diff
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

    if os.path.exists(temp_filename):
        os.remove(temp_filename)

    return ela_image

def random_sample(path: str, extension: str = None) -> str:
    """
    Selects a random image filepath from directory.
    """
    image_path = Path(path)
    if extension:
        items = list(image_path.glob(f'*.{extension}'))
    else:
        items = (
            list(image_path.glob('*.jpg')) +
            list(image_path.glob('*.png')) +
            list(image_path.glob('*.jpeg'))
        )

    if not items:
        raise FileNotFoundError(f"No matching images found in '{path}'.")

    p = random.choice(items)
    return p.as_posix()

def plot_ela_analysis(image_path: str):
    """
    Displays Error Level Analysis at 9 varying JPEG compression levels.
    """
    orig = cv2.imread(image_path)
    if orig is None:
        print(f"[ERROR] Unable to read image for ELA analysis: {image_path}")
        return

    orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB) / 255.0
    init_val = 100
    columns = 3
    rows = 3

    fig = plt.figure(figsize=(14, 9))
    fig.suptitle('Error Level Analysis (ELA)', fontsize=18, fontweight='bold')

    for i in range(1, columns * rows + 1):
        quality = init_val - (i - 1) * 8
        try:
            img = compute_ela_cv(path=image_path, quality=quality)
            if i == 1:
                img = orig.copy()
            ax = fig.add_subplot(rows, columns, i)
            ax.title.set_text(f'Quality: {quality}')
            plt.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
        except Exception as e:
            print(f"ELA error at quality {quality}: {e}")

    plt.tight_layout()
    plt.show()

# def compute_ela_cv(path, quality):
#     temp_filename = 'temp_file_name.jpeg'
#     SCALE = 15
#     orig_img = cv2.imread(path)
#     orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
#
#     cv2.imwrite(temp_filename, orig_img, [cv2.IMWRITE_JPEG_QUALITY, quality])
#
#     # read compressed image
#     compressed_img = cv2.imread(temp_filename)
#
#     # get absolute difference between img1 and img2 and multiply by scale
#     diff = SCALE * cv2.absdiff(orig_img, compressed_img)
#     return diff
#
# def convert_to_ela_image(path, quality):
#     temp_filename = 'temp_file_name.jpeg'
#     ela_filename = 'temp_ela.png'
#     image = Image.open(path).convert('RGB')
#     image.save(temp_filename, 'JPEG', quality=quality)
#     temp_image = Image.open(temp_filename)
#
#     ela_image = ImageChops.difference(image, temp_image)
#
#     extrema = ela_image.getextrema()
#     max_diff = max([ex[1] for ex in extrema])
#     if max_diff == 0:
#         max_diff = 1
#
#     scale = 255.0 / max_diff
#     ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
#
#     return ela_image
#
# def random_sample(path, extension=None):
#     if extension:
#         items = Path(path).glob(f'*.{extension}')
#     else:
#         items = Path(path).glob(f'*')
#
#     items = list(items)
#
#     p = random.choice(items)
#     return p.as_posix()
#
# # View random sample from the dataset
# p = random_sample('/kaggle/input/omdena-mango-leaf/MangoLeafBD_Without_Testset_Augmentation/Train/Anthracnose')
# orig = cv2.imread(p)
# orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB) / 255.0
# init_val = 100
# columns = 3
# rows = 3
#
# fig = plt.figure(figsize=(15, 10))
# fig.suptitle('Error Level Analysis', fontsize=20)  # Add super title
# for i in range(1, columns * rows + 1):
#     quality = init_val - (i - 1) * 8
#     img = compute_ela_cv(path=p, quality=quality)
#     if i == 1:
#         img = orig.copy()
#     ax = fig.add_subplot(rows, columns, i)
#     ax.title.set_text(f'q: {quality}')
#     plt.imshow(img)
#     ax.set_xticks([])  # Remove x-ticks
#     ax.set_yticks([])  # Remove y-ticks
# plt.show()
