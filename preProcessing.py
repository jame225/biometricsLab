"""
By: James Hassel
This file will preprocess the images to ensure gray scale and import the images into lists for manipulation.
"""
import os
import cv2
import numpy as np


def preprocess(image_path):
    """
    Helper function to preprocess images
    :param image_path: path to specific image
    :return: preprocessed image
    """
    # Load the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = img.astype(np.float32)
    # Normalize pixel values (optional)
    img = img / 255.0
    return img


def loadImages(dataset_path):
    """
    Loads the images into a list of paired images
    :param dataset_path: path to 1500 train images or 500 test images
    :return: list of images [f0, s0, f1, s1, ...]
    """
    images = []
    # Get all filenames in the train/test directory
    for prefix in ["f", "s"]:  # Loop for f and s files
        file_list = sorted([
            f for f in os.listdir(dataset_path)
            if f.startswith(prefix) and f.endswith('.png')  # Ensure valid files
        ])
        print(f"Found {len(file_list)} files starting with '{prefix}' in {dataset_path}")  # Debugging output
        for file_name in file_list:
            file_path = os.path.join(dataset_path, file_name)
            try:
                processed_image = preprocess(file_path)
                images.append(processed_image)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")  # Catch and log errors
    return images

