
import numpy as np
import cv2
import matplotlib.pyplot as plt

def segment_image(image_path, k):
    # Read in the image
    image = cv2.imread(image_path)
    
    if image is None:
        raise ValueError("Image not found or unable to read")
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Reshape image into 2D array of pixels and 3 color values (RGB)
    pixel_vals = image.reshape((-1, 3))
    
    # Convert to float type
    pixel_vals = np.float32(pixel_vals)
    
    # Define criteria for stopping the algorithm
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
    
    # Perform k-means clustering
    retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Convert data into 8-bit values
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]
    
    # Reshape data into the original image dimensions
    segmented_image = segmented_data.reshape(image.shape)
    
    return image, segmented_image


# streamlit_app.py
