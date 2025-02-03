import streamlit as st
from PIL import Image
import numpy as np
import cv2
import tempfile
import matplotlib

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

def main():
    st.title("Image Segmentation App")
    
    uploaded_file = st.file_uploader("Upload a PNG image", type=["png"])
    
    if uploaded_file is not None:
        # Load the image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        k = st.slider("Select number of clusters", min_value=2, max_value=50, value=3, step=1)
        
        if st.button("Segment Image"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_filename = temp_file.name
            
            original, segmented = segment_image(temp_filename, k)
            
            # Display segmented image
            st.image(segmented, caption=f"Segmented Image with {k} clusters", use_column_width=True)

if __name__ == "__main__":
    main()
