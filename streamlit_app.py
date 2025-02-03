import streamlit as st
from PIL import Image
import numpy as np
import tempfile
import scipy.cluster.vq as vq

def segment_image(image_path, k):
    # Read in the image
    image = Image.open(image_path)
    image = image.convert("RGB")  # Ensure it's in RGB mode
    image_np = np.array(image)
    
    # Reshape image into 2D array of pixels and 3 color values (RGB)
    pixel_vals = image_np.reshape((-1, 3))
    
    # Convert to float type
    pixel_vals = np.float32(pixel_vals)
    
    # Perform k-means clustering
    centers, labels = vq.kmeans2(pixel_vals, k, iter=10, minit='random')
    
    # Convert data into 8-bit values
    centers = np.uint8(centers)
    segmented_data = centers[labels]
    
    # Reshape data into the original image dimensions
    segmented_image = segmented_data.reshape(image_np.shape)
    
    return image_np, segmented_image

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