import streamlit as st
from PIL import Image
import numpy as np
import image_segmentation
import cv2
import tempfile

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
            
            original, segmented = image_segmentation.segment_image(temp_filename, k)
            
            # Display segmented image
            st.image(segmented, caption=f"Segmented Image with {k} clusters", use_column_width=True)

if __name__ == "__main__":
    main()
