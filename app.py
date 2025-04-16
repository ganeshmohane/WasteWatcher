import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image

# Load the trained YOLOv8 model
model = YOLO(r"D:\Desktop\AAI Mini\waste-detection\waste-detector.pt")

# Streamlit UI Title and Image Upload Section
st.set_page_config(page_title="🗑️👀 Waste Watcher", page_icon="🗑️", layout="centered")
st.markdown("""
    <style>
    .centered-text {
        text-align: center;
    }
    .centered-content {
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
    }
    </style>
    """, unsafe_allow_html=True)

# Center the title
st.markdown('<div class="centered-text"><h1>🗑️👀 Waste Watcher</h1></div>', unsafe_allow_html=True)

st.markdown("""
    <div class="centered-text">
    A prototype that detects garbage in public spaces using YOLOv8 model, 
    integrated with CCTV cameras. It can detect trash and send alerts to local authorities.
    </div>
""", unsafe_allow_html=True)

# File upload section
uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Center the file uploader section
    st.markdown('<div class="centered-content">', unsafe_allow_html=True)
    
    # Load the image
    image = Image.open(uploaded_image)
    #st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess image for model inference
    image = np.array(image)
    
    # Run YOLO model for inference
    results = model(image)
    # Get annotated image (with bounding boxes and labels)
    annotated_image = results[0].plot()

    # Display it
    st.image(annotated_image, caption="Detected Garbage", use_container_width=True)

    # Show detections and send alert if garbage is detected
    if results[0].boxes is not None and len(results[0].boxes) > 0:
        # If objects are detected (garbage), display alert message
        st.warning("🚨 **Garbage Detected!** 🚨")
        
        # Display fixed location (Navi Mumbai, India)
        st.write("Location: Navi Mumbai, India")
        st.write("Sending image and alert to NMMC authorities... 🚛")
        
        # Display success message simulating the alert being sent
        st.success("✅ Alert sent with location details!")
    else:
        st.success("✅ No Garbage Detected!")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer with GitHub icon and contribution message
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center;">
        Contribute or raise a PR on 
        <a href="https://github.com/ganeshmohane" target="_blank">
            <img src="https://img.icons8.com/ios-glyphs/30/000000/github.png" 
            style="vertical-align:middle; margin-right:8px;"/>
            Ganesh Mohane's GitHub
        </a>
    </div>
    """,
    unsafe_allow_html=True
)
