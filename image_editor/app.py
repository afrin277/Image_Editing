# ================= IMPORTS ================= 
import streamlit as st 
import cv2 
import numpy as np 
from filters import * 
from utils import * 
from PIL import Image
import matplotlib.pyplot as plt 

# ================= PAGE ================= 
st.set_page_config(page_title="Image Editor", layout="wide")

st.title("🖼️ Image Preprocessing App")

# ================= SIDEBAR ================= 
st.sidebar.header("Preprocessing Options") 

# Theme 
theme = st.sidebar.selectbox("Theme",["Light", "Dark"])

if theme == "Dark":
    st.info("🌙 For full dark theme, use Streamlit Settings ⚙️")

if theme == "Dark":

    st.markdown("""
    <style>

    .stApp {
        background-color: black;
        color: white;
    }

    </style>
    """, unsafe_allow_html=True)

# ================= SESSION STATE ================= 
if "original" not in st.session_state: 
    st.session_state.original = None 

if "processed" not in st.session_state: 
    st.session_state.processed = None
 
# ================= UPLOAD ================= 
uploaded = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded is not None:
    
    img = load_image(uploaded)

    st.session_state.original = img.copy()
    st.session_state.processed = img.copy()

    if st.session_state.original is None:
        st.warning("Please upload an image first")
        st.stop()

    original = st.session_state.original 
    img = cv2.cvtColor(original.copy(), cv2.COLOR_RGB2BGR)
    width = original.shape[1]
    height = original.shape[0]

rotate_check = st.sidebar.checkbox("Rotate Image") 
if rotate_check: 
    angle = st.sidebar.slider("Angle", 0, 360, 0) 

gray_check = st.sidebar.checkbox("Grayscale") 
blur_check = st.sidebar.checkbox("Blur") 
if blur_check: 
    k = st.sidebar.slider("Kernel Size", 1, 51, 5) 
    if k % 2 == 0:
        k += 1 

sharp_check = st.sidebar.checkbox("Sharpen") 
edge_detect = st.sidebar.checkbox("Edge Detection") 
brightness_check = st.sidebar.checkbox("Brightness") 
if brightness_check: 
    b = st.sidebar.slider("Brightness", -100, 100, 0)

contrast_check = st.sidebar.checkbox("Contrast") 
if contrast_check: 
    c = st.sidebar.slider("Contrast", 0.5, 3.0, 1.0) 

noise_check = st.sidebar.checkbox("Noise") 
flip_check = st.sidebar.checkbox("Flip Image") 
if flip_check:

    flip_direction = st.sidebar.selectbox(
        "Flip Direction",
        ["Horizontal", "Vertical"]
    )

# If no filter is selected → show original image
if st.session_state.original is not None:
    if not any([gray_check, blur_check, sharp_check, edge_detect, brightness_check, contrast_check, rotate_check, noise_check, flip_check]):
        st.session_state.processed = st.session_state.original.copy()

    if uploaded is None:
        st.warning("Please upload an image first")
        st.stop()

# ================= MAIN ================= 
original = st.session_state.original 
if original is None:
    st.stop()

img = cv2.cvtColor(original.copy(), cv2.COLOR_RGB2BGR)

# Apply filters step by step 
if gray_check: 
    img = grayscale(img) 

if blur_check: 
    img = gaussian_blur(img, k) 

if sharp_check: 
    img = sharpen(img) 
    
if edge_detect: 
    img = edge(img) 
            
if brightness_check: 
    img = adjust_brightness(img, b) 
    
if contrast_check: 
    img = adjust_contrast(img, c) 
    
if rotate_check:  
    img = rotate(img, angle) 

if noise_check: 
    img = add_noise(img) 

if flip_check:

    if flip_direction == "Horizontal":
        img = cv2.flip(img, 1)

    elif flip_direction == "Vertical":
        img = cv2.flip(img, 0)
                                
processed = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Save processed result 
st.session_state.processed = processed
        
# ================= DISPLAY ================= 
processed = st.session_state.processed
    
col1, col2 = st.columns(2)
with col1:
    st.subheader("Original Image (Before)")
    st.image(original)

with col2:
    st.subheader("Processed Image (After)")
    st.image(processed)

# ================= BEFORE vs AFTER (END) ================= 
original = st.session_state.get("original")
processed = st.session_state.get("processed")

if original is not None and processed is not None:

    st.subheader("Before vs After Comparison") 

    slider = st.slider("Compare", 0, 100, 50) 

# Ensure same size 
    processed_resized = processed.copy()
    h, w = original.shape[:2] 
    split = int(w * slider / 100) 

    combined = np.zeros_like(original) 
    combined[:, :split] = original[:, :split] 
    combined[:, split:] = processed_resized[:, split:] 

    st.image(combined, caption="Left = Original | Right = Processed") 

# ================= HISTOGRAM =================
if processed is not None: 
    if st.checkbox("Show Histogram"): 
        fig, ax = plt.subplots(figsize=(10, 5))

        # Convert image to grayscale for clean histogram
        gray_img = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)

        # Calculate histogram
        hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])

        # Plot histogram
        ax.plot(hist)

        ax.set_title("Pixel Intensity Distribution")
        ax.set_xlabel("Pixel Intensity (0-255)")
        ax.set_ylabel("Number of Pixels")

        ax.set_xlim([0, 256])
        ax.grid(True)
        st.pyplot(fig)

# ================= IMAGE INFO ================= 
original = st.session_state.get("original")

if original is not None:
    st.sidebar.subheader("Image Info")
    st.sidebar.write(f"Shape: {original.shape}")
    st.sidebar.write(f"Type: {original.dtype}")

# ================= DOWNLOAD ================= 
original = st.session_state.get("original")
processed = st.session_state.get("processed")

if original is not None: 
    orig_bytes = to_bytes(original.copy())
    st.download_button("Download Original", orig_bytes, "original.png",mime="image/png")

if processed is not None:
    proc_bytes = to_bytes(processed)
    st.download_button("Download Processed",proc_bytes,"processed.png",
    mime="image/png")