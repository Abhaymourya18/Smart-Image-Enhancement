import streamlit as st
import cv2
import numpy as np
from PIL import Image

from modules.enhancement import (
    convert_to_grayscale,
    adjust_brightness,
    adjust_contrast,
    histogram_equalization,
    sharpen_image
)
from modules.restoration import (
    gaussian_blur,
    median_filter,
    bilateral_filter,
    denoise_image
)
st.set_page_config(
    page_title="Smart Image Enhancement",
    layout="wide"
)

st.title("🖼️ Smart Image Enhancement System")
st.write("Enhance images using traditional computer vision techniques.")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    operation = st.selectbox(
        "Select Enhancement Operation",
        [
            "Original",
            "Grayscale",
            "Brightness",
            "Contrast",
            "Histogram Equalization",
            "Sharpen"
            "Gaussian Blur",
"Median Filter",
"Bilateral Filter",
"Image Denoising"
        ]
    )

    if operation == "Original":
        result = image_np

    elif operation == "Grayscale":
        result = convert_to_grayscale(image_bgr)

    elif operation == "Brightness":
        value = st.slider("Brightness", -100, 100, 30)
        result = adjust_brightness(image_bgr, value)
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    elif operation == "Contrast":
        alpha = st.slider("Contrast", 0.5, 3.0, 1.5)
        result = adjust_contrast(image_bgr, alpha)
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    elif operation == "Histogram Equalization":
        result = histogram_equalization(image_bgr)

    elif operation == "Sharpen":
        result = sharpen_image(image_bgr)
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    elif operation == "Gaussian Blur":
        result = gaussian_blur(image_bgr)
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    elif operation == "Median Filter":
       result = median_filter(image_bgr)
       result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    elif operation == "Bilateral Filter":
       result = bilateral_filter(image_bgr)
       result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    elif operation == "Image Denoising":
       result = denoise_image(image_bgr)
       result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image_np, use_container_width=True)

    with col2:
        st.subheader("Enhanced Image")
        st.image(result, use_container_width=True)