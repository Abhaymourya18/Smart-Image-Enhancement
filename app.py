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
from modules.analysis import (
    calculate_statistics,
    calculate_histogram,
    calculate_mse,
    calculate_psnr
)

# Page Configuration
st.set_page_config(
    page_title="Smart Image Enhancement",
    page_icon="🖼️",
    layout="wide"
)


# Application Title
st.title("🖼️ Smart Image Enhancement and Restoration System")

st.write(
    "Enhance, restore, and analyze images using "
    "traditional Computer Vision techniques."
)


# Image Upload
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    # Read Uploaded Image
    image = Image.open(uploaded_file).convert("RGB")

    image_np = np.array(image)

    # Convert RGB to BGR for OpenCV
    image_bgr = cv2.cvtColor(
        image_np,
        cv2.COLOR_RGB2BGR
    )


    # Operation Selection
    operation = st.selectbox(
        "Select Image Operation",
        [
            "Original",
            "Grayscale",
            "Brightness",
            "Contrast",
            "Histogram Equalization",
            "Sharpen",
            "Gaussian Blur",
            "Median Filter",
            "Bilateral Filter",
            "Image Denoising"
        ]
    )


    # Image Processing

    if operation == "Original":

        result = image_np


    elif operation == "Grayscale":

        result = convert_to_grayscale(image_bgr)


    elif operation == "Brightness":

        value = st.slider(
            "Brightness",
            -100,
            100,
            30
        )

        result = adjust_brightness(
            image_bgr,
            value
        )

        result = cv2.cvtColor(
            result,
            cv2.COLOR_BGR2RGB
        )


    elif operation == "Contrast":

        alpha = st.slider(
            "Contrast",
            0.5,
            3.0,
            1.5
        )

        result = adjust_contrast(
            image_bgr,
            alpha
        )

        result = cv2.cvtColor(
            result,
            cv2.COLOR_BGR2RGB
        )


    elif operation == "Histogram Equalization":

        result = histogram_equalization(
            image_bgr
        )


    elif operation == "Sharpen":

        result = sharpen_image(
            image_bgr
        )

        result = cv2.cvtColor(
            result,
            cv2.COLOR_BGR2RGB
        )


    elif operation == "Gaussian Blur":

        result = gaussian_blur(
            image_bgr
        )

        result = cv2.cvtColor(
            result,
            cv2.COLOR_BGR2RGB
        )


    elif operation == "Median Filter":

        result = median_filter(
            image_bgr
        )

        result = cv2.cvtColor(
            result,
            cv2.COLOR_BGR2RGB
        )


    elif operation == "Bilateral Filter":

        result = bilateral_filter(
            image_bgr
        )

        result = cv2.cvtColor(
            result,
            cv2.COLOR_BGR2RGB
        )


    elif operation == "Image Denoising":

        result = denoise_image(
            image_bgr
        )

        result = cv2.cvtColor(
            result,
            cv2.COLOR_BGR2RGB
        )


    # Display Original and Processed Images

    st.divider()

    col1, col2 = st.columns(2)


    with col1:

        st.subheader("Original Image")

        st.image(
            image_np,
            use_container_width=True
        )


    with col2:

        st.subheader("Processed Image")

        st.image(
            result,
            use_container_width=True
        )


    # Image Analysis Section

    st.divider()

    st.subheader("📊 Image Analysis")


    if st.button("Analyze Original Image"):

        statistics = calculate_statistics(
            image_bgr
        )


        st.write("### Image Statistics")


        metric_columns = st.columns(4)


        for column, (key, value) in zip(
            metric_columns,
            statistics.items()
        ):

            column.metric(
                key,
                value
            )


        # Histogram Calculation

        histogram = calculate_histogram(
            image_bgr
        )


        st.write("### Grayscale Histogram")


        st.line_chart(histogram)

# Image Comparison Metrics

st.divider()

st.subheader("📈 Image Comparison Metrics")

# Convert both images to grayscale
original_gray = cv2.cvtColor(
    image_bgr,
    cv2.COLOR_BGR2GRAY
)

if len(result.shape) == 3:
    processed_gray = cv2.cvtColor(
        result,
        cv2.COLOR_RGB2GRAY
    )
else:
    processed_gray = result

# Ensure matching dimensions
if original_gray.shape == processed_gray.shape:

    mse = calculate_mse(
        original_gray,
        processed_gray
    )

    psnr = calculate_psnr(
        original_gray,
        processed_gray
    )

    metric_col1, metric_col2 = st.columns(2)

    metric_col1.metric(
        "MSE",
        f"{mse:.2f}"
    )

    if np.isinf(psnr):
        psnr_display = "∞"
    else:
        psnr_display = f"{psnr:.2f} dB"

    metric_col2.metric(
        "PSNR",
        psnr_display
    )

else:
    st.warning(
        "Comparison unavailable because image dimensions differ."
    )