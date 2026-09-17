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
    calculate_psnr,
    calculate_color_histograms
)

# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Smart Image Enhancement",
    page_icon="🖼️",
    layout="wide"
)


# =====================================================
# APPLICATION TITLE
# =====================================================

st.title("🖼️ Smart Image Enhancement and Restoration System")

st.write(
    "Enhance, restore, and analyze images using "
    "traditional Computer Vision techniques."
)


# =====================================================
# IMAGE UPLOAD
# =====================================================

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    try:
        # Validate file size (maximum 10 MB)
        if uploaded_file.size > 10 * 1024 * 1024:
            st.error("File size must be less than 10 MB.")
            st.stop()

        # Read uploaded image
        image = Image.open(uploaded_file).convert("RGB")

        image_np = np.array(image)

        # Validate image dimensions
        if image_np.size == 0:
            st.error("The uploaded image is empty.")
            st.stop()

        # Convert RGB to BGR for OpenCV
        image_bgr = cv2.cvtColor(
            image_np,
            cv2.COLOR_RGB2BGR
        )

    except Exception as error:
        st.error(f"Unable to process image: {error}")
        st.stop()
    # =================================================
    # SIDEBAR CONTROLS
    # =================================================

    st.sidebar.header("⚙️ Processing Controls")

    category = st.sidebar.selectbox(
        "Select Category",
        [
            "Original",
            "Enhancement",
            "Restoration"
        ]
    )


    # Default values
    operation = "Original"

    kernel_size = 5
    diameter = 9
    sigma_color = 75
    sigma_space = 75
    strength = 10


    # =================================================
    # ENHANCEMENT OPTIONS
    # =================================================

    if category == "Enhancement":

        operation = st.sidebar.selectbox(
            "Select Enhancement",
            [
                "Original",
                "Grayscale",
                "Brightness",
                "Contrast",
                "Histogram Equalization",
                "Sharpen"
            ]
        )

        if operation == "Brightness":

            brightness_value = st.sidebar.slider(
                "Brightness",
                -100,
                100,
                30
            )

        elif operation == "Contrast":

            contrast_value = st.sidebar.slider(
                "Contrast",
                0.5,
                3.0,
                1.5
            )


    # =================================================
    # RESTORATION OPTIONS
    # =================================================

    elif category == "Restoration":

        operation = st.sidebar.selectbox(
            "Select Restoration",
            [
                "Gaussian Blur",
                "Median Filter",
                "Bilateral Filter",
                "Image Denoising"
            ]
        )

        if operation in [
            "Gaussian Blur",
            "Median Filter"
        ]:

            kernel_size = st.sidebar.slider(
                "Kernel Size",
                3,
                11,
                5,
                step=2
            )


        elif operation == "Bilateral Filter":

            diameter = st.sidebar.slider(
                "Filter Diameter",
                3,
                15,
                9,
                step=2
            )

            sigma_color = st.sidebar.slider(
                "Sigma Color",
                10,
                150,
                75
            )

            sigma_space = st.sidebar.slider(
                "Sigma Space",
                10,
                150,
                75
            )


        elif operation == "Image Denoising":

            strength = st.sidebar.slider(
                "Denoising Strength",
                1,
                30,
                10
            )


    # =================================================
    # IMAGE PROCESSING
    # =================================================

    if operation == "Original":

        result = image_np


    elif operation == "Grayscale":

        result = convert_to_grayscale(
            image_bgr
        )


    elif operation == "Brightness":

        result = adjust_brightness(
            image_bgr,
            brightness_value
        )

        result = cv2.cvtColor(
            result,
            cv2.COLOR_BGR2RGB
        )


    elif operation == "Contrast":

        result = adjust_contrast(
            image_bgr,
            contrast_value
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
            image_bgr,
            kernel_size
        )

        result = cv2.cvtColor(
            result,
            cv2.COLOR_BGR2RGB
        )


    elif operation == "Median Filter":

        result = median_filter(
            image_bgr,
            kernel_size
        )

        result = cv2.cvtColor(
            result,
            cv2.COLOR_BGR2RGB
        )


    elif operation == "Bilateral Filter":

        result = bilateral_filter(
            image_bgr,
            diameter,
            sigma_color,
            sigma_space
        )

        result = cv2.cvtColor(
            result,
            cv2.COLOR_BGR2RGB
        )


    elif operation == "Image Denoising":

        result = denoise_image(
            image_bgr,
            strength
        )

        result = cv2.cvtColor(
            result,
            cv2.COLOR_BGR2RGB
        )


    # =================================================
    # IMAGE DISPLAY
    # =================================================

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


    # =================================================
    # DOWNLOAD PROCESSED IMAGE
    # =================================================

    st.divider()

    st.subheader("⬇️ Download Processed Image")


    if len(result.shape) == 3:

        result_bgr = cv2.cvtColor(
            result,
            cv2.COLOR_RGB2BGR
        )

    else:

        result_bgr = result


    success, encoded_image = cv2.imencode(
        ".png",
        result_bgr
    )


    if success:

        st.download_button(
            label="Download Image",
            data=encoded_image.tobytes(),
            file_name="processed_image.png",
            mime="image/png"
        )


    # =================================================
    # IMAGE ANALYSIS
    # =================================================

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


        # Histogram
        histogram = calculate_histogram(
            image_bgr
        )


        st.write("### Grayscale Histogram")


        st.line_chart(histogram)


    # =================================================
    # IMAGE COMPARISON METRICS
    # =================================================

    st.divider()

    st.subheader("📈 Image Comparison Metrics")


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
            "Comparison unavailable because "
            "image dimensions differ."
        )

# =====================================================
# HISTOGRAM COMPARISON
# =====================================================

st.divider()

st.subheader("📊 Histogram Comparison")

if st.button("Compare Histograms"):

    original_histograms = calculate_color_histograms(
        image_bgr
    )

    if len(result.shape) == 3:
        processed_bgr = cv2.cvtColor(
            result,
            cv2.COLOR_RGB2BGR
        )
    else:
        processed_bgr = cv2.cvtColor(
            result,
            cv2.COLOR_GRAY2BGR
        )

    processed_histograms = calculate_color_histograms(
        processed_bgr
    )

    histogram_col1, histogram_col2 = st.columns(2)

    with histogram_col1:

        st.write("Original Image Histogram")

        st.line_chart(original_histograms)

    with histogram_col2:

        st.write("Processed Image Histogram")

        st.line_chart(processed_histograms)