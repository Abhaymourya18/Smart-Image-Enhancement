import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt

from streamlit_image_comparison import image_comparison

from modules.enhancement import (
    convert_to_grayscale,
    adjust_brightness,
    adjust_contrast,
    histogram_equalization,
    sharpen_image,
    emboss_image,
    detect_edges
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


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Smart Image Enhancement",
    page_icon="🖼️",
    layout="wide"
)

st.title("🖼️ Smart Image Enhancement and Restoration System")

st.write(
    "Enhance, restore, and analyze images using traditional "
    "computer vision techniques with OpenCV."
)


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history = []

if "reset_counter" not in st.session_state:
    st.session_state.reset_counter = 0


# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"],
    help="Maximum file size: 10 MB"
)

if uploaded_file is None:
    st.info("Please upload an image to begin.")
    st.stop()

if uploaded_file.size > 10 * 1024 * 1024:
    st.error("File size must be less than 10 MB.")
    st.stop()

try:
    image_pil = Image.open(uploaded_file).convert("RGB")
    image_rgb = np.array(image_pil)

    image_bgr = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2BGR
    )

except Exception as e:
    st.error(f"Error loading image: {e}")
    st.stop()

st.success("Image uploaded successfully!")


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.header("Processing Options")

if st.sidebar.button("Reset Application"):

    st.session_state.history = []
    st.session_state.reset_counter += 1

    st.rerun()

category = st.sidebar.selectbox(
    "Select Category",
    ["Original", "Enhancement", "Restoration"]
)

operation = "Original"


# --------------------------------------------------
# ENHANCEMENT OPTIONS
# --------------------------------------------------

if category == "Enhancement":

    operation = st.sidebar.selectbox(
        "Select Enhancement",
        [
            "Grayscale",
            "Brightness",
            "Contrast",
            "Histogram Equalization",
            "Sharpen",
            "Emboss",
            "Edge Detection"
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
            1.5,
            0.1
        )


# --------------------------------------------------
# RESTORATION OPTIONS
# --------------------------------------------------

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

    if operation in ["Gaussian Blur", "Median Filter"]:

        kernel_size = st.sidebar.slider(
            "Kernel Size",
            3,
            15,
            5,
            step=2
        )

    elif operation == "Bilateral Filter":

        diameter = st.sidebar.slider(
            "Diameter",
            5,
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

        denoise_strength = st.sidebar.slider(
            "Denoising Strength",
            1,
            30,
            10
        )


# --------------------------------------------------
# IMAGE PROCESSING
# --------------------------------------------------

result = image_rgb.copy()

if operation == "Original":

    result = image_rgb.copy()

elif operation == "Grayscale":

    result = convert_to_grayscale(image_bgr)

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

    result = histogram_equalization(image_bgr)

elif operation == "Sharpen":

    result = sharpen_image(image_bgr)

    result = cv2.cvtColor(
        result,
        cv2.COLOR_BGR2RGB
    )

elif operation == "Emboss":

    result = emboss_image(image_bgr)

    result = cv2.cvtColor(
        result,
        cv2.COLOR_BGR2RGB
    )

elif operation == "Edge Detection":

    result = detect_edges(image_bgr)

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
        denoise_strength
    )

    result = cv2.cvtColor(
        result,
        cv2.COLOR_BGR2RGB
    )


# --------------------------------------------------
# PROCESSING HISTORY
# --------------------------------------------------

if operation != "Original":

    if (
        not st.session_state.history
        or st.session_state.history[-1] != operation
    ):

        st.session_state.history.append(operation)

st.sidebar.subheader("Processing History")

if st.session_state.history:

    for index, item in enumerate(
        st.session_state.history,
        start=1
    ):

        st.sidebar.write(f"{index}. {item}")

else:

    st.sidebar.write("No processing history yet.")

if st.sidebar.button("Clear History"):

    st.session_state.history = []

    st.rerun()


# --------------------------------------------------
# INTERACTIVE IMAGE COMPARISON
# --------------------------------------------------

st.subheader("Interactive Image Comparison")

if len(result.shape) == 2:

    result_for_slider = cv2.cvtColor(
        result,
        cv2.COLOR_GRAY2RGB
    )

else:

    result_for_slider = result

image_comparison(
    img1=image_rgb,
    img2=result_for_slider,
    label1="Original",
    label2=operation,
    width=700
)


# --------------------------------------------------
# DOWNLOAD PROCESSED IMAGE
# --------------------------------------------------

st.subheader("Download Processed Image")

if len(result.shape) == 2:

    result_to_save = result

else:

    result_to_save = cv2.cvtColor(
        result,
        cv2.COLOR_RGB2BGR
    )

success, encoded_image = cv2.imencode(
    ".png",
    result_to_save
)

if success:

    download_bytes = io.BytesIO(
        encoded_image.tobytes()
    )

    st.download_button(
        label="Download Image",
        data=download_bytes,
        file_name="processed_image.png",
        mime="image/png"
    )


# --------------------------------------------------
# IMAGE ANALYSIS
# --------------------------------------------------

st.subheader("Image Analysis")

if st.button("Show Image Statistics"):

    original_statistics = calculate_statistics(
        image_bgr
    )

    processed_statistics = calculate_statistics(
        result
    )

    col1, col2 = st.columns(2)

    with col1:

        st.write("### Original Image Statistics")

        for key, value in original_statistics.items():

            st.write(f"**{key}:** {value}")

    with col2:

        st.write("### Processed Image Statistics")

        for key, value in processed_statistics.items():

            st.write(f"**{key}:** {value}")

    # Grayscale histogram

    st.write("### Grayscale Histogram Comparison")

    original_histogram = calculate_histogram(
        image_bgr
    )

    processed_histogram = calculate_histogram(
        result
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 4)
    )

    axes[0].plot(original_histogram)

    axes[0].set_title(
        "Original Image Histogram"
    )

    axes[0].set_xlabel(
        "Pixel Intensity"
    )

    axes[0].set_ylabel(
        "Frequency"
    )

    axes[0].set_xlim(0, 255)

    axes[1].plot(processed_histogram)

    axes[1].set_title(
        "Processed Image Histogram"
    )

    axes[1].set_xlabel(
        "Pixel Intensity"
    )

    axes[1].set_ylabel(
        "Frequency"
    )

    axes[1].set_xlim(0, 255)

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)


# --------------------------------------------------
# MSE AND PSNR
# --------------------------------------------------

st.subheader("Quality Metrics")

try:

    original_gray = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2GRAY
    )

    if len(result.shape) == 3:

        processed_gray = cv2.cvtColor(
            result,
            cv2.COLOR_RGB2GRAY
        )

    else:

        processed_gray = result

    mse_value = calculate_mse(
        original_gray,
        processed_gray
    )

    psnr_value = calculate_psnr(
        original_gray,
        processed_gray
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Mean Squared Error (MSE)",
            f"{mse_value:.2f}"
        )

    with col2:

        if np.isinf(psnr_value):

            st.metric(
                "PSNR",
                "Infinite"
            )

        else:

            st.metric(
                "PSNR",
                f"{psnr_value:.2f} dB"
            )

    st.info(
        "MSE measures the difference between the original "
        "and processed images. Lower MSE generally indicates "
        "less difference."
    )

    st.info(
        "PSNR measures image quality. Higher PSNR generally "
        "indicates greater similarity to the original image."
    )

except Exception as e:

    st.warning(
        f"Quality metrics could not be calculated: {e}"
    )


# --------------------------------------------------
# COLOR HISTOGRAM COMPARISON
# --------------------------------------------------

st.subheader("Color Histogram Comparison")

if st.button("Show Color Histograms"):

    if len(result.shape) == 2:

        st.warning(
            "Color histograms are not available "
            "for grayscale images."
        )

    else:

        original_color_histograms = (
            calculate_color_histograms(image_bgr)
        )

        processed_bgr = cv2.cvtColor(
            result,
            cv2.COLOR_RGB2BGR
        )

        processed_color_histograms = (
            calculate_color_histograms(processed_bgr)
        )

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(12, 4)
        )

        # Original histogram

        axes[0].plot(
            original_color_histograms["b"],
            label="Blue"
        )

        axes[0].plot(
            original_color_histograms["g"],
            label="Green"
        )

        axes[0].plot(
            original_color_histograms["r"],
            label="Red"
        )

        axes[0].set_title(
            "Original Color Histogram"
        )

        axes[0].set_xlabel(
            "Pixel Intensity"
        )

        axes[0].set_ylabel(
            "Frequency"
        )

        axes[0].set_xlim(0, 255)

        axes[0].legend()

        # Processed histogram

        axes[1].plot(
            processed_color_histograms["b"],
            label="Blue"
        )

        axes[1].plot(
            processed_color_histograms["g"],
            label="Green"
        )

        axes[1].plot(
            processed_color_histograms["r"],
            label="Red"
        )

        axes[1].set_title(
            "Processed Color Histogram"
        )

        axes[1].set_xlabel(
            "Pixel Intensity"
        )

        axes[1].set_ylabel(
            "Frequency"
        )

        axes[1].set_xlim(0, 255)

        axes[1].legend()

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)