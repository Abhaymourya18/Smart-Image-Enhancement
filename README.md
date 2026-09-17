# Smart Image Enhancement and Restoration System

## 1. Project Overview

The Smart Image Enhancement and Restoration System is a computer vision application developed using Python and OpenCV.

The system allows users to enhance, restore, and analyze images using traditional image processing techniques without using AI or machine learning.

## 2. Objectives

- Improve image brightness and contrast.
- Convert images into grayscale.
- Enhance image sharpness.
- Reduce image noise.
- Apply different image restoration filters.
- Analyze image quality using statistical metrics.
- Compare original and processed images.

## 3. Technologies Used

- Python
- OpenCV
- NumPy
- Streamlit
- Matplotlib
- Pillow
- scikit-image

## 4. Features

### Image Enhancement

- Grayscale Conversion
- Brightness Adjustment
- Contrast Adjustment
- Histogram Equalization
- Image Sharpening
- Emboss Effect
- Edge Detection

### Image Restoration

- Gaussian Blur
- Median Filter
- Bilateral Filter
- Non-Local Means Denoising

### Image Analysis

- Mean Intensity
- Minimum Intensity
- Maximum Intensity
- Standard Deviation
- Grayscale Histogram
- Color Histogram
- Mean Squared Error (MSE)
- Peak Signal-to-Noise Ratio (PSNR)

### User Interface

- Image Upload
- Interactive Before-and-After Comparison
- Processed Image Download
- Processing History
- Reset Application

## 5. Project Structure

```text
Smart-Image-Enhancement/
│
├── app.py
├── README.md
├── statement.md
├── requirements.txt
│
├── modules/
│   ├── __init__.py
│   ├── enhancement.py
│   ├── restoration.py
│   └── analysis.py
│
├── utils/
│   ├── __init__.py
│   └── image_utils.py
│
├── tests/
│   └── test_processing.py
│
└── sample_images/

6. Installation

Clone the repository:

git clone https://github.com/Abhaymourya18/Smart-Image-Enhancement.git

Navigate to the project directory:

cd Smart-Image-Enhancement

Install dependencies:

pip install -r requirements.txt
7. Run the Application
streamlit run app.py

The application will open in the browser.

8. Methodology

The application uses traditional digital image processing techniques:

Upload an image.
Select an enhancement or restoration operation.
Process the image using OpenCV.
Display the original and processed images.
Analyze image statistics and histograms.
Download the processed image.
9. Evaluation Metrics
Mean Squared Error (MSE)

MSE measures the average squared difference between the original and processed images.

Lower MSE generally indicates greater similarity.

Peak Signal-to-Noise Ratio (PSNR)

PSNR measures the quality of the processed image relative to the original image.

Higher PSNR generally indicates greater similarity.

10. Applications
Image preprocessing
Photography enhancement
Noise reduction
Computer vision applications
Image quality analysis
Digital image restoration
11. Limitations
Processing is based on traditional image processing techniques.
Results depend on selected filter parameters.
Excessive enhancement may introduce artifacts.
The system does not use deep learning-based restoration.
12. Future Scope
Add more enhancement techniques.
Support batch image processing.
Add advanced image segmentation.
Add video enhancement.
Improve automated parameter selection.
13. Conclusion

The project demonstrates the application of traditional computer vision techniques for image enhancement, restoration, and analysis. It provides an interactive interface for experimenting with different image processing methods and evaluating their effects.


### Save and push

```bash
git add README.md
git commit -m "Complete project documentation"
git push