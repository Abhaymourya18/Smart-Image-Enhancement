import cv2
import numpy as np


def convert_to_grayscale(image):
    """Convert a color image to grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def adjust_brightness(image, value=30):
    """Increase or decrease image brightness."""
    return cv2.convertScaleAbs(image, alpha=1.0, beta=value)


def adjust_contrast(image, alpha=1.5):
    """Adjust image contrast."""
    return cv2.convertScaleAbs(image, alpha=alpha, beta=0)


def histogram_equalization(image):
    """Improve contrast using histogram equalization."""
    gray = convert_to_grayscale(image)
    return cv2.equalizeHist(gray)


def sharpen_image(image):
    """Sharpen image using a convolution kernel."""
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])

    return cv2.filter2D(image, -1, kernel)