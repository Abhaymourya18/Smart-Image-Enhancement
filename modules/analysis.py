import cv2
import numpy as np


def calculate_statistics(image):
    """Calculate basic image intensity statistics."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return {
        "Mean Intensity": round(float(np.mean(gray)), 2),
        "Minimum Intensity": int(np.min(gray)),
        "Maximum Intensity": int(np.max(gray)),
        "Standard Deviation": round(float(np.std(gray)), 2)
    }


def calculate_histogram(image):
    """Calculate grayscale histogram."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    histogram = cv2.calcHist(
        [gray],
        [0],
        None,
        [256],
        [0, 256]
    )

    return histogram.flatten()