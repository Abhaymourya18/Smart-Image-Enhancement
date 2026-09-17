import cv2
import numpy as np


def calculate_statistics(image):
    """Calculate basic image intensity statistics."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    return {
        "Mean Intensity": round(float(np.mean(gray)), 2),
        "Minimum Intensity": int(np.min(gray)),
        "Maximum Intensity": int(np.max(gray)),
        "Standard Deviation": round(float(np.std(gray)), 2)
    }


def calculate_histogram(image):
    """Calculate grayscale histogram."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    histogram = cv2.calcHist(
        [gray],
        [0],
        None,
        [256],
        [0, 256]
    )

    return histogram.flatten()


def calculate_mse(original, processed):
    """Calculate Mean Squared Error."""
    original = original.astype(np.float32)
    processed = processed.astype(np.float32)

    return float(np.mean((original - processed) ** 2))


def calculate_psnr(original, processed):
    """Calculate Peak Signal-to-Noise Ratio."""
    mse = calculate_mse(original, processed)

    if mse == 0:
        return float("inf")

    return float(10 * np.log10((255 ** 2) / mse))