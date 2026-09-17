import cv2


def gaussian_blur(image, kernel_size=5):
    """Reduce noise using Gaussian blur."""
    return cv2.GaussianBlur(
        image,
        (kernel_size, kernel_size),
        0
    )


def median_filter(image, kernel_size=5):
    """Remove salt-and-pepper noise."""
    return cv2.medianBlur(image, kernel_size)


def bilateral_filter(image):
    """Reduce noise while preserving edges."""
    return cv2.bilateralFilter(
        image,
        d=9,
        sigmaColor=75,
        sigmaSpace=75
    )


def denoise_image(image):
    """Perform advanced image denoising."""
    return cv2.fastNlMeansDenoisingColored(
        image,
        None,
        10,
        10,
        7,
        21
    )