import cv2


def gaussian_blur(image, kernel_size=5):
    return cv2.GaussianBlur(
        image,
        (kernel_size, kernel_size),
        0
    )


def median_filter(image, kernel_size=5):
    return cv2.medianBlur(image, kernel_size)


def bilateral_filter(
    image,
    diameter=9,
    sigma_color=75,
    sigma_space=75
):
    return cv2.bilateralFilter(
        image,
        diameter,
        sigma_color,
        sigma_space
    )


def denoise_image(image, strength=10):
    return cv2.fastNlMeansDenoisingColored(
        image,
        None,
        strength,
        strength,
        7,
        21
    )