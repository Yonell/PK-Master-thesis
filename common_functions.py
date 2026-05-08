def blackout_pixels_with_text_and_values(image, specified_values):
    import numpy as np
    import cv2
    import easyocr
    reader = easyocr.Reader(['en'])

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    results = reader.readtext(gray)
    for (bbox, text, prob) in results:
        (top_left, _, bottom_right, _) = bbox
        top_left = (int(top_left[0]), int(top_left[1]))
        bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
        cv2.rectangle(image, top_left, bottom_right, (0, 0, 0), -1)
    
    mask = np.isin(image, specified_values)
    image[mask] = 0

    return image

def suppress_outliers_and_normalize_image(image, lower_percentile=1, upper_percentile=97):
    import numpy as np

    lower_value = np.percentile(image, lower_percentile)
    upper_value = np.percentile(image, upper_percentile)

    clipped_image = np.clip(image, lower_value, upper_value)

    normalized_image = (clipped_image - lower_value) / (upper_value - lower_value) * 255.0

    return normalized_image

def remove_white_top_hat(image, kernel_size=(5, 5)):
    import cv2
    import numpy as np

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    top_hat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)

    return image - top_hat

def morphex_close(image, kernel_size=(5, 5)):
    import cv2
    import numpy as np

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)

    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

def non_local_means_denoising(image, h=10, templateWindowSize=7, searchWindowSize=21):
    import cv2
    return cv2.fastNlMeansDenoising(image, None, h, templateWindowSize, searchWindowSize)

def apply_clahe(image, clipLimit=2.0, tileGridSize=(8, 8)):
    import cv2
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    return clahe.apply(image)

def apply_determinant_of_hessian_mask(image_gray, min_sigma=1, max_sigma=15, threshold=0.01):
    import numpy as np
    import cv2
    from skimage.feature import blob_doh
    blobs = blob_doh(image_gray, min_sigma=min_sigma, max_sigma=max_sigma, threshold=threshold)
    
    mask = np.zeros_like(image_gray, dtype=np.uint8)
    
    for blob in blobs:
        y, x, r = blob
        cv2.circle(mask, (int(x), int(y)), int(r), 255, -1)
        
    return mask

def replace_outliers_with_surrounding_mean(image, lower_percentile=1, upper_percentile=99):
    import numpy as np
    lower_value = np.percentile(image, lower_percentile)
    upper_value = np.percentile(image, upper_percentile)

    outliers_mask = (image < lower_value) | (image > upper_value)
    image[outliers_mask] = np.percentile(image, 10)

    return image

def rank(image):
    import skimage.filters.rank
    from skimage.morphology import disk
    from skimage.util import img_as_ubyte
    from skimage.exposure import rescale_intensity
    
    if image.dtype.kind == 'f':
        image = rescale_intensity(image, in_range='image', out_range=(0.0, 1.0))
        
    image = img_as_ubyte(image)
    image = skimage.filters.rank.equalize(image, disk(40))
    return image

def pipeline(img, in_situ=False):
    if not in_situ:
        img = img.copy()
    img = blackout_pixels_with_text_and_values(img, specified_values=[0, 96])
    img = replace_outliers_with_surrounding_mean(img, lower_percentile=1, upper_percentile=97)
    img = apply_clahe(img, clipLimit=2.0, tileGridSize=(4, 4))
    img = non_local_means_denoising(img, h=20, templateWindowSize=7, searchWindowSize=21)
    img = suppress_outliers_and_normalize_image(img, lower_percentile=1, upper_percentile=99)
    return img