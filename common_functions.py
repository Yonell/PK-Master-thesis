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

def find_ellipses_in_mask(mask):
    import cv2
    import numpy as np

    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    sure_bg = cv2.dilate(opening, kernel, iterations=2)

    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    mask_color = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(mask_color, markers)

    ellipses = []
    for label in np.unique(markers):
        if label == -1 or label == 1:
            continue

        label_mask = np.zeros(mask.shape, dtype="uint8")
        label_mask[markers == label] = 255

        contours, _ = cv2.findContours(label_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if len(contours) > 0:
            contour = max(contours, key=cv2.contourArea)
            if len(contour) >= 5:
                ellipse = cv2.fitEllipse(contour)
                _, (minor_axis, major_axis), _ = ellipse
                
                if major_axis > 0:
                    aspect_ratio = minor_axis / major_axis
                    if minor_axis / major_axis > 0.7:
                        ellipses.append(ellipse)

    return ellipses

def draw_circles_on_image(image, circles):
    import cv2
    import numpy as np

    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    for circle in circles:
        center, radius = circle
        center = (int(center[0]), int(center[1]))
        radius = int(radius)
        cv2.circle(output_image, center, radius, (0, 255, 0), 2)

    return output_image


def draw_ellipses_on_image(image, ellipses):
    import cv2
    import numpy as np

    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    for ellipse in ellipses:
        if np.isnan(ellipse[0][0]) or np.isnan(ellipse[0][1]) or np.isnan(ellipse[1][0]) or np.isnan(ellipse[1][1]) or np.isnan(ellipse[2]):
            continue
        center, axes, angle = ellipse
        center = (int(center[0]), int(center[1]))
        axes = (int(axes[0] / 2), int(axes[1] / 2))
        cv2.ellipse(output_image, center, axes, angle, 0, 360, (0, 255, 0), 2)

    return output_image

def generate_dataset_by_grid_patching(images_path, masks_path, output_path, size=(512, 512), preprocessing_pipeline = None, augment = False):
    import os
    import cv2
    import numpy as np
    from tqdm import tqdm
    from albumentations import HorizontalFlip, VerticalFlip, Rotate, Compose

    if os.path.exists(output_path):
        print(f"Output path {output_path} already exists.")
        return
    else:
        os.makedirs(os.path.join(output_path, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_path, 'masks'), exist_ok=True)

    image_files = sorted(os.listdir(images_path))
    mask_files = sorted(os.listdir(masks_path))

    for img_file, mask_file in tqdm(zip(image_files, mask_files), total=len(image_files)):
        img = cv2.imread(os.path.join(images_path, img_file), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(os.path.join(masks_path, mask_file), cv2.IMREAD_GRAYSCALE)

        if preprocessing_pipeline is not None:
            img = preprocessing_pipeline(img)

        h, w = img.shape
        for i_raw in range(20, h-20, size[0]):
            i = i_raw if i_raw + size[0] <= h - 20 else h - size[0] - 20
            for j_raw in range(20, w-20, size[1]):
                j = j_raw if j_raw + size[1] <= w - 20 else w - size[1] - 20

                img_patches = []
                mask_patches = []

                img_patches.append(img[i:i+size[0], j:j+size[1]])
                mask_patches.append(mask[i:i+size[0], j:j+size[1]])

                if augment:
                    augmentations = Compose([
                        HorizontalFlip(p=0.5),
                        VerticalFlip(p=0.5),
                        Rotate(limit=90, p=0.5)
                    ])

                    for img_patch, mask_patch in zip(img_patches.copy(), mask_patches.copy()):
                        augmented = augmentations(image=img_patch, mask=mask_patch)
                        img_patches.append(augmented['image'])
                        mask_patches.append(augmented['mask'])

                for idx, (img_patch, mask_patch) in enumerate(zip(img_patches, mask_patches)):
                    patch_name = f"{os.path.splitext(img_file)[0]}_patch_{i}_{j}_{idx}.png"
                    cv2.imwrite(os.path.join(output_path, 'images', patch_name), img_patch)
                    cv2.imwrite(os.path.join(output_path, 'masks', patch_name), mask_patch)

def load_images_as_dataset(dataset_path):
    import os
    import cv2
    import numpy as np

    images_path = os.path.join(dataset_path, 'images')
    masks_path = os.path.join(dataset_path, 'masks')

    image_files = sorted(os.listdir(images_path))
    mask_files = sorted(os.listdir(masks_path))

    images = []
    masks = []

    for img_file, mask_file in zip(image_files, mask_files):
        img = cv2.imread(os.path.join(images_path, img_file), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(os.path.join(masks_path, mask_file), cv2.IMREAD_GRAYSCALE)

        images.append(img)
        masks.append(mask)

    return np.array(images), np.array(masks)

def unet(input_shape, num_classes):
    import tensorflow as tf
    from tensorflow.keras import layers, models

    inputs = layers.Input(shape=input_shape)

    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling2D((2, 2))(c4)

    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)

    u6 = layers.UpSampling2D((2, 2))(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c6)

    u7 = layers.UpSampling2D((2, 2))(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c7)

    u8 = layers.UpSampling2D((2, 2))(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c8)

    u9 = layers.UpSampling2D((2, 2))(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c9)

    outputs = tf.keras.layers.Conv2D(
        num_classes, 
        (1, 1), 
        activation=None,
        dtype='float32'
    )(c9)
    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model

def train_unet_model(X_train, y_train, X_val, y_val, input_shape=(256, 256, 1), num_classes=1, epochs=50, batch_size=1, validation_batch_size=1, loss='binary_crossentropy'):
    import tensorflow as tf
    import numpy as np
    from tensorflow.keras import layers, models, mixed_precision
    
    tf.keras.backend.clear_session()
    mixed_precision.set_global_policy('mixed_float16')
    
    model = unet(input_shape, num_classes)
    
    if loss == 'binary_crossentropy':
        loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    else:
        loss_fn = loss

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0)
    optimizer = mixed_precision.LossScaleOptimizer(optimizer)
        
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
    
    if len(X_train.shape) == 3:
        X_train = np.expand_dims(X_train, axis=-1)
        X_val = np.expand_dims(X_val, axis=-1)
    if len(y_train.shape) == 3:
        y_train = np.expand_dims(y_train, axis=-1)
        y_val = np.expand_dims(y_val, axis=-1)
    
    def preprocess(image, mask):
        image = tf.image.resize(image, (input_shape[0], input_shape[1]))
        mask = tf.image.resize(mask, (input_shape[0], input_shape[1]), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        
        image = tf.cast(image, tf.float32) / 255.0
        mask = tf.cast(mask, tf.float32)
        mask = tf.where(mask > 1.0, mask / 255.0, mask) 
        
        return image, mask

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_dataset = val_dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(validation_batch_size).prefetch(tf.data.AUTOTUNE)
    
    history = model.fit(train_dataset, validation_data=val_dataset, epochs=epochs)
    
    return model, history