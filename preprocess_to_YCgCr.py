import os
import cv2
import numpy as np

# Dataset directories
DATASET_DIR = r'D:\coding\proper_working_ycgcr\augmented_dataset'  # Root dataset path
SAVE_DIR = 'data'  # Directory to save processed data

# Image size
IMG_SIZE = 100

def convert_to_YCgCr(image_path):
    """
    Convert an image to YCgCr color space and resize it.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found at {image_path}")

    # Convert to YCgCr
    img_YCgCr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)  # OpenCV's YCrCb is similar to YCgCr

    # Resize image
    img_resized = cv2.resize(img_YCgCr, (IMG_SIZE, IMG_SIZE))

    # Normalize pixel values to [0, 1]
    img_normalized = img_resized / 255.0

    return img_normalized

def load_and_process_images(directory):
    """
    Load and preprocess images from the dataset directory.
    Returns: numpy arrays for images and labels.
    """
    images = []
    labels = []

    for label, category in enumerate(['Unripe', 'Ripe']):  # Unripe = 0, Ripe = 1
        category_path = os.path.join(directory, category)

        if os.path.exists(category_path):
            print(f"Processing '{category}' images...")
            for image_name in os.listdir(category_path):
                image_path = os.path.join(category_path, image_name)

                try:
                    img = convert_to_YCgCr(image_path)
                    images.append(img)
                    labels.append(label)
                except ValueError as e:
                    print(e)
        else:
            print(f"Directory not found: {category_path}")

    return np.array(images), np.array(labels)

if __name__ == "__main__":
    # Process images
    print("Processing dataset...")
    X, y = load_and_process_images(DATASET_DIR)

    # Shuffle dataset
    print("Shuffling dataset...")
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X, y = X[indices], y[indices]

    # Split into training and testing sets
    split_index = int(0.8 * len(X))  # 80% for training, 20% for testing
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # Save processed data
    os.makedirs(SAVE_DIR, exist_ok=True)
    np.save(os.path.join(SAVE_DIR, 'X_train.npy'), X_train)
    np.save(os.path.join(SAVE_DIR, 'X_test.npy'), X_test)
    np.save(os.path.join(SAVE_DIR, 'y_train.npy'), y_train)
    np.save(os.path.join(SAVE_DIR, 'y_test.npy'), y_test)

    print(f"Processed data saved to '{SAVE_DIR}'.")
    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}, y_test: {y_test.shape}")
