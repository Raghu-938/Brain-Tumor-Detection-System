import numpy as np
import cv2
import os
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array
from tqdm import tqdm  # for progress bar

# Load DenseNet201 base model
base_model = DenseNet201(weights='imagenet', include_top=False, pooling='avg')

def extract_features_from_image(img, target_size=(224, 224)):
    img_resized = cv2.resize(img, target_size)
    if len(img_resized.shape) == 2 or img_resized.shape[2] == 1:
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
    img_array = img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = base_model.predict(img_array, verbose=0)
    return features.flatten()

# Paths

if __name__ == "__main__":
    input_dir = 'processed_datasett'
    features_list = []
    labels_list = []

    # Process each image
    for filename in tqdm(os.listdir(input_dir)):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(input_dir, filename)
            img = cv2.imread(img_path)

            features = extract_features_from_image(img)
            features_list.append(features)

            # Assign label from filename
            if filename.startswith('yes'):
                labels_list.append(1)
            elif filename.startswith('no'):
                labels_list.append(0)

    # Convert to arrays
    X = np.array(features_list)
    y = np.array(labels_list)

    # Save for future use
    np.save('models/features.npy', X)
    np.save('models/labels.npy', y)

    print(" Dataset creation complete.")
    print("Feature shape:", X.shape)
    print("Labels shape:", y.shape)

    input_root = 'Train_Dataset'
    data_features_list = []
    data_labels_list = []

    for class_folder in os.listdir(input_root):
        class_path = os.path.join(input_root, class_folder)
        if not os.path.isdir(class_path):
            continue  # skip non-directory files

        label = 1 if class_folder.lower().startswith('yes') else 0

        for filename in tqdm(os.listdir(class_path), desc=f"Processing {class_folder}"):
            if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                img_path = os.path.join(class_path, filename)
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Could not read image: {img_path}")
                    continue

                data_features = extract_features_from_image(img)
                data_features_list.append(data_features)
                data_labels_list.append(label)

    # Convert to arrays
    data_X = np.array(data_features_list)
    data_y = np.array(data_labels_list)
    np.save('data_features.npy', data_X)
    np.save('data_labels.npy', data_y)

    print("Feature matrix shape:", data_X.shape)
    print("Labels shape:", data_y.shape)
