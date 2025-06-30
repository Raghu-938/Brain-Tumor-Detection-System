import numpy as np
import cv2
import os

def normalize_image(img):
    return img / 255.0

def fuzzy_entropy(patch):
    epsilon = 1e-10
    μ = patch.flatten()
    return -np.mean(μ * np.log(μ + epsilon) + (1 - μ) * np.log(1 - μ + epsilon))

def fuzzy_inclusion(patch1, patch2):
    return np.mean(np.minimum(patch1, patch2))

def fuzzy_preprocess(img, patch_size=16, entropy_thresh=0.2, inclusion_thresh=0.3):
    img = normalize_image(img)
    h, w = img.shape
    new_img = np.zeros_like(img)

    for i in range(0, h - patch_size + 1, patch_size):
        for j in range(0, w - patch_size + 1, patch_size):
            patch = img[i:i+patch_size, j:j+patch_size]
            entropy = fuzzy_entropy(patch)

            if j + patch_size < w:
                neighbor = img[i:i+patch_size, j+patch_size:j+2*patch_size]
                inclusion = fuzzy_inclusion(patch, neighbor)
            else:
                inclusion = 1

            if entropy >= entropy_thresh and inclusion >= inclusion_thresh:
                new_img[i:i+patch_size, j:j+patch_size] = patch

    return new_img

if __name__ == "__main__":
    input_root = 'Train_Dataset'
    output_folder = 'processed_datasett'
    os.makedirs(output_folder, exist_ok=True)

    for class_folder in os.listdir(input_root):
        class_path = os.path.join(input_root, class_folder)

        for filename in os.listdir(class_path):
            if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                input_path = os.path.join(class_path, filename)
                img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (224, 224))

                processed_img = fuzzy_preprocess(img)

                new_filename = f"{class_folder}_{filename}"
                output_path = os.path.join(output_folder, new_filename)

                cv2.imwrite(output_path, (processed_img * 255).astype(np.uint8))