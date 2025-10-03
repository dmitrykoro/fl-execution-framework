import os
import numpy as np
from PIL import Image


def add_noise_to_image(image_path, output_path, noise_level=0.1):
    """Add noise to a single image."""
    with Image.open(image_path) as img:
        img_array = np.array(img)
        noise = np.random.randn(*img_array.shape) * 255 * noise_level
        noisy_img_array = img_array + noise
        noisy_img_array = np.clip(noisy_img_array, 0, 255).astype(np.uint8)
        noisy_img = Image.fromarray(noisy_img_array)
        noisy_img.save(output_path)


def add_noise_to_images_in_folder(input_folder, output_folder, noise_level=0.1):
    """Add noise to all images in the specified folder."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            add_noise_to_image(image_path, output_path, noise_level)


input_folder = "../../femnist_subsets/pure/client_10/1"
output_folder = "../../femnist_subsets/noise_noise/client_10/1"
noise_level = 0.5

add_noise_to_images_in_folder(input_folder, output_folder, noise_level)
