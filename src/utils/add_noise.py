import os
import numpy as np
from PIL import Image


def add_noise_to_image(image_path, output_path, noise_level=0.1):
    # Open an image file
    with Image.open(image_path) as img:
        # Convert image to numpy array
        img_array = np.array(img)

        # Generate random noise
        noise = np.random.randn(*img_array.shape) * 255 * noise_level

        # Add noise to the image
        noisy_img_array = img_array + noise

        # Clip the values to be in the valid range [0, 255] and convert to uint8
        noisy_img_array = np.clip(noisy_img_array, 0, 255).astype(np.uint8)

        # Convert numpy array back to image
        noisy_img = Image.fromarray(noisy_img_array)

        # Save the noisy image
        noisy_img.save(output_path)


def add_noise_to_images_in_folder(input_folder, output_folder, noise_level=0.1):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            add_noise_to_image(image_path, output_path, noise_level)


# Example usage
input_folder = '../../flair_subsets/animal_liquid_pure/client_10/liquid'
output_folder = '../../flair_subsets/animal_liquid_noise_noise/client_10/liquid'
noise_level = 0.5  # Adjust noise level as needed

add_noise_to_images_in_folder(input_folder, output_folder, noise_level)
