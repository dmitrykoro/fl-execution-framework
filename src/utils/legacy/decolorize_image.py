import os

from PIL import Image


def decolorize_image(image_path, output_path):
    """Convert a single image to grayscale."""
    with Image.open(image_path) as img:
        gray_img = img.convert("L")
        rgb_gray_img = gray_img.convert("RGB")
        rgb_gray_img.save(output_path)


def decolorize_images_in_folder(input_folder, output_folder):
    """Convert all images in folder to grayscale"""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            decolorize_image(image_path, output_path)


for i in range(12):
    for label in ("class_0", "class_1"):
        decolorize_images_in_folder(
            f"../../its_subsets/pure/client_{i}/{label}",
            f"../../its_subsets/grayscale_flip_100_100/client_{i}/{label}",
        )
