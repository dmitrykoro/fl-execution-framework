import cv2
import os

def crop_center(image_path, output_path, crop_size):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    start_x = w // 2 - crop_size // 2
    start_y = h // 2 - crop_size // 2
    cropped_img = img[start_y:start_y+crop_size, start_x:start_x+crop_size]
    cv2.imwrite(output_path, cropped_img)

def process_images(input_root, output_root, crop_size):
    for root, dirs, files in os.walk(input_root):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, input_root)
                output_dir = os.path.join(output_root, relative_path)
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, file)
                crop_center(input_path, output_path, crop_size)

# Example usage
input_root = 'lung_photos'
output_root = 'cropped_lung_photos'
crop_size = 1300

process_images(input_root, output_root, crop_size)

print("Cropping completed.")