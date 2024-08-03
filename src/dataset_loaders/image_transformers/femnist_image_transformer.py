from torchvision import transforms

femnist_image_transformer = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Ensure grayscale images
    transforms.Resize((28, 28)),                  # Ensure the images are 28x28
    transforms.ToTensor(),                        # Convert images to tensors
    transforms.Normalize((0.5,), (0.5,))          # Normalize the images
])