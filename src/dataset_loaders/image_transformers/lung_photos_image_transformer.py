from torchvision import transforms

lung_cancer_image_transformer = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),  # Ensure grayscale
        transforms.Resize((224, 224)),  # Resize to match model input
        transforms.ToTensor(),  # Convert to [0,1] tensor
        transforms.Normalize((0.5,), (0.5,)),  # Normalize to [-1,1]
    ]
)
