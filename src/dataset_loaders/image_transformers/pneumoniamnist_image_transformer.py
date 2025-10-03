from torchvision import transforms

pneumoniamnist_image_transformer = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),  # grayscale
        transforms.Resize((28, 28)),  # 28x28 pixels
        transforms.ToTensor(),  # to tensors
        transforms.Normalize((0.5,), (0.5,)),  # normalize
    ]
)
