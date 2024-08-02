from torchvision import transforms

femnist_image_transformer = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])