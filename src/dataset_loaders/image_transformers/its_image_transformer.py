from torchvision import transforms

its_image_transformer = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
