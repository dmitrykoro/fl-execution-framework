from torchvision import transforms

its_image_transformer = transforms.Compose([
    transforms.Resize((224, 224)),  # 224x224
    transforms.ToTensor()  # to tensor
])
