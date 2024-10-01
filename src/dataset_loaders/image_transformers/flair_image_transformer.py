from torchvision import transforms

flair_image_transformer = transforms.Compose([
    transforms.Resize((256, 256)),  # 256x256
    transforms.ToTensor()  # to tensor
])
