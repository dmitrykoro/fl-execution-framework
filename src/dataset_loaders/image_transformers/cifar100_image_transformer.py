from torchvision import transforms

# CIFAR-100 per-channel mean and std computed over the full training set
cifar100_image_transformer = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.5071, 0.4867, 0.4408),
        std=(0.2675, 0.2565, 0.2761)
    )
])
