import os
import numpy as np
import medmnist
from medmnist import INFO, Evaluator
import torchvision.transforms as transforms
from PIL import Image
from sklearn.model_selection import train_test_split

# Set the dataset you want to use from MedMNIST
DATA_FLAG = 'pneumoniamnist'  # Example dataset
BATCH_SIZE = 32
NUM_CLIENTS = 10  # Adjust based on your needs
NUM_LABELS = 2  # Adjust based on dataset

# Load dataset info and dataset
info = INFO[DATA_FLAG]
DataClass = getattr(medmnist, info['python_class'])
data_transform = transforms.Compose([transforms.ToPILImage()])

# Load training data
train_dataset = DataClass(split='train', transform=data_transform, download=True)

# Prepare client data distribution (e.g., randomly assign data to clients)
X, y = train_dataset.imgs, train_dataset.labels
client_data = []

# Split the data for the number of clients
for client_id in range(NUM_CLIENTS):
    X_client, _, y_client, _ = train_test_split(X, y, test_size=0.8)  # Adjust test_size as needed
    client_data.append((X_client, y_client))

# Create folder structure and save images
base_dir = 'medmnist_clients'
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

for client_id, (X_client, y_client) in enumerate(client_data, start=1):
    client_folder = os.path.join(base_dir, f'client_{client_id}')
    os.makedirs(client_folder, exist_ok=True)

    for label in np.unique(y_client):
        label_folder = os.path.join(client_folder, f'label_{label}')
        os.makedirs(label_folder, exist_ok=True)

        for idx, (img, lbl) in enumerate(zip(X_client, y_client)):
            if lbl == label:
                img = Image.fromarray(img.squeeze(), mode='L')  # For grayscale images
                img.save(os.path.join(label_folder, f'img_{idx}.png'))  # Change to .jpeg if needed
