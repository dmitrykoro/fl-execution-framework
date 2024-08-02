import os
import random

client_directory = os.path.join('../../femnist_subsets/flipping_100_100/client_11')

label_dirs = os.listdir(client_directory)

for label_dir in label_dirs:
    if label_dir.startswith('.'):  # skip .DS_store
        continue

    old_dir = os.path.join(client_directory, label_dir)
    new_dir = os.path.join(client_directory, label_dir + '_old')

    os.rename(old_dir, new_dir)


label_dirs = os.listdir(client_directory)

available_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

for label_dir in label_dirs:
    if label_dir.startswith('.'):  # skip .DS_store
        continue

    old_dir = os.path.join(client_directory, label_dir)

    while True:
        new_label = random.choice(available_labels)

        if new_label != int(label_dir.split('_')[0]):
            break

    new_dir = os.path.join(client_directory, f'{new_label}')

    os.rename(old_dir, new_dir)

    available_labels.remove(new_label)


