"""Convert JSON FEMNIST data to PNG images for IID."""

import os
import json
import numpy as np
from PIL import Image

data_path = "train"
output_path = "organized_data"

if not os.path.exists(output_path):
    os.makedirs(output_path)

num_users = 12

i = 0

user_files = os.listdir("train")

name_list = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
]

for current_user_file in user_files:
    all_user_data = json.load(open(os.path.join(data_path, current_user_file)))

    current_user = all_user_data["user_data"]
    num_samples = all_user_data["num_samples"][0]

    for user_number, user in current_user.items():
        user_id = f"client_{user_number}"
        user_path = os.path.join(output_path, user_id)
        os.makedirs(user_path, exist_ok=True)

        user_data = user["x"]
        user_labels = user["y"]

        size = (num_samples, 28, 28, 1)

        image_numpy = np.array(user_data, dtype=np.float32).reshape(size)
        label_numpy = np.array(user_labels, dtype=np.int32)

        for i in range(image_numpy.shape[0]):
            img = image_numpy[i] * 255
            im = Image.fromarray(np.squeeze(img))
            im = im.convert("L")

            os.makedirs(
                os.path.join(user_path, f"{name_list[label_numpy[i]]}"), exist_ok=True
            )

            label_folder = os.path.join(user_path, f"{name_list[label_numpy[i]]}")

            img_name = (
                str(label_numpy[i])
                + "_"
                + name_list[label_numpy[i]]
                + "_"
                + str(i)
                + ".png"
            )

            image_path = os.path.join(label_folder, img_name)
            im.save(image_path)

print("Data organized successfully.")
