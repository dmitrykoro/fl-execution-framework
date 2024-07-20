import json
import os
import shutil


with open(os.path.join('labels_and_metadata.json')) as f:
    metadata_list = json.load(f)

images_src = 'small_images'

label_1 = 'plant'
label_2 = 'animal'

desired_labels = 'labels'  # for coarse-grained labels, for fine-grained use 'fine_grained_labels'

previous_user_id = ''

subset_dirname = 'subsets'
client_dirname = 'client'

current_client_number = 0

label_1_src_list = []
label_2_src_list = []

MIN_PICS_PER_CLASS = 60


def copy_images(dst_dir_label_1, dst_dir_label_2):
    os.makedirs(dst_dir_label_1)
    os.makedirs(dst_dir_label_2)

    number_of_copied_images = 0

    for src_label_1, src_label_2 in zip(label_1_src_list, label_2_src_list):
        if number_of_copied_images < MIN_PICS_PER_CLASS:
            shutil.copy(src_label_1, dst_dir_label_1)
            shutil.copy(src_label_2, dst_dir_label_2)
            number_of_copied_images += 1
        else:
            return


for metadata_entry in metadata_list:
    picture_labels = metadata_entry[desired_labels]

    if label_1 in picture_labels and label_2 in picture_labels:  # if picture contains both classes we do not use it
        continue

    if label_1 in picture_labels or label_2 in picture_labels:
        current_user_id = metadata_entry['user_id']

        if current_user_id != previous_user_id:

            if len(label_1_src_list) >= MIN_PICS_PER_CLASS and len(label_2_src_list) >= MIN_PICS_PER_CLASS:
                copy_images(
                    f'{subset_dirname}/{client_dirname}_{current_client_number}/{label_1}',
                    f'{subset_dirname}/{client_dirname}_{current_client_number}/{label_2}'
                )

                current_client_number += 1

            label_1_src_list = []
            label_2_src_list = []

        if label_1 in picture_labels:
            label_1_src_list.append(f"{images_src}/{metadata_entry['image_id']}.jpg")
        if label_2 in picture_labels:
            label_2_src_list.append(f"{images_src}/{metadata_entry['image_id']}.jpg")

        previous_user_id = metadata_entry['user_id']
