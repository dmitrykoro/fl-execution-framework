import json
import os
import shutil


class SubsetCreator:
    MIN_PICS_PER_CLASS = 200

    def __init__(self):
        with open(os.path.join('labels_and_metadata.json')) as f:
            self.metadata_list = json.load(f)

        self.images_src = 'small_images'

        self.label_1 = 'equipment'
        self.label_2 = 'outdoor'

        self.desired_labels = 'labels'  # for coarse-grained labels, for fine-grained use 'fine_grained_labels'

        self.subset_dirname = 'subsets'
        self.client_dirname = 'client'

        self.label_1_src_list = []
        self.label_2_src_list = []

        self.create_subset()

    def copy_images(self, dst_dir_label_1, dst_dir_label_2):
        """
        Copy images per one user
        :param dst_dir_label_1: destination directory for the label 1
        :param dst_dir_label_2: destination directory for the label 2
        """
        os.makedirs(dst_dir_label_1)
        os.makedirs(dst_dir_label_2)

        number_of_copied_images = 0

        for src_label_1, src_label_2 in zip(self.label_1_src_list, self.label_2_src_list):
            if number_of_copied_images < SubsetCreator.MIN_PICS_PER_CLASS:
                shutil.copy(src_label_1, dst_dir_label_1)
                shutil.copy(src_label_2, dst_dir_label_2)
                number_of_copied_images += 1
            else:
                return

    def create_subset(self):
        """
        Create subset of Flair dataset.
        """
        previous_user_id = ''
        current_client_number = 0

        for metadata_entry in self.metadata_list:
            picture_labels = metadata_entry[self.desired_labels]

            # if picture contains both classes we do not use it
            if self.label_1 in picture_labels and self.label_2 in picture_labels:
                continue

            if self.label_1 in picture_labels or self.label_2 in picture_labels:
                current_user_id = metadata_entry['user_id']

                if current_user_id != previous_user_id:
                    if (
                            len(self.label_1_src_list) >= SubsetCreator.MIN_PICS_PER_CLASS
                            and len(self.label_2_src_list) >= SubsetCreator.MIN_PICS_PER_CLASS
                    ):
                        self.copy_images(
                            f'{self.subset_dirname}/{self.client_dirname}_{current_client_number}/{self.label_1}',
                            f'{self.subset_dirname}/{self.client_dirname}_{current_client_number}/{self.label_2}'
                        )
                        current_client_number += 1

                    self.label_1_src_list = []
                    self.label_2_src_list = []

                if self.label_1 in picture_labels:
                    self.label_1_src_list.append(f"{self.images_src}/{metadata_entry['image_id']}.jpg")
                if self.label_2 in picture_labels:
                    self.label_2_src_list.append(f"{self.images_src}/{metadata_entry['image_id']}.jpg")

                previous_user_id = metadata_entry['user_id']


if __name__ == '__main__':
    subset_creator = SubsetCreator()
