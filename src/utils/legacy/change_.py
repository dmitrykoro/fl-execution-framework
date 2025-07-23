import os
import re

def rename_all_stage_folders(root_dir):
    pattern = re.compile(r"Stage (\d+)")  # Matches "Stage 1", "Stage 2", etc.

    for dirpath, dirnames, filenames in os.walk(root_dir):
        for dirname in dirnames:
            match = pattern.fullmatch(dirname)
            if match:
                stage_number = match.group(1)
                new_dirname = f"Stage_{stage_number}"
                old_path = os.path.join(dirpath, dirname)
                new_path = os.path.join(dirpath, new_dirname)
                print(f"Renaming: {old_path} → {new_path}")
                os.rename(old_path, new_path)

# Run this
rename_all_stage_folders("lung_photos")