import os

def rename_files(directory_path):
    count = 0

    for foldername, subfolders, filenames in os.walk(directory_path):
        for file in filenames:
            old_path = os.path.join(foldername, file)
            new_name = str(count) + ".jpg"
            new_path = os.path.join(foldername, new_name)

            os.rename(old_path, new_path)
            count += 1

# rename_files("../images/...")