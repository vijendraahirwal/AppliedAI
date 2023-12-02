import os

#source directory
source_dir = r"Data"


def rename_files_in_directory(directory_path):
    directory_name = os.path.basename(directory_path)
    files = os.listdir(directory_path)
    sequence_number = 1

    for file in files:
        # Building file name
        new_name = f"{directory_name}_{sequence_number:03}"
        file_name, file_extension = os.path.splitext(file)
        original_file_path = os.path.join(directory_path, file)
        new_file_path = os.path.join(directory_path, f"{new_name}{file_extension}")
        #print(f"Renaming {original_file_path} to {new_file_path}")
        os.rename(original_file_path, new_file_path)
        sequence_number += 1


for root, dirs, files in os.walk(source_dir):
    for directory in dirs:
        #Building full path to the subdirectory
        directory_path = os.path.join(root, directory)
        #print(directory_path)
        rename_files_in_directory(directory_path)

print("File renaming completed.")
