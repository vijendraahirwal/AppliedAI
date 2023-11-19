
from sklearn.model_selection import train_test_split
import shutil
import time
import os
import subprocess

print("Setting up appropriate directories...")
original_dataset = "AppliedAI/Data"
split_original_dataset = "AppliedAI/Part_2/dataset_inshape"
rand = int(time.time() % 1000)
#if not make it 
if not os.path.exists(split_original_dataset):
    os.makedirs(split_original_dataset)
data_catgories = os.listdir(original_dataset)

print("Splitting dataset into train and test...")

# Loop each class lbl
for l in data_catgories:
    lbl = os.path.join(original_dataset, l)
    train_data_directory = os.path.join(split_original_dataset, "train", l)
    test_data_directory = os.path.join(split_original_dataset, "test", l)
    os.makedirs(train_data_directory, exist_ok=True)
    os.makedirs(test_data_directory, exist_ok=True)

    if os.path.isdir(lbl):
        files = os.listdir(lbl)
        train_files, test_files = train_test_split(files, test_size=0.15, random_state=rand)
        for file in train_files:
            src = os.path.join(lbl, file)
            dst = os.path.join(train_data_directory, file)
            shutil.copy(src, dst)
        for file in test_files:
            src = os.path.join(lbl, file)
            dst = os.path.join(test_data_directory, file)
            shutil.copy(src, dst)

directory = split_original_dataset

for root, _, files in os.walk(directory):
        for filename in files:
            if not (filename.lower().endswith(('.jpg', '.jpeg')) or filename == ".DS_Store"):
                file_path = os.path.join(root, filename)
                os.remove(file_path)
                print(f"Removed: {file_path}")
                
command = 'find . -name ".DS_Store" -delete'

try:
    result = subprocess.run(command, shell=True, cwd=directory, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode == 0:
        print("Command executed successfully:")
        print(result.stdout)
    else:
        print("Command failed with error:")
        print(result.stderr)
except:
    print("no success")

print("Splitting dataset into train and test completed successfully...")
