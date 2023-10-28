import os
import matplotlib.pyplot as plt
import random
from matplotlib.ticker import NullLocator
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
import numpy as np


def plot_histogram(main_directory,subdirectories):
    
    counts = []
    labels = []

    for subdir in subdirectories:
        subdir_path = os.path.join(main_directory, subdir)
        if os.path.exists(subdir_path) and os.path.isdir(subdir_path):
            file_count = len([f for f in os.listdir(subdir_path) if f.endswith('.jpg') or f.endswith('.png')])  # Adjust file extensions as needed
            counts.append(file_count)
            labels.append(subdir)
    plt.figure(figsize=(8, 6))
    plt.bar(labels, counts, color='blue')
    plt.xlabel('Facial Expression')
    plt.ylabel('Number of Photos')
    plt.title('Number of Photos in Each Facial Expression Category')
    plt.show()



def display_random_images(main_directory, subdirectories, num_images_per_class=5):
    fig, axs = plt.subplots(5, 5, figsize=(8.5, 11)) 
    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    randomly_chosen_image_paths=[]

    for i in range(5):
        for j in range(5):
            #random behaviour
            class_name = random.choice(subdirectories)  

            subdir_path = os.path.join(main_directory, class_name)

            if os.path.exists(subdir_path) and os.path.isdir(subdir_path):
                image_files = [f for f in os.listdir(subdir_path) if f.lower().endswith(('.jpg', '.png'))]
                random.shuffle(image_files)

                if len(image_files) > 0:
                    image = Image.open(os.path.join(subdir_path, image_files[0]))
                    randomly_chosen_image_paths.append(os.path.join(subdir_path, image_files[0]))

                    axs[i, j].imshow(image,cmap="gray")
                    axs[i, j].axis('off')

    plt.show()
    return randomly_chosen_image_paths


def plot_histograms_from_paths(image_paths):
    fig, axs = plt.subplots(5, 5, figsize=(20, 25))  
    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    common_xlabel = "Pixel Intensity"
    common_ylabel = "Frequency"
    fig.text(0.5, 0.00001, common_xlabel, ha='center')
    fig.text(0.00001, 0.5, common_ylabel, va='center', rotation='vertical')
   

    for i in range(5):
        for j in range(5):
            idx = i * 5 + j 
            if idx < len(image_paths):
                image_path = image_paths[idx]
                image = Image.open(image_path)
                pixel_values = np.array(image).ravel()

                axs[i, j].hist(pixel_values, bins=50, color='blue', alpha=0.7)
                
                

    plt.show()
    
main_directory = 'AppliedAI/Data'
subdirectories = ['angry', 'bored', 'focused', 'neutral']
plot_histogram(main_directory, subdirectories)
chosen_images_paths=display_random_images(main_directory, subdirectories)
plot_histograms_from_paths(chosen_images_paths)

