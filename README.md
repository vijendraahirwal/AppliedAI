

# A.I.ducation Analytics

## Team Members
- Ronak 40221814
- Vijendra 40221273
- Aayush 40272388

## Project Aim
The aim of this project is to develop an efficient and accurate system for classifying facial expressions from given photos using Convolutional Neural Networks (CNN). This project can become a valuable tool for applications in fields such as human-computer interaction, psychology, and market research.

## Explaination of the files
1. **data_visualisation.py**: In this file, there are three functions 
   The first function, **plot_histogram**, generates a histogram to visualize the distribution of images in different expression categories. The second function, **display_random_images**,
   displays a grid of randomly chosen images from four categories. The third function, **plot_histograms_from_paths**, plots histograms of pixel intensities for the selected images.

2. **convert_greyscale_and_resize.py**: This code automates the process of converting color images to grayscale and also resizing them to 48x48 pixels.
3. **experiment.py**: This file contains code that we did while working on the project doing different experiments. So disregard the code.
4. **image_sharpening.py**: This code defines a function, **preprocess_images_in_directories**, which processes and enhances images in specified subdirectories within a base directory. The image preprocessing involves grayscale conversion, blurring, and unsharp masking to sharpen the images. The code overwrites the original images with the processed versions.
5. **image_labeling.py**: This python script iterates over all the images inside the directories and labels them in this format <class_label_name>_XXXX.jpg, where XXX is the sequence number.

## Steps to Run the Project
Follow these steps to run the project: 

Before running any python script please make sure you have folder named Data for the images. 

1. Run the `filtering_locating_faces.py` file.
2. Run the `convert_greyscale_and_resize.py` file.
3. Run the `image_sharpening.py` file.
4. Run the `image_labeling.py` file.