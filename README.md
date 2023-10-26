

# Facial Expression Detection

## Team Members
- Ronak 
- Vijendra
- Aayush

## Project Aim

## Explaination of the files
1. **data_visualisation.py**: In this file, there are three functions 
   The first function, **plot_histogram**, generates a histogram to visualize the distribution of images in different expression categories. The second function, **display_random_images**,
   displays a grid of randomly chosen images from four categories. The third function, **plot_histograms_from_paths**, plots histograms of pixel intensities for the selected images.

2. **convertToGrayscale.py**: This code automates the process of converting color images to grayscale and saves the processed images.
3. **resizeImages.py**: This code takes grayscale images from a source folder and resizes them to 48x48 pixels.
4. **experiment.py**: This file contains code that we did while working on the project doing different experiments. So disregard the code.
5. **preprocessing.py**: This code defines a function, **preprocess_images_in_directories**, which processes and enhances images in specified subdirectories within a base directory. The image preprocessing involves grayscale conversion, blurring, and unsharp masking to
   sharpen the images. The code overwrites the original images with the processed versions.

## Steps to Run the Project
Follow these steps to run the project:

1. Run the `convertToGrayscale.py` file.
2. Run the `resizeImages.py` file.
3. Run the `preprocessing.py` file.
4. Run the `data_visualisation.py` file.
