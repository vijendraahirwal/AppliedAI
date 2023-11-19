# A.I.ducation Analytics

## Team Members
- Ronak 40221814
- Vijendra 40221273
- Aayush 40272388

## Project Aim
The aim of this project is to develop an efficient and accurate system for classifying facial expressions from given photos using Convolutional Neural Networks (CNN).

## Overview of Project Files

### Core Scripts
- **data_visualisation.py**: 
  - `plot_histogram`: Visualizes image distribution across expression categories.
  - `display_random_images`: Shows random images grid from four categories.
  - `plot_histograms_from_paths`: Plots histograms of pixel intensities.

- **convert_greyscale_and_resize.py**: 
  - Automates conversion of images to grayscale and resizes them to 48x48 pixels.

- **image_sharpening.py**: 
  - `preprocess_images_in_directories`: Enhances images with grayscale conversion, blurring, and unsharp masking.

- **image_labeling.py**: 
  - Labels images in the format `<class_label_name>_XXXX.jpg`.

- **filtering_locating_faces.py**: 
  - Detects and extracts faces from images.

- **experiment.py**: 
  - Contains experimental code used during project development.

### Model Training and Application
- **train.py (Part-2 folder)**: 
  - Trains the model using preprocessed and labeled images.

- **application.py (Part-2 folder)**: 
  - Loads the trained model for facial expression prediction.

## Steps to Run the Project
Ensure the `Data` folder is set up correctly and all paths are relative to the **Applied AI** main directory.

1. Run `filtering_locating_faces.py`.
2. Run `convert_greyscale_and_resize.py`.
3. Run `image_sharpening.py`.
4. Run `image_labeling.py`.

## Part-2: Model Creation and Testing

### Prerequisites
- A folder named `Data` with four subfolders for facial expression classes.

### Data Preparation
- Run `data_prep.py`.
- The dataset will be in `dataset_inshape` under the `Part_2` directory.

### Model Training
- Navigate to `Part-2`.
- Run `train.py` and choose option `2` for `Model_variant2`.
- The trained model is saved in `Part_2/saved_model`.

### Running the Application
- Run `application.py`.
- Provide an image path for facial expression identification.
- Use external images for testing.

## Conclusion
Follow these instructions for successful project execution. For support or contributions, open an issue or submit a pull request.
