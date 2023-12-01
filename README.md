# A.I.ducation Analytics

## Team Members
- Ronak 40221814
- Vijendra 40221273
- Aayush 40272388

## Project Aim
The aim of this project is to develop an efficient and accurate system for classifying facial expressions from given photos using Convolutional Neural Networks (CNN).

### Important
- Make sure the root directory is set as `APPLIEDAI` to make sure all the relative paths are working corrently.

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

## Part-3: Finding Bias in the model and Mitigating for it

This part of the project focuses on identifying any biases in the trained model and implementing strategies to mitigate these biases.

### Prerequisites
- Dataset related folders such as `DataKF`, `dataset_inshape`, `updated_datasetinshape`, `UpdatedDataKF` and `saved_models` with four subfolders for facial expression classes.

### Improvements over Step-2
- `generate_confusion_matrix_on_test_set.py` file is used to run an existing model over test dataset to generate confusion matrix and check the performance of the model
- `helper_function.py` file is modified to halt the training of the model if the accuracy keeps decreasing and saved the best model with the highest accuracy over all the successful execution
- `train.py` file is modified to use the same split of dataset to train and validate different models to maintain consistency in the results

### Model Training
- Navigate to `Part_3`.
- Run `train.py`.
- The trained model will be saved in `saved_models` named `best_model.pth`.

### Running the Application
- Run `application.py`.
- Provide an image path for facial expression identification.
- Use external images for testing.

## Conclusion
Follow these instructions for successful project execution. For support or contributions, open an issue or submit a pull request.
