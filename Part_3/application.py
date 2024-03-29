# write a code that takes image file path and runs the cnn model by loading saved model does the prediction and prints the predicted class

#start writing code
import torch
import torch.nn as nn
import torch.nn.functional as F
import helper_functions as helperfunctions
from torchvision import transforms
from matplotlib import image as mp_image
from PIL import Image
import model_variant2
import common_vars as GLOBAL_VARS
import numpy as np
import matplotlib.pyplot as plt

img_path = input("Enter Image Path: ")
img_path = img_path.strip("\"'")

loaded_m=helperfunctions.loadModel(modelArchitecture=model_variant2.ModelVariant2(numOfChannels=1,numOfClasses=4),savedModelName="best_so_far_after_bias1.pth",directoryPath="Part_3/saved_models")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
loaded_m.to(device)

face_img = helperfunctions.loadExternalImage(imagePath=img_path,cascadeClassifierPath="Part 1/haarcascade_frontalface_alt2.xml")

if face_img is not None:
    
    # Define transformations for the input image
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5), inplace=True)
    ])

    # Load and preprocess the image
    input_image = mp_image.imread(img_path)

    face_img=np.mean(face_img, axis=2).astype(np.uint8)

    plt.imshow(face_img)
    plt.show()
    input_tensor = transform(face_img).unsqueeze(0)
    # set the device we will be using to train the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Move the input tensor to the device
    input_tensor = input_tensor.to(device)

    # Make a prediction
    with torch.no_grad():
        output = loaded_m(input_tensor)

    # Get the predicted class
    _, predicted_class = torch.max(output.data, 1)
    predicted_class = predicted_class.item()


    data_classes =GLOBAL_VARS.DATA_CLASSES


    # Display the result
    print(f'The predicted class is as follows {data_classes[predicted_class].upper()}')
    
else:
    print("No face detected")


