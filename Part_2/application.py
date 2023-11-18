import torch
import torch.nn as nn
import torch.nn.functional as F
import helper_functions as helperfunctions
from cnn_building_blocks import conv_block
from torchvision import transforms
from matplotlib import image as mp_image
from PIL import Image
import model_variant2
import model_variant1
import model_variant3
import common_vars as GLOBAL_VARS 
import numpy as np
import matplotlib.pyplot as plt

img_path = input("Enter Image Path: ")
img_path = img_path.strip("\"'")

#old code
#loaded_m=helperfunctions.loadModel(filePath="AppliedAI/Part_2/saved_model",modelArchitecture=model_variant2.ModelVariant2(numOfChannels=1,numOfClasses=4),savedModelName="mymodel.pth")

loaded_m = helperfunctions.loadModel("AppliedAI/Part_2/saved_model/mymodel.pth")

face_img = helperfunctions.loadExternalImage(imagePath=img_path,cascadeClassifierPath="AppliedAI/Part 1/haarcascade_frontalface_alt2.xml")
if face_img is not None:
    
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_tensor = input_tensor.to(device)

    # Make a prediction
    with torch.no_grad():
        output = loaded_m(input_tensor)

    _, predicted_class = torch.max(output.data, 1)
    predicted_class = predicted_class.item()

    data_classes =GLOBAL_VARS.DATA_CLASSES

    print(f'The predicted class is as follows {data_classes[predicted_class].upper()}')
    
else:
    print("No face detected")


