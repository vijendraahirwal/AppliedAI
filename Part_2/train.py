# set the matplotlib backend so figures can be saved in the background
import matplotlib

import helper_functions as helperfunctions

#from AppliedAI.Part_2.model_variant1 import CustomCNN
import math
from sklearn.metrics import classification_report
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import ToTensor
from torchvision.datasets import ImageFolder
from torch.optim import Adam
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch
import time
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms, models

import numpy as np
import sklearn
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import common_vars as GLOBAL_VARS


# set the device we will be using to train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data, test_data = helperfunctions.loadAndTransformDataSet()

#70% for training, 15% for validation
numTrainSamples = int(len(train_data) * GLOBAL_VARS.TRAIN_SPLIT)
numValSamples = math.ceil(len(train_data) * GLOBAL_VARS.VALIDATION_SPLIT)

(final_train_data, final_val_data) = random_split(train_data,
	[numTrainSamples, numValSamples],
	generator=torch.Generator().manual_seed(23))

print("Loading the data loader...")
trainDataLoader, valDataLoader, testDataLoader = helperfunctions.getTheDataLoader(final_train_data,final_val_data,test_data)



#helperfunctions.runCV(train_loader=trainDataLoader,valid_loader=valDataLoader,dataset_classes=['angry', 'bored', 'focused', 'neutral'],device=device)

model = helperfunctions.getModel(model_name="variant2",numberOfInputChannels=1,numberOfClasses=4)



# # model = models.alexnet(weights=models.alexnet.DEFAULT) # load pretrained model
# # model.fc = nn.Linear(512, 4)
# print("Getting the optimizer and loss function...")
optmizer = helperfunctions.getOptimizer(model=model,optimizer_name= "adam",learning_rate=GLOBAL_VARS.INIT_LR)
lossFn = helperfunctions.getLossFunction(loss_function_name="cross_entropy")

# # this function returs train_loss_history, train_accuracy_history, val_loss_history, val_accuracy_history
# print("Training the model and Evaluating the model on validation data ...")
train_loss_history, train_accuracy_history, val_loss_history, val_accuracy_history = helperfunctions.trainLoopOfTheNetwork(model=model,device=device,
                                                                                                                           trainDataLoader=trainDataLoader,
                                                                                                                   valDataLoader=valDataLoader,
                                                                                                                   lossFunction=lossFn,
                                                                                                                   optimizer=optmizer,
                                                                                                                   lenTrainingData=len(final_train_data),
                                                                                                                   lenValData=len(final_val_data),
                                                                                                                   earlystopping=True,
                                                                                                                   )

 


# # Evaluate the network on the test set
# print("[INFO] evaluating network...")

# # Initialize lists to store true labels and predicted labels
true_labels = []
predicted_labels = []

true_labels,predicted_labels = helperfunctions.evaluateModelWithTestData(model=model,testDataLoader=testDataLoader,device=device)

# Calculate scores
accuracy = helperfunctions.getScores(trueLabels=true_labels,predictedLabels=predicted_labels,typeOfScore="accuracy")
precision_macro = helperfunctions.getScores(trueLabels=true_labels,predictedLabels=predicted_labels,typeOfScore="precision",averageType="macro")
recall_macro = helperfunctions.getScores(trueLabels=true_labels,predictedLabels=predicted_labels,typeOfScore="recall",averageType="macro")
f1_macro = helperfunctions.getScores(trueLabels=true_labels,predictedLabels=predicted_labels,typeOfScore="f1",averageType="macro") 
precision_micro = helperfunctions.getScores(trueLabels=true_labels,predictedLabels=predicted_labels,typeOfScore="precision",averageType="micro")
recall_micro = helperfunctions.getScores(trueLabels=true_labels,predictedLabels=predicted_labels,typeOfScore="recall",averageType="micro")
f1_micro = helperfunctions.getScores(trueLabels=true_labels,predictedLabels=predicted_labels,typeOfScore="f1",averageType="micro")

# Print metrics in a table
print("\nMetrics Summary:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision (macro): {precision_macro:.2f}")
print(f"Recall (macro): {recall_macro:.2f}")
print(f"F1-score (macro): {f1_macro:.2f}")
print(f"Precision (micro): {precision_micro:.2f}")
print(f"Recall (micro): {recall_micro:.2f}")
print(f"F1-score (micro): {f1_micro:.2f}")



print('Accuracy of the network on the test images: %.2f%%' % (100 * accuracy))


helperfunctions.generateConfusionMatrix(trueLabels=true_labels,predictedLabels=predicted_labels,classes=GLOBAL_VARS.DATA_CLASSES)



# Printing classification report
helperfunctions.printClassificationReport(trueLabels=true_labels,predictedLabels=predicted_labels,classes=GLOBAL_VARS.DATA_CLASSES)

# print(train_loss_history)
# print(train_accuracy_history)
# print(val_loss_history)
# print(val_accuracy_history)

helperfunctions.plotTrainingHistory(trainLossHistory=train_loss_history,trainAccuracyHistory= train_accuracy_history, valLossHistory=val_loss_history, valAccuracyHistory=val_accuracy_history)
 
helperfunctions.saveModel(model_state_dict=model.state_dict(),directoryPath="AppliedAI/Part_2/saved_model",modelName="mymodel.pth")

 
 
 



  
  
  
  
  
  
  




