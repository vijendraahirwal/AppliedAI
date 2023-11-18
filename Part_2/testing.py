# set the matplotlib backend so figures can be saved in the background
import matplotlib

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
import cv2
import numpy as np
import sklearn
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import os

# define training hyperparameters
INIT_LR = 0.002
BATCH_SIZE = 32
EPOCHS = 30
# define the train and val splits
TRAIN_SPLIT = 0.85
VALIDATION_SPLIT = 1 - TRAIN_SPLIT

TRAINING_DATA_DIR ="AppliedAI/Part_2/dataset_inshape/train"
TESTING_DATA_DIR = "AppliedAI/Part_2/dataset_inshape/test"

DATA_CLASSES = ('Angry', 'Focused', 'Bored', 'Neutral')




def loadAndTransformDataSet():

    training_data_dir = TRAINING_DATA_DIR
    test_data_dir = TESTING_DATA_DIR

    resizing = transforms.Resize(size=(48, 48))
    horizontal_flip = transforms.RandomHorizontalFlip(p=0.3)
    grayscale = transforms.Grayscale(num_output_channels=1)
    #normalise the image
    normalize = transforms.Normalize((0.5), (0.5), inplace=True)

    train_data_transformations = transforms.Compose([grayscale,resizing, horizontal_flip, ToTensor(),normalize])
    test_data_transformations = transforms.Compose([grayscale,resizing,ToTensor(),normalize])
    print("Loading the data set...")
    train_data = ImageFolder(root=training_data_dir, transform=train_data_transformations)
    test_data  = ImageFolder(root=test_data_dir, transform=test_data_transformations)

    return train_data, test_data


def getTheDataLoader(train_data,val_data,test_data):

    trainDataLoader = DataLoader(train_data, shuffle=True,batch_size=BATCH_SIZE)
    valDataLoader = DataLoader(val_data, batch_size=BATCH_SIZE)
    testDataLoader = DataLoader(test_data, batch_size=BATCH_SIZE)

    return trainDataLoader, valDataLoader, testDataLoader

def getModel(model_name,numberOfInputChannels, numberOfClasses):

    if(model_name == "variant1"):
        return ModelVariant1(numberOfInputChannels, numberOfClasses)
    elif(model_name == "variant2"):
        return ModelVariant2(numberOfInputChannels, numberOfClasses)
    elif(model_name == "variant3"):
        return ModelVariant3(numberOfInputChannels, numberOfClasses)
    else:
        raise Exception("Invalid model name")

def getOptimizer(model,optimizer_name,learning_rate):
    if(optimizer_name == "adam"):
        return Adam(model.parameters(), lr=learning_rate)
    else:
        raise Exception("Invalid optimizer name")








def getLossFunction(loss_function_name):
    if(loss_function_name == "cross_entropy"):
        return nn.CrossEntropyLoss()
    else:
        raise Exception("Invalid loss function name")





def trainLoopOfTheNetwork(model,device,trainDataLoader,valDataLoader,lossFunction,optimizer,lenTrainingData,lenValData):
    train_loss_history = []
    train_accuracy_history = []
    val_loss_history = []
    val_accuracy_history = []

    for e in range(0,EPOCHS):
        train_loss = 0
        validation_loss = 0
        train_correct = 0
        val_correct = 0

        model.train()
        for data, labels in trainDataLoader:
            data, labels = data.to(device), labels.to(device)
            #initialising the gradients to zero
            optimizer.zero_grad()
            outputs = model(data)

            loss = lossFunction(outputs,labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, preds = torch.max(outputs,1)
            train_correct += torch.sum(preds == labels.data)

        #validate the model#
        model.eval()
        for data,labels in valDataLoader:
            data, labels = data.to(device), labels.to(device)
            val_outputs = model(data)
            val_loss = lossFunction(val_outputs, labels)
            validation_loss += val_loss.item()
            _, val_preds = torch.max(val_outputs,1)
            val_correct += torch.sum(val_preds == labels.data)

        train_loss = train_loss/ lenTrainingData
        train_acc = train_correct.double() / lenTrainingData
        validation_loss =  validation_loss / lenValData
        val_acc = val_correct.double() / lenValData


        # Append training and validation metrics to history lists
        train_loss_history.append(train_loss)
        train_accuracy_history.append(train_acc)
        val_loss_history.append(validation_loss)
        val_accuracy_history.append(val_acc)


        print('Epoch: {} \tTraining Loss: {:.8f} \tValidation Loss {:.8f} \tTraining Accuracy {:.3f}% \tValidation Accuracy {:.3f}%'
                                                            .format(e+1, train_loss,validation_loss,train_acc * 100, val_acc*100))

    return train_loss_history, train_accuracy_history, val_loss_history, val_accuracy_history



def evaluateModelWithTestData(testDataLoader,device,model):
    true_labels = []
    predicted_labels = []


    with torch.no_grad():
        for data, labels in testDataLoader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            pred = F.softmax(outputs, dim=1)
            classs = torch.argmax(pred, 1)
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(classs.cpu().numpy())


    return true_labels, predicted_labels


#write function with one default argument
def getScores(trueLabels,predictedLabels,typeOfScore = "notype",averageType = "notype"):
    if typeOfScore == "accuracy":
        return accuracy_score(trueLabels, predictedLabels)
    elif typeOfScore == "precision":
        return precision_score(trueLabels, predictedLabels, average=averageType)
    elif typeOfScore == "recall":
        return recall_score(trueLabels, predictedLabels, average=averageType)
    elif typeOfScore == "f1":
        return f1_score(trueLabels, predictedLabels, average=averageType)
    else:
        raise Exception("Invalid type of score")


def generateConfusionMatrix(trueLabels,predictedLabels,classes):
    # Generate and display a confusion matrix
    conf_matrix = confusion_matrix(trueLabels, predictedLabels)
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = conf_matrix.max() / 2.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, format(conf_matrix[i, j], fmt),
                    ha="center", va="center",
                    color="white" if conf_matrix[i, j] > thresh else "black")

    plt.ylabel('True labels')
    plt.xlabel('Predicted labels')
    plt.show()

def printClassificationReport(trueLabels,predictedLabels,classes):
    print("Classification Report:\n", classification_report(trueLabels, predictedLabels, target_names=classes))

def plotTrainingHistory(trainLossHistory, trainAccuracyHistory, valLossHistory, valAccuracyHistory):
     # Plot training and validation history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, EPOCHS + 1), trainLossHistory, label='Training Loss')
    plt.plot(range(1, EPOCHS + 1), valLossHistory, label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, EPOCHS + 1), trainAccuracyHistory, label='Training Accuracy')
    plt.plot(range(1, EPOCHS + 1), valAccuracyHistory, label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()

    plt.show()

def saveModel(model_state_dict,directoryPath=os.getcwd(),modelName="model.pth"):
    try:
        torch.save(model_state_dict, directoryPath+"/"+modelName)
        print("Model saved successfully")
    except:
        print("Error in saving the model")

# def loadModel(directoryPath=os.getcwd()):
#     return torch.(directoryPath)

def loadModel(modelArchitecture,savedModelName, directoryPath):
    model = modelArchitecture
    checkpoint = torch.load(directoryPath+'/'+savedModelName)
    model.load_state_dict(checkpoint)
    return model



#write me a function that loads the image and shows it
def loadAndShowImage(imagePath):
    try:
        # image = plt.imread(imagePath)
        # plt.imshow(image)
        # plt.show()

        img = cv2.imread()
        plt.imshow(img, cmap='gray')
        plt.show()
    except Exception as e:
        print(f"Error loading the image {imagePath}: {str(e)}")


def loadExternalImage(imagePath,cascadeClassifierPath):

    try:
        image = cv2.imread(imagePath)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_cascade_classifier = cv2.CascadeClassifier(cascadeClassifierPath)
        detected_faces = face_cascade_classifier.detectMultiScale(gray_image, 1.1, 4)

        # If faces are detected
        if len(detected_faces) > 0:
            # Loop through detected faces and save them
            for i, (x, y, w, h) in enumerate(detected_faces):
                face = image[y:y + h, x:x + w]
                face = cv2.resize(face, (48, 48))

                #normalise image face for cnn input
                #show the image
                # print("inside")
                # plt.imshow(face)
                # plt.show()
            return face

        else:
            print("No faces detected")

    except Exception as e:
        print(f"Error loading the image {imagePath}: {str(e)}")


def conv_block(in_channels, out_channels,kernal_size, pool=False):
    layers = [
              nn.Conv2d(in_channels, out_channels, kernel_size=kernal_size, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)
    ]

    if pool:
        layers.append(nn.MaxPool2d(kernel_size=2))

    return nn.Sequential(*layers)

class ModelVariant1(nn.Module):

    def __init__(self,numOfChannels, numOfClasses):
        super(ModelVariant1, self).__init__()
        self.conv1 = conv_block(numOfChannels, 16,3,pool=False) # 16 x 48 x 48
        self.conv2 = conv_block(16, 32,3,pool=True) # 32 x 24 x 24
        self.res1 = nn.Sequential( #  32 x 24 x 24
            conv_block(32, 32,3, pool=False),
            conv_block(32, 32,3, pool=False)
        )

        self.conv3 = conv_block(32, 64,3, pool=True) # 64 x 12 x 12
        self.conv4 = conv_block(64, 128,3, pool=True) # 128 x 6 x 6

        self.res2 = nn.Sequential( # 128 x 6 x 6
             conv_block(128, 128,3),
             conv_block(128, 128,3)
        )

        self.classifier = nn.Sequential(
            nn.MaxPool2d(kernel_size=2), # 128 x 3 x 3
            nn.Flatten(),
            nn.Linear(128*3*3, 512), #512
            nn.Linear(512, numOfClasses) # 7
        )
        self.network = nn.Sequential(
            self.conv1,
            self.conv2,
            self.res1,
            self.conv3,
            self.conv4,
            self.res2,
            self.classifier,
        )



    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out


class ModelVariant2(nn.Module):
    def __init__(self, numOfChannels, numOfClasses):

        super(ModelVariant2, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=numOfChannels, out_channels=8, kernel_size=3)
        self.cnn2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3)
        self.cnn3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.cnn4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.cnn5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.cnn6 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3)
        self.cnn7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.cnn1_bn = nn.BatchNorm2d(8)
        self.cnn2_bn = nn.BatchNorm2d(16)
        self.cnn3_bn = nn.BatchNorm2d(32)
        self.cnn4_bn = nn.BatchNorm2d(64)
        self.cnn5_bn = nn.BatchNorm2d(128)
        self.cnn6_bn = nn.BatchNorm2d(256)
        self.cnn7_bn = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, numOfClasses)
        self.dropout = nn.Dropout(0.3)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.relu(self.pool1(self.cnn1_bn(self.cnn1(x))))
        x = self.relu(self.pool1(self.cnn2_bn(self.dropout(self.cnn2(x)))))
        x = self.relu(self.pool1(self.cnn3_bn(self.cnn3(x))))
        x = self.relu(self.pool1(self.cnn4_bn(self.dropout(self.cnn4(x)))))
        x = self.relu(self.pool2(self.cnn5_bn(self.cnn5(x))))
        x = self.relu(self.pool2(self.cnn6_bn(self.dropout(self.cnn6(x)))))
        x = self.relu(self.pool2(self.cnn7_bn(self.dropout(self.cnn7(x)))))

        x = x.view(x.size(0), -1)

        x = self.relu(self.dropout(self.fc1(x)))
        x = self.relu(self.dropout(self.fc2(x)))
        x = self.log_softmax(self.fc3(x))
        return x


class ModelVariant3(nn.Module):
    def __init__(self, numOfChannels, numOfClasses):

        super(ModelVariant3, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=numOfChannels, out_channels=10, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3)
        self.poo2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3)
        self.poo4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.norm = nn.BatchNorm2d(10)

        self.fc1 = nn.Linear(in_features=810, out_features=50)
        self.fc2 = nn.Linear(in_features=50, out_features=numOfClasses)

        self.fc_loc = nn.Sequential(
            nn.Linear(640, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 3 * 2)
        )




    def forward(self, input):
        #out = self.stn(input)

        out = F.relu(self.conv1(input))
        out = self.conv2(out)
        out = F.relu(self.poo2(out))

        out = F.relu(self.conv3(out))
        out = self.norm(self.conv4(out))
        out = F.relu(self.poo4(out))

        out=F.dropout(out)
        out = out.view(-1, 810)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)

        return out



# set the device we will be using to train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

train_data, test_data = loadAndTransformDataSet()

#70% for training, 15% for validation
numTrainSamples = int(len(train_data) * TRAIN_SPLIT)
numValSamples = math.ceil(len(train_data) * VALIDATION_SPLIT)

(final_train_data, final_val_data) = random_split(train_data,
	[numTrainSamples, numValSamples],
	generator=torch.Generator().manual_seed(23))

print("Loading the data loader...")
trainDataLoader, valDataLoader, testDataLoader = getTheDataLoader(final_train_data,final_val_data,test_data)

# initialize the LeNet model
print("[INFO] initializing the LeNet model...")

model = getModel(model_name="variant2",numberOfInputChannels=1,numberOfClasses=4)


# model = CustomCNN(
# 	numOfChannels=1,
# 	numOfClasses=4).to(device)


# model = models.alexnet(weights=models.alexnet.DEFAULT) # load pretrained model
# model.fc = nn.Linear(512, 4)
print("Getting the optimizer and loss function...")
optmizer = getOptimizer(model=model,optimizer_name= "adam",learning_rate=INIT_LR)
lossFn = getLossFunction(loss_function_name="cross_entropy")

# this function returs train_loss_history, train_accuracy_history, val_loss_history, val_accuracy_history
print("Training the model and Evaluating the model on validation data ...")
train_loss_history, train_accuracy_history, val_loss_history, val_accuracy_history = trainLoopOfTheNetwork(model=model,device=device,
                                                                                                                           trainDataLoader=trainDataLoader,
                                                                                                                   valDataLoader=valDataLoader,
                                                                                                                   lossFunction=lossFn,
                                                                                                                   optimizer=optmizer,
                                                                                                                   lenTrainingData=len(final_train_data),
                                                                                                                   lenValData=len(final_val_data)
                                                                                                                   )




# Evaluate the network on the test set
print("[INFO] evaluating network...")

# Initialize lists to store true labels and predicted labels
true_labels = []
predicted_labels = []

true_labels,predicted_labels = evaluateModelWithTestData(model=model,testDataLoader=testDataLoader,device=device)

# Calculate scores
accuracy = getScores(trueLabels=true_labels,predictedLabels=predicted_labels,typeOfScore="accuracy")
precision_macro = getScores(trueLabels=true_labels,predictedLabels=predicted_labels,typeOfScore="precision",averageType="macro")
recall_macro = getScores(trueLabels=true_labels,predictedLabels=predicted_labels,typeOfScore="recall",averageType="macro")
f1_macro = getScores(trueLabels=true_labels,predictedLabels=predicted_labels,typeOfScore="f1",averageType="macro")
precision_micro = getScores(trueLabels=true_labels,predictedLabels=predicted_labels,typeOfScore="precision",averageType="micro")
recall_micro = getScores(trueLabels=true_labels,predictedLabels=predicted_labels,typeOfScore="recall",averageType="micro")
f1_micro = getScores(trueLabels=true_labels,predictedLabels=predicted_labels,typeOfScore="f1",averageType="micro")

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


generateConfusionMatrix(trueLabels=true_labels,predictedLabels=predicted_labels,classes=DATA_CLASSES)



# Printing classification report
printClassificationReport(trueLabels=true_labels,predictedLabels=predicted_labels,classes=DATA_CLASSES)


plotTrainingHistory(trainLossHistory=train_loss_history,trainAccuracyHistory= train_accuracy_history, valLossHistory=val_loss_history, valAccuracyHistory=val_accuracy_history)

saveModel(model_state_dict=model.state_dict(),directoryPath="saved_model",modelName="mymodel.pth")
