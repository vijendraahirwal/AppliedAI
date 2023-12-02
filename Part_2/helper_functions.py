import common_vars as GLOBAL_VARS
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets import ImageFolder
from torch.optim import Adam
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms, models
import os
import sklearn
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import cv2 
from model_variant2 import ModelVariant2
from model_variant3 import ModelVariant3
from model_variant1 import ModelVariant1
from early_stopping import EarlyStopping
import torch.optim as optim
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from skorch import NeuralNetClassifier
from skorch.helper import predefined_split
from sklearn.model_selection import GridSearchCV
import torch.nn.init as init

def loadAndTransformDataSet():
    
    training_data_dir = GLOBAL_VARS.TRAINING_DATA_DIR
    test_data_dir = GLOBAL_VARS.TESTING_DATA_DIR
    
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
    print(f"Classes are {train_data.classes}")
    return train_data, test_data


def getTheDataLoader(train_data,val_data,test_data):
    
    trainDataLoader = DataLoader(train_data, shuffle=True,batch_size=GLOBAL_VARS.BATCH_SIZE)
    valDataLoader = DataLoader(val_data, batch_size=GLOBAL_VARS.BATCH_SIZE)
    testDataLoader = DataLoader(test_data, batch_size=GLOBAL_VARS.BATCH_SIZE)
    
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
    elif(optimizer_name == "sgd"):
        return torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    else:
        raise Exception("Invalid optimizer name")
    
def getLossFunction(loss_function_name):
    if(loss_function_name == "cross_entropy"):
        return nn.CrossEntropyLoss()
    else:
        raise Exception("Invalid loss function name")
    
def trainLoopOfTheNetwork(model,device,trainDataLoader,valDataLoader,lossFunction,optimizer,lenTrainingData,lenValData,earlystopping=False):
    train_loss_history = []
    train_accuracy_history = []
    val_loss_history = []
    val_accuracy_history = []
    
    earlystopping = EarlyStopping(tolerance=15, min_delta=0.02)

    for e in range(0,GLOBAL_VARS.EPOCHS):
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
        # if earlystopping :
        #     earlystopping(train_loss, validation_loss)
        #     if earlystopping.early_stop:
        #         print("[IMPORTANT] Early stopping at epoch: ",e+1)
        #         break

    train_accuracy_history = [tensor.item() for tensor in train_accuracy_history]
    val_accuracy_history = [tensor.item() for tensor in val_accuracy_history]
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
    
def runCV(train_loader,valid_loader,dataset_classes,device):
    # Instance of `NeuralNetClassifier` to be passed to `GridSearchCV` 
    net = NeuralNetClassifier(
        module=ModelVariant2, max_epochs=20,
        optimizer=torch.optim.Adam,
        criterion=nn.CrossEntropyLoss(),
        lr=0.001, verbose=1
    )
    
    params = {
    'batch_size' : [7,20,50,70],
    'lr': [0.001,0.0001,0.005,0.01],
    'module__numOfChannels':[1],
    'module__numOfClasses':[4],
    'module__activation': [nn.Identity(), nn.ReLU(), nn.ELU(), nn.ReLU6(),
                        nn.Softsign(), nn.Tanh(),
                           nn.Sigmoid()
                    
                           ],
    'max_epochs': [10,20,40,50],
    'optimizer': [optim.SGD, optim.RMSprop,optim.Adamax, optim.NAdam,optim.Adagrad, optim.Adadelta,
                  optim.Adam],
    'module__dropout':[0.1,0.2,0.3,0.5]
    
        
    }
    """
    Define `GridSearchCV`.
    4 lrs * 7 max_epochs * 4 module__first_conv_out * 3 module__first_fc_out
    * 2 CVs = 672 fits.
    """
    gs = GridSearchCV(
        net, params, refit=True, scoring='accuracy', n_jobs=-1, verbose=1, cv=2
    )
    counter = 0
    
    search_batches = 2
   
    for i, data in enumerate(train_loader):
        counter += 1
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)
        outputs = gs.fit(image, labels)
        # GridSearch for `search_batches` number of times.
        if counter == search_batches:
            break
    print('SEARCH COMPLETE')
    print("best score: {:.3f}, best params: {}".format(gs.best_score_, gs.best_params_))
    saveBestHyperparameters(gs.best_score_, f"Part_2/best_params.yml")
    saveBestHyperparameters(gs.best_params_, f"Part_2/best_params.yml")


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
    epochs = range(1, len(trainLossHistory) + 1)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, trainLossHistory, label='Training Loss', marker='o')
    plt.plot(epochs, valLossHistory, label='Validation Loss', marker='o')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, trainAccuracyHistory, label='Training Accuracy', marker='o')
    plt.plot(epochs, valAccuracyHistory, label='Validation Accuracy', marker='o')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()
 
#saves the model to the given path with all the parameters
def saveModel(model, numOfChannels, numOfClasses, directoryPath, modelName):
    model_info = {
        "state_dict": model.state_dict(),
        "model_class": model.__class__.__name__,
        "init_args": {
            "numOfChannels": numOfChannels,
            "numOfClasses": numOfClasses
        }
    }
    save_path = os.path.join(directoryPath, modelName)
    try:
        torch.save(model_info, save_path)
        print(f"Model saved successfully at {save_path}")
    except Exception as e:
        print(f"Error in saving the model: {e}")

#loads the model from the given path
def loadModel(filePath):
    checkpoint = torch.load(filePath) 
    model_class = checkpoint['model_class']
    init_args = checkpoint['init_args']
    print(f"Loading model of type {model_class}...")
    
    model = globals()[model_class](**init_args)
    model.load_state_dict(checkpoint['state_dict'])

    return model

# def loadModel(filePath,modelArchitecture,savedModelName):
#     model = modelArchitecture
#     checkpoint = torch.load(filePath+'/'+savedModelName)
#     model.load_state_dict(checkpoint)
#     return model


#loads image from the given path and shows it
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
    
    
#Calculate the output dimensions of a convolutional layer
def calcDimensons(width, height, padding, kernel_size, stride):
    out_width = ((width + (2 * padding) - (kernel_size)) / stride) + 1
    out_height = ((height + (2 * padding) - (kernel_size))/ stride) + 1
    return int(out_width), int(out_height)


def saveBestHyperparameters(txt, path):
    with open(path, 'a') as f:
        f.write(f"{str(txt)}\n")
    print("Saved best hyperparameters to file.")

#take input from user
def getmodelname():
    print("Select a model from the following: ")
    print("1. Variant1")
    print("2. Variant2")
    print("3. Variant3")
    option = input("Enter your choice: ")
    switcher = {
        "1": "variant1",
        "2": "variant2",
        "3": "variant3",
    }
    return switcher.get(option, "Invalid model name")
