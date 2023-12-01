
import common_vars as GLOBAL_VARS
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from torchvision import transforms
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
import numpy as np
import sklearn
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import cv2 
from model_variant2 import ModelVariant2
from model_variant3 import ModelVariant3
from matplotlib import image as mp_image
from model_variant1 import ModelVariant1
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from skorch import NeuralNetClassifier
from skorch.helper import predefined_split
from sklearn.model_selection import GridSearchCV
import torch.nn.init as init
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from sklearn.model_selection import train_test_split

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
    return train_data,test_data


def loadWholeDataSet():
    dataset_dir = GLOBAL_VARS.WHOLE_DATASET_DIR
   
    resizing = transforms.Resize(size=(48, 48))
    horizontal_flip = transforms.RandomHorizontalFlip(p=0.3)
    grayscale = transforms.Grayscale(num_output_channels=1)
    #normalise the image 
    normalize = transforms.Normalize((0.5), (0.5), inplace=True)   
    data_transformations = transforms.Compose([grayscale,resizing, horizontal_flip, ToTensor(),normalize])
    data = ImageFolder(root=dataset_dir, transform=data_transformations)
    return data


def getTheDataLoader(train_data,val_data,test_data):
    
    trainDataLoader = DataLoader(train_data, shuffle=True,batch_size=GLOBAL_VARS.BATCH_SIZE)
    valDataLoader = DataLoader(val_data, batch_size=GLOBAL_VARS.BATCH_SIZE)
    testDataLoader = DataLoader(test_data, batch_size=GLOBAL_VARS.BATCH_SIZE)
    
    return trainDataLoader, valDataLoader, testDataLoader

def getTheDataLoader(train_data,val_data):
    
    trainDataLoader = DataLoader(train_data, shuffle=True,batch_size=GLOBAL_VARS.BATCH_SIZE)
    valDataLoader = DataLoader(val_data, batch_size=GLOBAL_VARS.BATCH_SIZE)
    
    return trainDataLoader, valDataLoader

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
        return torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.4)
    else:
        raise Exception("Invalid optimizer name")
    

def getLossFunction(loss_function_name):
    if(loss_function_name == "cross_entropy"):
        return nn.CrossEntropyLoss()
    else:
        raise Exception("Invalid loss function name")
    

def evaluate(model, dataloader,criterion,device):
    model.eval()
    all_labels = []
    all_preds = []

    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    

    with torch.no_grad():
        for inputs, labels in dataloader:
            # Forward pass
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels) 

            # Calculate accuracy
            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels).item()
            total_samples += labels.size(0)

            # Accumulate total loss
            total_loss += loss.item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # Calculate accuracy and loss
    accuracy = correct_predictions / total_samples
    average_loss = total_loss / len(dataloader)


    return all_labels, all_preds,average_loss
    
def printClassificationReport(trueLabels,predictedLabels,classes):
    print("Classification Report:\n", classification_report(trueLabels, predictedLabels, target_names=classes))
    
    
def train(model, train_dataset, optimizer,device, criterion,save_path,patience=5):
    best_valid_loss = float('inf')
    current_patience = 0
    
    train_size = int(0.8 * len(train_dataset))  # Adjust the ratio as needed
    val_size = len(train_dataset) - train_size
    train_dataset1, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset1, batch_size=64, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    for epoch in range(GLOBAL_VARS.EPOCHS):
        # Training
        model.train()
        running_loss = 0.0
        total_correct = 0
        total_samples = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            
            loss.backward()
            optimizer.step()

           
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

        # Calculate average loss and accuracy for the epoch
        average_loss = running_loss / len(train_loader)
        accuracy = total_correct / total_samples
        #print(f'Epoch [{epoch + 1}], Train Loss: {average_loss:.4f}, Accuracy: {accuracy:.4f}')

        # Validation
        model.eval()
        valid_loss = 0.0
        total_correct_valid = 0
        total_samples_valid = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                valid_loss += criterion(outputs, labels).item()

                _, predicted_valid = torch.max(outputs, 1)
                total_correct_valid += (predicted_valid == labels).sum().item()
                total_samples_valid += labels.size(0)

        valid_loss /= len(val_loader)
        accuracy_valid = total_correct_valid / total_samples_valid
        #print(f'Epoch [{epoch + 1}], Validation Loss: {valid_loss:.4f}, Accuracy: {accuracy_valid:.4f}')

        # Early stopping check
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            current_patience = 0
            # Save the best model
            torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pth'))
        else:
            current_patience += 1
            if current_patience >= patience:
                print(f'Validation loss has not improved. Early stopping...')
                break

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


def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def get_true_label(file_path):
    # Assuming your file path structure is consistent
    # You can split the path and get the label from the appropriate position
    label = str(file_path.split("/")[-2]).lower()  # Assumes the label is the second-to-last element in the path
    if(label=="angry"):
        return 0
    elif(label=="bored"):
        return 1
    elif(label=="focused"):
        return 2
    elif(label=="neutral"):
        return 3


# def inputCategory():
#     input=input("Enter the category")
#     if(input=="young"):
#         return "young"
#     elif(input=="middle_age"):
#         return "middle_age"
#     elif(input=="senior"):
#         return "senior"
#     else:
#         raise Exception("Invalid input")    

def Loop():
    Category = [1,2,3]
    Gender = ["m","f"]
    for i in Category:
        evaluateModelOnSpecificCategory('Age Category',i)
    
    for i in Gender:
        evaluateModelOnSpecificCategory('Gender',i)
    
def evaluateModelOnSpecificCategory(Column,value):
    # Load CSV file containing file_path, gender, and age
    csv_file_path = 'test_labels1.csv'
    df = pd.read_csv(csv_file_path)

    # Filter images based on gender

    #images = df[df['Gender'] =='m' ]['Image Path'].tolist()
    

    #images = df[df['Gender'] =='f']['Image Path'].tolist()
    images = df[df[Column] ==value]['Image Path'].tolist()

    # Initialize your model and load the pre-trained weights
    model = ModelVariant2(numOfChannels=1, numOfClasses=4)
    model_path = 'Part_3/saved_models/best_model.pth'
    model = load_model(model, model_path)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)


    # Evaluate on the test set
    model.eval()
    test_predictions = []
    test_labels = []

    with torch.no_grad():
        for file_path in images:
        
            file_path = file_path.replace("AppliedAI/", "")
           
            # Define transformations for the input image
            input_image = mp_image.imread(file_path)
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize((0.5), (0.5), inplace=True)
            ])
            input_tensor = transform(input_image).unsqueeze(0)
            #input_tensor = input_tensor.to(device)
            output = model(input_tensor)
            _, predicted = torch.max(output, 1)
            test_predictions.append(predicted.item())
            # Get the true label from the file_path or dataframe
            
            true_label = get_true_label(file_path)  # Implement this function
            test_labels.append(true_label)


    
    # # Calculate evaluation metrics
    print("#############################################")
    print("AFTER RESOLVING THE BIAS")
    #print("For Senior people")
    if Column=='Age Category':
        if value==1:
            print("For Senior people")
        elif value==2:
            print("For Middle Age people")
        elif value==3:
            print("For Young people")
    else:
        print("For",Column,"-",value)

    accuracy = accuracy_score(test_labels, test_predictions)
    precision = precision_score(test_labels, test_predictions, average='weighted')
    recall = recall_score(test_labels, test_predictions, average='weighted')
    f1 = f1_score(test_labels, test_predictions, average='weighted')

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print()
#Loop()

      
def separatelyRunOnTestSet(model, directory_path):
    model.eval()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,), inplace=True)
    ])

    # Lists to store predictions and true labels
    test_predictions = []
    test_labels = []

    for root, dirs, files in os.walk("Part_3/dataset_inshape/test"):
       
        for file in files:
            file_path = os.path.join(root, file)
            path_components = file_path.split(os.path.sep)
            label = path_components[-2].lower()

            #label = str(file_path.split("/")[-1]).lower()
            #print("label is "+label)
            if(label=="angry"):
                label=0
            elif(label=="bored"):
                label=1
            elif(label=="focused"):
                label=2
            elif(label=="neutral"):
                label=3


            # Read the image
            input_image = mp_image.imread(file_path) # Convert to grayscale

            # Apply transformations
            input_tensor = transform(input_image).unsqueeze(0)

            # Make predictions
            output = model(input_tensor)
            _, predicted = torch.max(output, 1)
            test_predictions.append(predicted.item())
            
            true_label = label
            test_labels.append(true_label)

    

    
    # # List all files in the directory
    # #image_files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]

    # # Transformation pipeline
    # transform = transforms.Compose([
    #     transforms.ToPILImage(),
    #     transforms.Grayscale(num_output_channels=1),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5,), (0.5,), inplace=True)
    # ])

    

    # # Loop through each image in the directory
    # for file_name in image_files:
    #     file_path = os.path.join(directory_path, file_name)

    #     # Read the image
    #     input_image = mp_image.imread(file_path) # Convert to grayscale

    #     # Apply transformations
    #     input_tensor = transform(input_image).unsqueeze(0)

    #     # Make predictions
    #     output = model(input_tensor)
    #     _, predicted = torch.max(output, 1)
    #     test_predictions.append(predicted.item())
        
    #     true_label = label
    #     test_labels.append(true_label)

    # # Print or use test_predictions and test_labels as needed
    # print("Predictions:", test_predictions)
    # print("True Labels:", test_labels)

    # Calculate and print evaluation metrics
    accuracy = accuracy_score(test_labels, test_predictions)
    precision = precision_score(test_labels, test_predictions, average='weighted')
    recall = recall_score(test_labels, test_predictions, average='weighted')
    f1 = f1_score(test_labels, test_predictions, average='weighted')

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

    generateConfusionMatrix(trueLabels=test_labels,predictedLabels=test_predictions,classes=GLOBAL_VARS.DATA_CLASSES)

    # Printing classification report
    #printClassificationReport(trueLabels=test_labels,predictedLabels=test_predictions,classes=GLOBAL_VARS.DATA_CLASSES)

   


model = ModelVariant2(numOfChannels=1, numOfClasses=4)
model_path = 'Part_3/saved_models/best_so_far_after_bias1.pth'
model = load_model(model, model_path)

directory_path = 'Part_3/dataset_inshape/test/neutral'
separatelyRunOnTestSet(model, directory_path)
    
    
def runCV(train_loader,valid_loader,dataset_classes,device):
    
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
    saveBestHyperparameters(gs.best_score_, f"AppliedAI/Part_2/best_params.yml")
    saveBestHyperparameters(gs.best_params_, f"AppliedAI/Part_2/best_params.yml")


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
 
    
 
 
        
def saveModel(model_state_dict,directoryPath=os.getcwd(),modelName="model.pth"):
    try:
        torch.save(model_state_dict, directoryPath+"/"+modelName)
        print("Model saved successfully")
    except:
        print("Error in saving the model")
        

    
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
    
    
    
def calcDimensons(width, height, padding, kernel_size, stride):
    """
    Calculate the output dimensions of a convolutional layer
    """
    out_width = ((width + (2 * padding) - (kernel_size)) / stride) + 1
    out_height = ((height + (2 * padding) - (kernel_size))/ stride) + 1
    return int(out_width), int(out_height)


def saveBestHyperparameters(txt, path):
    with open(path, 'a') as f:
        f.write(f"{str(txt)}\n")
    print("Saved best hyperparameters to file.")
    
    
    
def k_fold_cross_validation(dataset, k=10):
    skf = StratifiedKFold(n_splits=k, shuffle=True)
    for train_index, test_index in skf.split(dataset.data, dataset.targets):
        yield train_index, test_index

def log_metrics_to_table(metrics_dict):
    df = pd.DataFrame(metrics_dict)
    # Calculate averages
    df.loc['Average'] = df.mean()
    return df

