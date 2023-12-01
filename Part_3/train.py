import helper_functions as helperfunctions
import math
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import ToTensor
from torchvision.datasets import ImageFolder
from torch.optim import SGD
import torch.nn as nn
import torch
import time
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import common_vars as GLOBAL_VARS
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import cross_validate, KFold
from model_variant2 import ModelVariant2
from model_variant1 import ModelVariant1
# set the device we will be using to train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Load and transform the dataset
dataset = helperfunctions.loadWholeDataSet()

num_splits = 10
#Improvement over Step2
#Keep the same data regardless of the model architecture
skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=42)

model = ModelVariant2(numOfChannels=1, numOfClasses=4)

#PART 3 K-fold cross-validation loop
# K-fold cross-validation loop
for fold, (train_indices, test_indices) in enumerate(skf.split(dataset,dataset.targets)):
    print(f"Fold {fold + 1}/{num_splits}")


    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    model = helperfunctions.getModel(model_name="variant2", numberOfInputChannels=1, numberOfClasses=4)
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = helperfunctions.getOptimizer(model=model, optimizer_name="adam", learning_rate=GLOBAL_VARS.INIT_LR)
    
    helperfunctions.train(model, train_dataset, optimizer, device,criterion,'Part_3/saved_models')
        
    
    true_labels, predicted_labels,c = helperfunctions.evaluate(model, test_loader,criterion=criterion,device=device)

    precision = precision_score(true_labels, predicted_labels, average='macro')
    recall = recall_score(true_labels, predicted_labels, average='macro')
    f1 = f1_score(true_labels, predicted_labels, average='macro')
    precision1 = precision_score(true_labels, predicted_labels, average='micro')
    recall1 = recall_score(true_labels, predicted_labels, average='micro')
    f11 = f1_score(true_labels, predicted_labels, average='micro')
    accuracy = accuracy_score(true_labels, predicted_labels)

    print(f"Macro Precision: {precision:.4f} | Macro Recall: {recall:.4f} | Macro F1 Score: {f1:.4f}\n"
      f"Micro Precision: {precision1:.4f} | Micro Recall: {recall1:.4f} | Micro F1 Score: {f11:.4f}\n"
      f"Accuracy: {accuracy:.4f}\n")


