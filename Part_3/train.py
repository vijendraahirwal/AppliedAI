import helper_functions as helperfunctions
import math
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import ToTensor
from torchvision.datasets import ImageFolder
from torch.optim import SGD
import torch.nn as nn
import torch
import time
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import common_vars as GLOBAL_VARS

# set the device we will be using to train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and transform the dataset
train_data, test_data = helperfunctions.loadAndTransformDataSet()

# Combine train and validation data for k-fold cross-validation
full_train_data = torch.utils.data.ConcatDataset([train_data, test_data])

# Define k-fold cross-validation
num_splits = 2  # You can choose the number of splits
kf = KFold(n_splits=num_splits, shuffle=True, random_state=42)

# Initialize lists to store metrics for each fold
all_train_loss_history = []
all_train_accuracy_history = []
all_val_loss_history = []
all_val_accuracy_history = []

# Perform k-fold cross-validation
for fold, (train_indices, val_indices) in enumerate(kf.split(full_train_data)):
    print(f"\nFold {fold + 1}/{num_splits}")

    # Create data loaders for the current fold
    train_subset = torch.utils.data.Subset(full_train_data, train_indices)
    val_subset = torch.utils.data.Subset(full_train_data, val_indices)

    trainDataLoader, valDataLoader, testDataLoader = helperfunctions.getTheDataLoader(train_subset, val_subset, test_data)

    # Initialize the LeNet model
    print("[INFO] Initializing the LeNet model...")
    model = helperfunctions.getModel(model_name="variant3", numberOfInputChannels=1, numberOfClasses=4)

    # Get the optimizer and loss function
    optimizer = helperfunctions.getOptimizer(model=model, optimizer_name="sgd", learning_rate=GLOBAL_VARS.INIT_LR)
    loss_fn = helperfunctions.getLossFunction(loss_function_name="cross_entropy")

    # Train and evaluate the model for the current fold
    train_loss_history, train_accuracy_history, val_loss_history, val_accuracy_history = helperfunctions.trainLoopOfTheNetwork(
        model=model,
        device=device,
        trainDataLoader=trainDataLoader,
        valDataLoader=valDataLoader,
        lossFunction=loss_fn,
        optimizer=optimizer,
        lenTrainingData=len(train_subset),
        lenValData=len(val_subset),
        earlystopping=True,
    )

    # Save metrics for the current fold
    all_train_loss_history.append(torch.tensor(train_loss_history))  # Convert to PyTorch tensor
    all_train_accuracy_history.append(torch.tensor(train_accuracy_history))  # Convert to PyTorch tensor
    all_val_loss_history.append(torch.tensor(val_loss_history))  # Convert to PyTorch tensor
    all_val_accuracy_history.append(torch.tensor(val_accuracy_history))  # Convert to PyTorch tensor

    # Convert lists to PyTorch tensors
    all_train_loss_history = torch.stack(all_train_loss_history)
    all_train_accuracy_history = torch.stack(all_train_accuracy_history)
    all_val_loss_history = torch.stack(all_val_loss_history)
    all_val_accuracy_history = torch.stack(all_val_accuracy_history)

    # Calculate and print the average metrics over all folds
    avg_train_loss_history = torch.mean(all_train_loss_history, dim=0)
    avg_train_accuracy_history = torch.mean(all_train_accuracy_history, dim=0)
    avg_val_loss_history = torch.mean(all_val_loss_history, dim=0)
    avg_val_accuracy_history = torch.mean(all_val_accuracy_history, dim=0)

    # Print average training history over all folds
    print("\nAverage Metrics Over All Folds:")
    print(f"Average Train Loss: {avg_train_loss_history[-1]:.4f}")
    print(f"Average Train Accuracy: {avg_train_accuracy_history[-1]:.4f}")
    print(f"Average Validation Loss: {avg_val_loss_history[-1]:.4f}")
    print(f"Average Validation Accuracy: {avg_val_accuracy_history[-1]:.4f}")

    # Evaluate the network on the test set
    print("[INFO] Evaluating network...")

    # Initialize lists to store true labels and predicted labels
    true_labels = []
    predicted_labels = []

    # Evaluate the model on the test set for each fold
    model.eval()
    for data, labels in testDataLoader:
        data, labels = data.to(device), labels.to(device)
        outputs = model(data)
        _, preds = torch.max(outputs, 1)
        true_labels.extend(labels.cpu().numpy())
        predicted_labels.extend(preds.cpu().numpy())

    # Calculate scores
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision_macro = precision_score(true_labels, predicted_labels, average='macro')
    recall_macro = recall_score(true_labels, predicted_labels, average='macro')
    f1_macro = f1_score(true_labels, predicted_labels, average='macro')
    precision_micro = precision_score(true_labels, predicted_labels, average='micro')
    recall_micro = recall_score(true_labels, predicted_labels, average='micro')
    f1_micro = f1_score(true_labels, predicted_labels, average='micro')

    # Print metrics for each fold
    print("\nMetrics for Fold {}: ".format(fold + 1))
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision (macro): {precision_macro:.2f}")
    print(f"Recall (macro): {recall_macro:.2f}")
    print(f"F1-score (macro): {f1_macro:.2f}")
    print(f"Precision (micro): {precision_micro:.2f}")
    print(f"Recall (micro): {recall_micro:.2f}")
    print(f"F1-score (micro): {f1_micro:.2f}")
    
    all_train_loss_history = []
    all_train_accuracy_history = []
    all_val_loss_history = []
    all_val_accuracy_history = []
        



# Plot average training history over all folds
helperfunctions.plotTrainingHistory(
    trainLossHistory=avg_train_loss_history,
    trainAccuracyHistory=avg_train_accuracy_history,
    valLossHistory=avg_val_loss_history,
    valAccuracyHistory=avg_val_accuracy_history,
)
