# define training hyperparameters
INIT_LR = 0.001
BATCH_SIZE = 64
EPOCHS = 100
# define the train and val splits
TRAIN_SPLIT = 0.85  
VALIDATION_SPLIT = 1 - TRAIN_SPLIT
WHOLE_DATASET_DIR = "Part_3/DataKF" 
TRAINING_DATA_DIR ="Part_3/dataset_inshape/train"
TESTING_DATA_DIR = "Part_3/dataset_inshape/test"

DATA_CLASSES = ('Angry', 'Bored', 'Focused', 'Neutral')

NUMBER = 0