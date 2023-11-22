# define training hyperparameters
INIT_LR = 0.001
BATCH_SIZE = 64
EPOCHS = 100
# define the train and val splits
TRAIN_SPLIT = 0.85
VALIDATION_SPLIT = 1 - TRAIN_SPLIT

TRAINING_DATA_DIR ="AppliedAI/Part_2/dataset_inshape/train"
TESTING_DATA_DIR = "AppliedAI/Part_2/dataset_inshape/test"

DATA_CLASSES = ('Angry', 'Focused', 'Bored', 'Neutral')

NUMBER = 0