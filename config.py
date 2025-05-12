import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'data/processed')

TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'val')
TEST_DIR = os.path.join(DATA_DIR, 'test')

MODEL_SAVE_DIR = os.path.join('models')
PREDICTION_SAVE_DIR = os.path.join('outputs')

BATCH_SIZE = 8
LEARNING_RATE = 1e-4
NUM_EPOCHS = 20
INPUT_CHANNELS = 1  
OUTPUT_CHANNELS = 1