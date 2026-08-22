import multiprocessing
import datetime
import os

import torch


Use_Model = "CNN"

# Paths to CSV lists (point to repo root CSV files)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
train_data = os.path.join(BASE_DIR, "Train_data.csv")
test_data = os.path.join(BASE_DIR, "Test_data.csv")

# Local dataset paths (relative to repository root)
TRAIN_DATA_DIR = "audio-dev-clean/LibriSpeech/dev-clean"
TEST_DATA_DIR = "audio-dev-clean/LibriSpeech/test-clean"

timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
filename = f"saved_model_{timestamp}.pt"
# Local models directory
SAVED_MODEL_PATH = "models/"

BATCH_SIZE = 8
NUM_PROCESSES = min(multiprocessing.cpu_count(), BATCH_SIZE)

TRAINING_STEPS = 100

USE_TRANSFORMER = False
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

N_MFCC = 40
LSTM_HIDDEN_SIZE = 64
LSTM_NUM_LAYERS = 3
BI_LSTM = True
FRAME_AGGREGATION_MEAN = True
LEARNING_RATE = 0.0001
SEQ_LEN = 100
SPECAUG_TRAINING = False
SAVE_MODEL_FREQUENCY=10

# Parameters for SpecAugment training.
SPECAUG_FREQ_MASK_PROB = 0.3
SPECAUG_TIME_MASK_PROB = 0.3
SPECAUG_FREQ_MASK_MAX_WIDTH = N_MFCC // 5
SPECAUG_TIME_MASK_MAX_WIDTH = SEQ_LEN // 5
TRIPLET_ALPHA = 0.1

# If true, we use transformer instead of LSTM.
USE_TRANSFORMER = False
# Dimension of transformer layers.
TRANSFORMER_DIM = 32
# Number of encoder layers for transformer
TRANSFORMER_ENCODER_LAYERS = 2
# Number of heads in transformer layers.
TRANSFORMER_HEADS = 8

# Number of triplets to evaluate for computing Equal Error Rate (EER).
# Both the number of positive trials and number of negative trials will be equal to this number.
NUM_EVAL_TRIPLETS = 100


USE_FULL_SEQUENCE_INFERENCE = True
SLIDING_WINDOW_STEP = 50
EVAL_THRESHOLD_STEP = 0.001
