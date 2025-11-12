"""
Configuration file for Banana Ripeness Classification
"""
import os

# Dataset paths
DATASET_PATH = r"C:\Users\Admin\Downloads\archive\Banana Ripeness Classification Dataset"
TRAIN_DIR = os.path.join(DATASET_PATH, "train")
VALID_DIR = os.path.join(DATASET_PATH, "valid")
TEST_DIR = os.path.join(DATASET_PATH, "test")

# Model paths
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")
FINAL_MODEL_PATH = os.path.join(MODEL_DIR, "final_model.pth")

# Classes and their corresponding days until dying
CLASS_TO_DAYS = {
    'unripe': 7,      # Green banana - 7 days
    'ripe': 3,        # Perfect yellow - 3 days
    'overripe': 1,    # Brown spots - 1 day
    'rotten': 0       # Too late - 0 days
}

CLASS_NAMES = ['overripe', 'ripe', 'rotten', 'unripe']
NUM_CLASSES = len(CLASS_NAMES)

# Training hyperparameters
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
IMAGE_SIZE = 224
NUM_WORKERS = 4

# Model architecture
MODEL_NAME = 'resnet50'  # Options: resnet50, efficientnet_b0, mobilenet_v2
PRETRAINED = True

# Training settings
PATIENCE = 10  # Early stopping patience
MIN_DELTA = 0.001  # Minimum improvement for early stopping

# Device
DEVICE = 'cuda'  # Will auto-detect in code
