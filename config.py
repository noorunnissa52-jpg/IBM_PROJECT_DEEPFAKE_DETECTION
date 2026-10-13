"""Central configuration. Command-line flags in train.py override these defaults."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "checkpoints"
OUT_DIR = ROOT / "outputs"

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
LR = 3e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 2
SEED = 42

# Class index mapping (alphabetical, same as ImageFolder): fake -> 0, real -> 1
CLASS_NAMES = ["fake", "real"]
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
