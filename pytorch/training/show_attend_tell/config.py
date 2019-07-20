## Data paths
from paths import flickr8k_bigger_paths as PATHS

## Preprocessing options
NEW_INTERMEDIATE_DATA = False # voc, captions, image names
NEW_CNN_ACTIVATIONS = False # annotation vectors
MIN_WORD_COUNT = 1 # voc size is fixed to 10000 in the paper
MAX_CAPTION_LENGTH = 40 # does not count <start>, <end>

## Checkpointing
LOAD_MODEL = False
MODEL_LOAD_FILE = PATHS["ckpt_dir"]+'model_bigger_F_0003600'
MODEL_SAVE_FILE = PATHS["ckpt_dir"]+'model_bigger_F'
PRINT_EVERY = 500
SAVE_EVERY = 2000

## Model dimensions
CELL_DIM = 512 # 'n'
EMBEDDING_DIM = 512  # 'm'

## Training options
TRAIN = True
BATCH_SIZE = 64
N_ITERATIONS = 1000000
LEARNING_RATE = 0.001
CLIP = 50.0
DOUBLE_ATTN_LAMBDA = 0.01

## Testing
BATCH_TEST = False # otherwise interactive mode

## Debug
NUM_LINES = None


