## Data paths
from paths import flickr8k_bigger_paths as PATHS

## Preprocessing options
NEW_INTERMEDIATE_DATA = False # voc, captions, image names
NEW_CNN_ACTIVATIONS = False # annotation vectors
MIN_WORD_COUNT = 1 # voc size is fixed to 10000 in the paper
MAX_CAPTION_LENGTH = 40 # does not count <start>, <end>

## Checkpointing
LOAD_MODEL = True
MODEL_LOAD_FILE = PATHS["ckpt_dir"]+'model_bigger_H_0004000'
MODEL_SAVE_FILE = PATHS["ckpt_dir"]+'model_bigger_K'
PRINT_EVERY = 100
SAVE_EVERY = 1000

## Model dimensions
CELL_DIM = 512 # 'n'
EMBEDDING_DIM = 64 # 'm'

## Training options
TRAIN = True
BATCH_SIZE = 64
N_ITERATIONS = 1000000
LEARNING_RATE = 0.001
CLIP = 50.0
DOUBLE_ATTN_LAMBDA = 10

## Testing
BATCH_TEST = False # otherwise interactive mode

## Debug
NUM_LINES = None

