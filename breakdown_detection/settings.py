SEED = 1234  # Seeding for consistency in reproducibility

AVAILABLE_FEATURES = [
    'intents',  # one hot representation
    'entities_enc',  # entities encoded as strings using SBERT
    'entities_mh',  # multi-one hot representation
    'callers'  # NLU or NLG - one hot representation
    ]

DATA_FOLDER = 'dataset/'
SENTENCE_ENCODER_MODEL_NAME = 'all-MiniLM-L6-v2'
PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'
SELECTED_TARGET = 'luhfs'

# the ratio of augmented calls to add wrt to the total training instances
AUGMENTATION_RATIO = 0.3
BATCH_SIZE = 256

TURN_THRESHOLD = 8  # defines when a call is considered "LATE"
