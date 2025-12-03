# src/config.py

# These will be overridden to match the HF model's labels exactly
EMOTIONS = ["sadness", "joy", "love", "anger", "fear", "surprise"]

# Use a HuggingFace classification model directly (no local folder)
MODEL_NAME = "bhadresh-savani/distilbert-base-uncased-emotion"
MAX_LENGTH = 64
BATCH_SIZE = 8
NUM_EPOCHS = 3
LR = 5e-5

# Paths (relative to project root)
CSV_PATH = "data/songs.csv"
MODEL_DIR = MODEL_NAME
LEXICON_PATH = "lexicons/emotion_lexicon.csv"
OUTPUT_TIMELINES = "outputs/timelines"
OUTPUT_WORDCLOUDS = "outputs/wordclouds"
OUTPUT_SUMMARIES = "outputs/summaries"
