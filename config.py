import transformers

DEVICE = "cpu"
MAX_LEN = 64
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 5
BERT_PATH = "bert-base-uncased"
MODEL_PATH = "model.bin"
TRAINING_FILE = "twitter_cleaned_1.csv"
TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)
OUTPUT_PATH = "output/"

