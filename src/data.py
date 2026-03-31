from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

TOKENIZER_NAME = "bert-base-uncased"
MAX_LENGTH = 128
BATCH_SIZE = 32

def get_tokenizer():
    return AutoTokenizer.from_pretrained(TOKENIZER_NAME)

def get_datasets(tokenizer):
    """
    Returns: dict with keys 'train', 'validation', 'test'
    Each value is a HuggingFace Dataset with columns:
        input_ids       : LongTensor [seq_len]
        attention_mask  : LongTensor [seq_len]
        label           : int (0 = negative, 1 = positive)
    """
    raw = load_dataset("imdb")

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )

    tokenized = raw.map(tokenize, batched=True)

    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    # IMDb has no official val split — carve 10% from train
    split = tokenized["train"].train_test_split(test_size=0.1, seed=42)
    return {
        "train":      split["train"],
        "validation": split["test"],
        "test":       tokenized["test"],
    }

def get_dataloaders(datasets, batch_size=BATCH_SIZE):
    """
    Returns: dict with keys 'train', 'validation', 'test'
    Each value is a DataLoader ready for training.
    """
    return {
        split: DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"))
        for split, ds in datasets.items()
    }