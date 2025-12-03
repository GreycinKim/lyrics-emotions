from torch.utils.data import Dataset
from transformers import AutoTokenizer
import pandas as pd
from .config import EMOTIONS, MODEL_NAME, MAX_LENGTH

class EmotionDataset(Dataset):
    def __init__(self, csv_path: str):
        self.df: pd.DataFrame = pd.read_csv(csv_path)
        self.df = self.df.reset_index(drop=True)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.label2id = {e: i for i, e in enumerate(EMOTIONS)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = row["text"]
        label = self.label2id[row["label"]]

        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
            return_tensors="pt"
        )

        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = label
        return item
