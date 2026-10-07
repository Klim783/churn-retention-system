import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class ChurnMultimodalDataset(Dataset):
    def __init__(
        self,
        df_tabular,
        sequences,
        texts,
        labels,
        tokenizer_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_token_len: int = 128,
    ):
        self.x_tab = torch.tensor(df_tabular.values, dtype=torch.float32)
        self.x_seq = torch.tensor(sequences, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
        self.texts = texts

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_token_len = max_token_len

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_token_len,
            return_tensors="pt",
        )

        return {
            "x_tab": self.x_tab[idx],
            "x_seq": self.x_seq[idx],
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": self.labels[idx],
        }