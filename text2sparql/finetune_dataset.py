import json
import torch
from torch.utils.data import Dataset

class SFTDataset(Dataset):
    def __init__(self, sft_file, tokenizer, mode='train', max_length=768):
        with open(sft_file, "r", encoding="utf-8") as f:
            self.examples = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        sft_text = example["sft"]
        tokenized_sft = self.tokenizer(
            sft_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            add_special_tokens=True,
            return_tensors="pt"
        )

        input_ids = tokenized_sft['input_ids'][0].to(torch.long)
        attention_mask = tokenized_sft['attention_mask'][0].to(torch.bool)

        if self.mode == "train":
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
        else:
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "gold_query": example['sparql'],
                "id": example['id'],
            }
