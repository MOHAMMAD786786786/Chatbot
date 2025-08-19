import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer

# ------------------------
# Step 1: Raw text data
# ------------------------
corpus = [
    "Artificial intelligence is changing the world.",
    "Language models learn from huge amounts of text.",
    "Neural networks are inspired by the brain.",
    "Transformers use attention to understand sequences."
]

# ------------------------
# Step 2: Tokenizer
# ------------------------
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# ------------------------
# Step 3: Custom Dataset
# ------------------------
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=16):
        self.encodings = []
        for text in texts:
            enc = tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt"
            )
            self.encodings.append(enc)

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        enc = self.encodings[idx]
        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze()
        }

# ------------------------
# Step 4: DataLoader
# ------------------------
dataset = TextDataset(corpus, tokenizer)
loader = DataLoader(dataset, batch_size=2, shuffle=True)

# ------------------------
# Step 5: Inspect batches
# ------------------------
for batch in loader:
    print(\"Input IDs:\", batch[\"input_ids\"])
    print(\"Attention Mask:\", batch[\"attention_mask\"])
    break
