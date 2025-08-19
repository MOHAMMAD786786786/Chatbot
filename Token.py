import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW

# ------------------------
# Step 1: Tokenizer
# ------------------------
# Tokenizer splits text into tokens (numbers for words/subwords)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Example text data
texts = [
    "I love machine learning.",
    "Transformers are powerful models.",
    "Large language models can write code."
]

# ------------------------
# Step 2: Custom Dataset
# ------------------------
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=32):
        self.inputs = []
        for text in texts:
            enc = tokenizer(text, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
            self.inputs.append(enc)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        item = self.inputs[idx]
        return {
            "input_ids": item["input_ids"].squeeze(),
            "attention_mask": item["attention_mask"].squeeze()
        }

dataset = TextDataset(texts, tokenizer)
loader = DataLoader(dataset, batch_size=2, shuffle=True)

# ------------------------
# Step 3: Model
# ------------------------
# GPT-2 small model
model = GPT2LMHeadModel.from_pretrained("gpt2")

# ------------------------
# Step 4: Training Setup
# ------------------------
optimizer = AdamW(model.parameters(), lr=5e-5)
loss_fn = torch.nn.CrossEntropyLoss()

# ------------------------
# Step 5: Training Loop
# ------------------------
model.train()
for epoch in range(2):  # small demo: 2 epochs
    for batch in loader:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1} - Loss: {loss.item()}")

# ------------------------
# Step 6: Generate Text
# ------------------------
model.eval()
prompt = "AI is"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=15, num_return_sequences=1)
print("Generated:", tokenizer.decode(outputs[0], skip_special_tokens=True))
