import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------
# Token + Positional Embeddings
# ------------------------
class Embeddings(nn.Module):
    def __init__(self, vocab_size, d_model, max_len):
        super().__init__()
        # Turns each word (as a number) into a vector of size d_model
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        # Adds information about the position of each word in the sentence
        self.position_embeddings = nn.Embedding(max_len, d_model)

    def forward(self, x):
        # Create position numbers for each word in the sentence
        positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        # Add word meaning (token) + position info together
        return self.token_embeddings(x) + self.position_embeddings(positions)

# ------------------------
# Multi-Head Attention
# ------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        # d_model is split across heads (like splitting tasks into groups)
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        # Layers that prepare queries, keys, and values (like ingredients for attention)
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        # Final layer to combine the heads back together
        self.fc = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        B, T, D = q.size()  # Batch size, sentence length, hidden size

        # Make queries, keys, and values, and split into multiple heads
        q = self.q_linear(q).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(k).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        # Compare queries and keys to see which words should pay attention to others
        scores = (q @ k.transpose(-2, -1)) / (self.d_k ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)  # Turn into probabilities

        # Use those attention scores to mix the values
        out = attn @ v

        # Put the heads back together into one vector
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.fc(out)

# ------------------------
# Feed Forward Network
# ------------------------
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        # A simple 2-layer mini neural network
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.net(x)

# ------------------------
# Transformer Block
# ------------------------
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        # Attention layer + feed forward layer
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff)

        # Normalization to keep values stable
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        # First, do attention (words look at each other)
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x), mask)
        # Then, do feed forward (extra processing)
        x = x + self.ff(self.ln2(x))
        return x

# ------------------------
# GPT-like Model
# ------------------------
class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model=256, num_heads=8, d_ff=512, num_layers=4, max_len=512):
        super().__init__()
        # Turns words into vectors and adds positions
        self.embed = Embeddings(vocab_size, d_model, max_len)

        # Several transformer blocks stacked on top of each other
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff) for _ in range(num_layers)
        ])

        # Final normalization and output to word predictions
        self.ln = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x, mask=None):
        # Convert words to embeddings
        x = self.embed(x)
        # Process through each transformer block
        for block in self.blocks:
            x = block(x, mask)
        # Normalize and predict the next word
        x = self.ln(x)
        return self.fc_out(x)

# ------------------------
# Example Usage
# ------------------------

# Let's say we have a dictionary of 10,000 words
vocab_size = 10000
model = MiniGPT(vocab_size)

# Example input: 2 sentences, each with 10 words (random for demo)
x = torch.randint(0, vocab_size, (2, 10))

# Forward pass: get predictions for the next word at each position
logits = model(x)
print("Logits shape:", logits.shape)  # (batch, seq_len, vocab_size)
