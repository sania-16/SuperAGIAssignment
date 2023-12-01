import torch
import torch.nn as nn
import torch.nn.functional as F

class GPT2Block(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion):
        super(GPT2Block, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_size, heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        self.layer_norm1 = nn.LayerNorm(embed_size)
        self.layer_norm2 = nn.LayerNorm(embed_size)

    def forward(self, x):
        attn_output, _ = self.multihead_attn(x, x, x)
        x = x + attn_output
        x = self.layer_norm1(x)

        ff_output = self.feed_forward(x)
        x = x + ff_output
        x = self.layer_norm2(x)
        return x

class GPT2(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, heads, forward_expansion):
        super(GPT2, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.positional_encoding = self.get_positional_encoding(max_len=512, embed_size=embed_size)
        self.layers = nn.ModuleList([
            GPT2Block(embed_size, heads, forward_expansion) for _ in range(num_layers)
        ])
        self.fc = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        seq_len, batch_size = x.shape
        x = self.embedding(x)
        x = x + self.positional_encoding[:seq_len, :].unsqueeze(1).repeat(1, batch_size, 1)

        for layer in self.layers:
            x = layer(x)

        x = self.fc(x)
        return x

    def get_positional_encoding(self, max_len, embed_size):
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * -(torch.log(torch.tensor(10000.0)) / embed_size))
        pos_enc = torch.zeros((max_len, embed_size))
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        pos_enc = pos_enc.unsqueeze(0)
        return pos_enc

# Example usage
vocab_size = 10000  # Replace with the actual vocabulary size
embed_size = 768
num_layers = 12
heads = 12
forward_expansion = 4

model = GPT2(vocab_size, embed_size, num_layers, heads, forward_expansion)

# Load the GPT-2 125M model checkpoints
model_path = 'test.pt'          # enter checkpoints here
model = GPT2.load_state_dict(torch.load(model_path))

# Sample prediction
input_text = "The quick brown fox jumps over the lazy dog"
input_ids = tokenizer(input_text)['input_ids']

with torch.no_grad():
    output_logits = model(input_ids)
    output_probs = F.softmax(output_logits, dim=-1)
    output_tokens = output_probs.argmax(dim=-1)
    generated_text = tokenizer.decode(output_tokens)

print(generated_text)
