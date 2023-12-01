import torch
import torch.nn as nn
import torch.nn.functional as F

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, embed_size, max_len=512):
        super(RotaryPositionalEmbedding, self).__init__()
        self.embed_size = embed_size
        self.max_len = max_len
        self.alpha = nn.Parameter(torch.zeros(embed_size // 2))
        self.beta = nn.Parameter(torch.zeros(embed_size // 2))

    def forward(self, x):
        positions = torch.arange(0, x.size(1)).float()
        angle = self.alpha.unsqueeze(0) * positions.unsqueeze(1) + self.beta.unsqueeze(0)
        angle_radians = angle * (3.141592653589793 / 180.0)
        sinusoids = torch.cat([torch.sin(angle_radians), torch.cos(angle_radians)], dim=-1)
        return x + sinusoids.unsqueeze(0)

# Modifying the GPT2Block to use RotaryPositionalEmbedding
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
        self.rotary_positional_embedding = RotaryPositionalEmbedding(embed_size)

    def forward(self, x):
        x = self.rotary_positional_embedding(x)
        attn_output, _ = self.multihead_attn(x, x, x)
        x = x + attn_output
        x = self.layer_norm1(x)

        ff_output = self.feed_forward(x)
        x = x + ff_output
        x = self.layer_norm2(x)
        return x

        class GroupQueryAttention(nn.Module):
            def __init__(self, embed_size, num_heads):
                super(GroupQueryAttention, self).__init__()
                self.multihead_attn = nn.MultiheadAttention(embed_size, num_heads)

            def forward(self, x):
                # Assume x has shape (seq_len, batch_size, embed_size)
                x = x.permute(1, 0, 2)  # Change to (batch_size, seq_len, embed_size) for Multihead Attention
                attn_output, _ = self.multihead_attn(x, x, x)
                attn_output = attn_output.permute(1, 0, 2)  # Change back to (seq_len, batch_size, embed_size)
                return x + attn_output

        class SlidingWindowAttention(nn.Module):
            def __init__(self, embed_size, num_heads, window_size):
                super(SlidingWindowAttention, self).__init__()
                self.window_size = window_size
                self.num_heads = num_heads
                self.attention = nn.MultiheadAttention(embed_size, num_heads)

            def forward(self, x):
                seq_len, batch_size, embed_size = x.size()

                # Extract overlapping windows
                windows = x.unfold(0, self.window_size, self.window_size - 1)

                # Reshape windows for attention
                windows = windows.permute(2, 0, 1, 3)

                # Flatten the windows
                windows = windows.view(self.num_heads, -1, self.window_size * embed_size)

                # Apply attention separately to each window
                attn_output, _ = self.attention(windows, windows, windows)

                # Reshape the output
                attn_output = attn_output.view(self.num_heads, -1, self.window_size, embed_size)

                # Combine overlapping windows
                attn_output = attn_output.permute(1, 2, 0, 3).contiguous().view(seq_len, batch_size, -1)

                return x + attn_output

            class GPT2Block(nn.Module):
              def __init__(self, embed_size, heads, forward_expansion):
                super(GPT2Block, self).__init__()
                self.group_query_attention = GroupQueryAttention(embed_size, heads)
                self.sliding_window_attention = SlidingWindowAttention(embed_size, heads, window_size=5)
                self.feed_forward = nn.Sequential(
                    nn.Linear(embed_size, forward_expansion * embed_size),
                    nn.ReLU(),
                    nn.Linear(forward_expansion * embed_size, embed_size)
                )
                self.layer_norm1 = nn.LayerNorm(embed_size)
                self.layer_norm2 = nn.LayerNorm(embed_size)

            def forward(self, x):
                x = self.group_query_attention(x)
                x = self.sliding_window_attention(x)

                attn_output, _ = self.multihead_attn(x, x, x)
                x = x + attn_output
                x = self.layer_norm1(x)

                ff_output = self.feed_forward(x)
                x = x + ff_output
                x = self.layer_norm2(x)
                return x

        
