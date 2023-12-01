import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Assume you have a DataLoader for your dataset, a GPT2 model, and a loss function

# Create sample data (replace this with your actual data loading logic)
data = torch.randint(0, vocab_size, (100, 10))  # 100 sequences of length 10
loader = DataLoader(TensorDataset(data), batch_size=8, shuffle=True)

# Model, Loss, and Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPT2(vocab_size, embed_size, num_layers, heads, forward_expansion).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for batch in loader:
        inputs = batch[0].to(device)
        targets = inputs.clone().detach()  # Just for illustration, replace with your actual targets

        outputs = model(inputs)
        loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(loader)}")

import torch.distributed as dist
from fsdp import FullyShardedDataParallel

# Assume you have a DataLoader for your dataset, a GPT2 model, and a loss function

# Sample data (replace this with your actual data loading logic)
vocab_size = 10000
data = torch.randint(0, vocab_size, (100, 10))  # 100 sequences of length 10
loader = DataLoader(TensorDataset(data), batch_size=8, shuffle=True)

# Model, Loss, and Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPT2(vocab_size, embed_size, num_layers, heads, forward_expansion).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# DDP initialization
if torch.cuda.is_available():
    torch.distributed.init_process_group(backend='nccl')
    model = torch.nn.parallel.DistributedDataParallel(model)

# FSDP initialization
fsdp_config = {
    "mixed_precision": False,  # Set to True for mixed precision training
    "flatten_parameters": True,
    "gradient_as_bucket_view": True
}
model_fsdp = FullyShardedDataParallel(model, **fsdp_config)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for batch in loader:
        inputs = batch[0].to(device)
        targets = inputs.clone().detach()  # Just for illustration, replace with your actual targets

        outputs = model(inputs)
        loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(loader)}")

# Cleanup for DDP
if torch.cuda.is_available():
    torch.distributed.destroy_process_group()
