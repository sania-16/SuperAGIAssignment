# SuperAGIAssignment

Q/A Part - in Q:A_Assignment.pdf

Task 1 | GPT-2 Model & Checkpoints - part1.py

### Model Implementation

The GPT-2 model has been implemented following the specifications from the GPT-2 paper. The model consists of transformer blocks with multi-head self-attention mechanisms, feed-forward networks, and positional embeddings.

The implementation includes the following components:

- GPT2Block
- GPT2
- Rotary Positional Embedding
- Group Query Attention
- Sliding Window Attention

### Checkpoints

Pretrained GPT-2 125M model checkpoints have been loaded successfully. The `load_gpt2_weights` function is for loading the weights into the GPT-2 model.

Task 2 | Transformer Architectural Changes - part2.py

- The RotaryPositionalEmbedding class is used to replace the original positional embeddings.
- The Group Query Attention mechanism has been implemented following the insights from the GQA paper. The GroupQueryAttention class is integrated into the GPT2Block.
- The Sliding Window Attention mechanism is successfully integrated into the GPT-2 model. The SlidingWindowAttention class is used in the GPT2Block.

Task 3 | Training Loop Implementation - part3.py
Install all required libraries by following commands:
- pip install torch torchtext torchvision torchgpipe
- pip install fsdp


###
The modifications to the GPT-2 model and the implementation of distributed training contribute to improved performance. Experimentation with different architectural changes and distributed training options allows for a comprehensive understanding of the model's capabilities and efficiency.






