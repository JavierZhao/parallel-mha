import torch
import torch.nn as nn
import numpy as np

# Define the parameters
embed_dim = 128
num_heads = 4
batch_size = 2

# Initialize random input
x = torch.randn(batch_size, embed_dim)

# Initialize the Multi-Head Attention layer
mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

# Run MHA
output, _ = mha(x, x, x)

# Extract initial weights
Wqkv = mha.in_proj_weight.detach().numpy()
Wo = mha.out_proj.weight.detach().numpy()

# Split Wqkv into Wq, Wk, Wv
Wq = Wqkv[:embed_dim, :]
Wk = Wqkv[embed_dim : 2 * embed_dim, :]
Wv = Wqkv[2 * embed_dim :, :]

# Save the weights, input matrix x, and output to files
np.savez("mha_weights.npz", Wq=Wq, Wk=Wk, Wv=Wv, Wo=Wo)
np.save("mha_input.npy", x.detach().numpy())
np.save("mha_output.npy", output.detach().numpy())

print("Saved weights, input, and output.")
