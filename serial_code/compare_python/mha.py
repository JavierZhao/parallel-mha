import torch
import torch.nn as nn
import numpy as np


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True
        )

    def forward(self, x):
        output, _ = self.mha(x, x, x)
        return output


# Parameters
embed_dim = 128
num_heads = 4
batch_size = 2

# Initialize random input
x = torch.randn(batch_size, embed_dim, dtype=torch.float32)  # Ensure input is float32

# Initialize the Multi-Head Attention layer
mha = MultiHeadAttention(embed_dim, num_heads)

# Run MHA
output, _ = mha(x)

# Extract initial weights
Wqkv = (
    mha.mha.in_proj_weight.detach().numpy().astype(np.float32)
)  # Ensure weights are float32
Wo = (
    mha.mha.out_proj.weight.detach().numpy().astype(np.float32)
)  # Ensure weights are float32

# Split Wqkv into Wq, Wk, Wv
Wq = Wqkv[:embed_dim, :]
Wk = Wqkv[embed_dim : 2 * embed_dim, :]
Wv = Wqkv[2 * embed_dim :, :]

# Save the weights, input matrix x, and output to files
np.savez("mha_weights.npz", Wq=Wq, Wk=Wk, Wv=Wv, Wo=Wo)
np.save("mha_input.npy", x.detach().numpy())
np.save("mha_output.npy", output.detach().numpy())

print("Saved weights, input, and output.")

print("Wq dtype:", Wq.dtype)
print("Wk dtype:", Wk.dtype)
print("Wv dtype:", Wv.dtype)
print("Wo dtype:", Wo.dtype)
print("x dtype:", x.detach().numpy().dtype)
print("output dtype:", output.detach().numpy().dtype)
