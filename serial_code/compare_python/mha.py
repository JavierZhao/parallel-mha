import torch
import torch.nn as nn
import numpy as np

# Set random seed
torch.manual_seed(0)


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
x = torch.randn(batch_size, embed_dim, dtype=torch.float64)  # Ensure input is float64

# Initialize the Multi-Head Attention layer
mha = MultiHeadAttention(embed_dim, num_heads)

# Convert weights to float64
with torch.no_grad():
    mha.mha.in_proj_weight = nn.Parameter(mha.mha.in_proj_weight.to(torch.float64))
    mha.mha.in_proj_bias = nn.Parameter(mha.mha.in_proj_bias.to(torch.float64))
    mha.mha.out_proj.weight = nn.Parameter(mha.mha.out_proj.weight.to(torch.float64))
    mha.mha.out_proj.bias = nn.Parameter(mha.mha.out_proj.bias.to(torch.float64))

# Print weight data types to verify
print("in_proj_weight dtype:", mha.mha.in_proj_weight.dtype)
print("in_proj_bias dtype:", mha.mha.in_proj_bias.dtype)
print("out_proj_weight dtype:", mha.mha.out_proj.weight.dtype)
print("out_proj_bias dtype:", mha.mha.out_proj.bias.dtype)

# Run MHA
output, _ = mha(x)

# Extract initial weights and biases
Wqkv = mha.mha.in_proj_weight.detach().numpy()  # Ensure weights are float64
Wo = mha.mha.out_proj.weight.detach().numpy()  # Ensure weights are float64
bqkv = mha.mha.in_proj_bias.detach().numpy()  # Ensure biases are float64
bo = mha.mha.out_proj.bias.detach().numpy()  # Ensure biases are float64

# Split Wqkv into Wq, Wk, Wv
Wq = Wqkv[:embed_dim, :]
Wk = Wqkv[embed_dim : 2 * embed_dim, :]
Wv = Wqkv[2 * embed_dim :, :]

# Split bqkv into bq, bk, bv
bq = bqkv[:embed_dim]
bk = bqkv[embed_dim : 2 * embed_dim]
bv = bqkv[2 * embed_dim :]

# Save the weights, biases, input matrix x, and output to files
np.savez("mha_weights.npz", Wq=Wq, Wk=Wk, Wv=Wv, Wo=Wo, bq=bq, bk=bk, bv=bv, bo=bo)
np.save("mha_input.npy", x.detach().numpy())
np.save("mha_output.npy", output.detach().numpy())

# Print data types to verify
print("Wq dtype:", Wq.dtype)
print("Wk dtype:", Wk.dtype)
print("Wv dtype:", Wv.dtype)
print("Wo dtype:", Wo.dtype)
print("bq dtype:", bq.dtype)
print("bk dtype:", bk.dtype)
print("bv dtype:", bv.dtype)
print("bo dtype:", bo.dtype)
print("x dtype:", x.detach().numpy().dtype)
print("output dtype:", output.detach().numpy().dtype)

print("Saved weights, biases, input, and output.")
