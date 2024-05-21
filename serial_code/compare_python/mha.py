import torch
import torch.nn as nn
import numpy as np
import csv

# Set random seed
torch.manual_seed(0)


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True
        )

    def forward(self, x):
        output, attn = self.mha(x, x, x, average_attn_weights=False)
        return output, attn


# Parameters
embed_dim = 4
num_heads = 2
batch_size = 1
seq_length = 2

# Initialize random input
x = torch.randn(batch_size, seq_length, embed_dim, dtype=torch.float64)  # Ensure input is float64
print("Input x shape:", x.shape)

# Initialize the Multi-Head Attention layer
mha = MultiHeadAttention(embed_dim, num_heads)

# Convert weights to float64
with torch.no_grad():
    mha.mha.in_proj_weight = nn.Parameter(mha.mha.in_proj_weight.to(torch.float64))
    mha.mha.in_proj_bias = nn.Parameter(mha.mha.in_proj_bias.to(torch.float64))
    mha.mha.out_proj.weight = nn.Parameter(mha.mha.out_proj.weight.to(torch.float64))
    mha.mha.out_proj.bias = nn.Parameter(mha.mha.out_proj.bias.to(torch.float64))

print("in_proj_weight dtype:", mha.mha.in_proj_weight.dtype)
print("in_proj_bias dtype:", mha.mha.in_proj_bias.dtype)
print("out_proj_weight dtype:", mha.mha.out_proj.weight.dtype)
print("out_proj_bias dtype:", mha.mha.out_proj.bias.dtype)

# Run MHA
output, attn = mha(x)
print("Output shape:", output.shape)
print("attn", attn)

# Extract weights and biases
Wqkv = mha.mha.in_proj_weight.detach().numpy()
Wo = mha.mha.out_proj.weight.detach().numpy()
bqkv = mha.mha.in_proj_bias.detach().numpy()
bo = mha.mha.out_proj.bias.detach().numpy()

# Split Wqkv into Wq, Wk, Wv
Wq = Wqkv[:embed_dim, :]
Wk = Wqkv[embed_dim : 2 * embed_dim, :]
Wv = Wqkv[2 * embed_dim :, :]

# Split bqkv into bq, bk, bv
bq = bqkv[:embed_dim]
bk = bqkv[embed_dim : 2 * embed_dim]
bv = bqkv[2 * embed_dim :]


# Save the weights, biases, input matrix x, and output to CSV files
def save_array_to_csv(data, filename):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        for row in data:
            writer.writerow(row)


save_array_to_csv(Wq, "Wq.csv")
save_array_to_csv(Wk, "Wk.csv")
save_array_to_csv(Wv, "Wv.csv")
save_array_to_csv(Wo, "Wo.csv")
save_array_to_csv(bq[:, None].T, "bq.csv")  # Transpose to make it a single row
save_array_to_csv(bk[:, None].T, "bk.csv")
save_array_to_csv(bv[:, None].T, "bv.csv")
save_array_to_csv(bo[:, None].T, "bo.csv")
save_array_to_csv(x.detach().numpy().squeeze(), "mha_input.csv")
save_array_to_csv(output.detach().numpy().squeeze(), "mha_output.csv")

print("Saved weights, biases, input, and output.")
