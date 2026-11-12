import math
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def attention(self, Q, K, V):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, V), weights

    def forward(self, x):
        batch, seq_len, _ = x.shape

        # Proyecciones lineales y division en cabezas
        Q = self.W_q(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Atencion escalada y concatenacion
        out, weights = self.attention(Q, K, V)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)

        return self.W_o(out), weights


if __name__ == "__main__":
    x = torch.randn(4, 12, 64)  # (batch, seq_len, d_model)
    mha = MultiHeadAttention(d_model=64, num_heads=8)
    salida, pesos = mha(x)

    print("Entrada:", x.shape)
    print("Salida:", salida.shape)
    print("Pesos de atencion:", pesos.shape)
