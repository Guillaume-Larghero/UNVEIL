"""A small gated-attention ABMIL with the attributes the default signal adapter reads
(deep_attention.theta1/theta2/W/temperature, mlp ending in a sigmoid, input_dim)."""

import torch
from torch import nn


class DeepAttention(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.theta1 = nn.Linear(d, h)
        self.theta2 = nn.Linear(d, h)
        self.W = nn.Linear(h, 1)
        self.temperature = 1.0


class GatedABMIL(nn.Module):
    def __init__(self, input_dim, hidden=16):
        super().__init__()
        self.input_dim = input_dim
        self.deep_attention = DeepAttention(input_dim, hidden)
        self.mlp = nn.Sequential(nn.Linear(input_dim, 1), nn.Sigmoid())

    def forward(self, x):
        da = self.deep_attention
        a = torch.softmax(da.W(torch.tanh(da.theta1(x)) * torch.sigmoid(da.theta2(x)))
                          .squeeze(-1) / da.temperature, dim=0)
        return self.mlp((a[:, None] * x).sum(0, keepdim=True)).reshape(())
