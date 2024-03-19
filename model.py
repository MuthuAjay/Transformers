import math
import torch
from torch import nn


class InputEmbedding(nn.Module):

    def __init__(self,
                 d_model: int,
                 vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size  # size of the vocabulary
        self.d_model = d_model  # dimension of the model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self,
                x: torch.Tensor):
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEmbedding(nn.Module):

    def __init__(self,
                 d_model: int,
                 seq_len: int,
                 dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)

        # create a vector of shape (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()) * (-math.log(10000.0) / d_model)

        # Apply the sin fn to EVEN positions
        pe[:, 0::2] = torch.sin(position * div_term)

        # Apply the cosine fn to the ODD positions
        pe[:, 1::2] = torch.cos(position * div_term)

        self.pe = pe.unsqueeze(0)  # (1, seq_len , d_model)

        self.register_buffer('pe', self.pe)

    def forward(self,
                x: torch.Tensor):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)


class LayerNormalization(nn.Module):

    def __init__(self,
                 eps: float = 10e-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(data=torch.ones(1))  # Multiplied
        self.bias = nn.Parameter(data=torch.zeros(1))  # Added

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1,
                      keepdim=True)
        std = x.std(dim=-1,
                    keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):

    def __init__(self,
                 d_model: int,
                 d_ff: int,
                 dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(in_features=d_model,
                                  out_features=d_ff)  # W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(in_features=d_ff,
                                  out_features=d_model)

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        # ( Batch ,Seq_len, d_model) --> (Batch, seq_len, d_ff) --> (Batch, seq_len, d_model)

        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

