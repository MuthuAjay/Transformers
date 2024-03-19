import torch
import torch.nn as nn
import math


class InputEmbedding(nn.Module):
    """Embedding layer for input tokens.

    Args:
        d_model (int): The dimension of the model.
        vocab_size (int): Size of the vocabulary.
    """

    def __init__(self,
                 d_model: int,
                 vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self,
                x: torch.Tensor):
        """Perform forward pass through the embedding layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Embedded tensor of shape (batch_size, seq_len, d_model).
        """
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEmbedding(nn.Module):
    """Positional encoding to provide positional information to the input tokens.

    Args:
        d_model (int): The dimension of the model.
        seq_len (int): Length of the input sequence.
        dropout (int|float|any): Dropout probability.
    """

    def __init__(self,
                 d_model: int,
                 seq_len: int,
                 dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()) * (-math.log(10000.0) / d_model)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.pe = pe.unsqueeze(0)

        self.register_buffer('pe', self.pe)

    def forward(self,
                x: torch.Tensor):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)


class LayerNormalization(nn.Module):
    """Layer normalization module.

    Args:
        eps (float): Epsilon value for numerical stability.
    """

    def __init__(self,
                 eps: float = 10e-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(data=torch.ones(1))
        self.bias = nn.Parameter(data=torch.zeros(1))

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):
    """Feed-forward block with dropout and ReLU activation.

    Args:
        d_model (int): The dimension of the model.
        d_ff (int): The dimension of the feed-forward layer.
        dropout (int|float|any): Dropout probability.
    """

    def __init__(self,
                 d_model: int,
                 d_ff: int,
                 dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(in_features=d_model, out_features=d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(in_features=d_ff, out_features=d_model)

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


class MultiHeadSelfAttention(nn.Module):

    def __init__(self,
                 d_model: int,
                 h: int,
                 dropout: float
                 ):
        super().__init__()
        self.attention_scores = None
        self.d_model = d_model
        self.h = h

        assert d_model % h == 0, "d_model is not divisible by number of heads"

        self.d_k = d_model // h
        self.w_q = nn.Linear(in_features=d_model,
                             out_features=d_model)
        self.w_k = nn.Linear(in_features=d_model,
                             out_features=d_model)
        self.w_v = nn.Linear(in_features=d_model,
                             out_features=d_model)
        self.w_o = nn.Linear(in_features=d_model,
                             out_features=d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        attention_scores = (query @ key.transpose(-2. - 1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=1)  # (Batch, h, seq_len, seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q)  # (Batch , seq_len, d_model) -> (Batch , seq_len, d_model)
        key = self.w_k(k)  # (Batch , seq_len, d_model) -> (Batch , seq_len, d_model)
        value = self.w_v(v)  # (Batch , seq_len, d_model) -> (Batch , seq_len, d_model)

        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0, value.shape[1], self.h, self.d_k]).transpose(1, 2)

        x, self.attention_scores = MultiHeadSelfAttention.attention(query=query,
                                                                    key=key,
                                                                    value=value,
                                                                    mask=mask,
                                                                    dropout=self.dropout)

        # (Batch, h, Seq_len, d_k) --> (Batch, Seq_len, h, d_k) --> (Batch, Seq_Len, d_model)

        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        return self.w_o(x)



