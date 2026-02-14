import torch
import torch.nn as nn
import torch.nn.functional as F

class BahdanauAttention(nn.Module):
    """
    Bahdanau (additive) attention.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_s = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, encoder_outputs, mask=None):
        """
        encoder_outputs: (batch, seq_len, hidden)
        mask: (batch, seq_len) with 1 for valid tokens
        """
        # (batch, seq_len, hidden)
        score = torch.tanh(self.W_h(encoder_outputs) + self.W_s(encoder_outputs))
        # (batch, seq_len)
        score = self.v(score).squeeze(-1)

        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(score, dim=1)
        # (batch, hidden)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)

        return context, attn_weights
