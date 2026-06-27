#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class DemographicLSTM(nn.Module):
    """LSTM-based model for demographic prediction from audio embeddings."""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_layers,
        num_classes,
        dropout=0.2,
        use_layer_norm=True,
        use_residual=True,
        pooling="attn",
        bidirectional=True,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )
        self.bidirectional = bidirectional
        self.hidden_multiplier = 2 if bidirectional else 1
        self.pooling = pooling
        self.use_residual = use_residual

        if use_residual:
            # bias=False keeps padded zero frames mapped to zero.
            self.residual_proj = nn.Linear(input_dim, hidden_dim * self.hidden_multiplier, bias=False)
        else:
            self.residual_proj = None

        if pooling == "attn":
            self.attn_proj = nn.Linear(hidden_dim * self.hidden_multiplier, hidden_dim * self.hidden_multiplier)
            self.attn_score = nn.Linear(hidden_dim * self.hidden_multiplier, 1, bias=False)
        else:
            self.attn_proj = None
            self.attn_score = None

        self.layer_norm = nn.LayerNorm(hidden_dim * self.hidden_multiplier) if use_layer_norm else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * self.hidden_multiplier, num_classes)

    def forward(self, x, lengths=None):
        """
        Args:
            x: (batch_size, seq_len, input_dim)
            lengths: (batch_size,) valid sequence lengths before padding
        Returns:
            logits: (batch_size, num_classes)
        """
        if lengths is not None:
            # Clamp for safety and run packed LSTM to skip padded timesteps.
            lengths = lengths.to(x.device).clamp(min=1, max=x.size(1))
            packed = pack_padded_sequence(
                x, lengths=lengths.detach().to("cpu"), batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.lstm(packed)
            lstm_out, _ = pad_packed_sequence(
                packed_out, batch_first=True, total_length=x.size(1)
            )
        else:
            lstm_out, _ = self.lstm(x)
        if self.use_residual:
            lstm_out = lstm_out + self.residual_proj(x)

        if self.pooling == "last":
            if lengths is None:
                pooled = lstm_out[:, -1, :]
            else:
                lengths = lengths.to(x.device)
                last_indices = (lengths - 1).clamp(min=0)
                batch_indices = torch.arange(lstm_out.size(0), device=x.device)
                pooled = lstm_out[batch_indices, last_indices, :]
        elif self.pooling == "mean":
            if lengths is None:
                pooled = lstm_out.mean(dim=1)
            else:
                lengths = lengths.to(x.device)
                max_len = lstm_out.size(1)
                mask = torch.arange(max_len, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
                mask = mask.unsqueeze(-1).to(lstm_out.dtype)
                summed = (lstm_out * mask).sum(dim=1)
                denom = lengths.clamp(min=1).unsqueeze(1).to(lstm_out.dtype)
                pooled = summed / denom
        else:  # attn
            if lengths is None:
                mask = None
            else:
                lengths = lengths.to(x.device)
                max_len = lstm_out.size(1)
                mask = torch.arange(max_len, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
            h = torch.tanh(self.attn_proj(lstm_out))
            scores = self.attn_score(h).squeeze(-1)
            if mask is not None:
                scores = scores.masked_fill(~mask, float("-inf"))
            weights = F.softmax(scores, dim=1).unsqueeze(-1)
            pooled = (weights * lstm_out).sum(dim=1)

        pooled = self.layer_norm(pooled)
        return self.fc(self.dropout(pooled))
