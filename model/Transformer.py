import torch
import torch.nn as nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding


class Model(nn.Module):
    """
    Transformer for BP Regression
    Predicts SBP & DBP directly from PPG segments
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.use_norm = configs.use_norm

        # Embedding layer
        self.enc_embedding = DataEmbedding(
            configs.enc_in, configs.d_model, configs.embed,
            configs.freq, configs.dropout
        )

        # Encoder (no decoder needed for regression)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model)
        )

        # Regression head: output SBP & DBP
        self.regression_head = nn.Sequential(
            nn.LayerNorm(configs.d_model),
            nn.Linear(configs.d_model, 2)  # Output [B, 2]
        )

    def forward(self, x_enc, x_mark_enc=None, *args, **kwargs):
        # Optional normalization (per sample)
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        # Embedding: [B, L, 1] → [B, L, d_model]
        enc_out = self.enc_embedding(x_enc, x_mark_enc)

        # Encoder: [B, L, d_model] → [B, L, d_model]
        enc_out, _ = self.encoder(enc_out, attn_mask=None)

        # Pooling: mean over sequence length → [B, d_model]
        pooled = enc_out.mean(dim=1)

        # Regression output: [B, 2]
        output = self.regression_head(pooled)

        return output
