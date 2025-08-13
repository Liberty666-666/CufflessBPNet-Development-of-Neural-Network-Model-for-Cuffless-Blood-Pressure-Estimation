import torch
import torch.nn as nn
from layers.Mamba_EncDec import Encoder, EncoderLayer
from layers.Embed import DataEmbedding_inverted
from mamba_ssm import Mamba

class Model(nn.Module):  # Renamed to Model for experiment compatibility
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.use_norm = configs.use_norm

        # Embedding
        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len, configs.d_model, configs.embed,
            configs.freq, configs.dropout
        )

        # Encoder
        self.encoder = Encoder([
            EncoderLayer(
                Mamba(
                    d_model=configs.d_model,
                    d_state=configs.d_state,
                    d_conv=2,
                    expand=1
                ),
                Mamba(
                    d_model=configs.d_model,
                    d_state=configs.d_state,
                    d_conv=2,
                    expand=1
                ),
                configs.d_model,
                configs.d_ff,
                dropout=configs.dropout,
                activation=configs.activation
            ) for _ in range(configs.e_layers)
        ], norm_layer=nn.LayerNorm(configs.d_model))

        # Regression head: output SBP & DBP
        self.regression_head = nn.Sequential(
            nn.LayerNorm(configs.d_model),
            nn.Linear(configs.d_model, 2)  # Output shape: [B, 2]
        )

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        # Embedding: B L 1 -> B L E
        enc_out = self.enc_embedding(x_enc, x_mark_enc)

        # Encoder
        enc_out, _ = self.encoder(enc_out, attn_mask=None)

        # Pooling: Mean over token dimension
        pooled = enc_out.mean(dim=1)  # [B, d_model]

        # Regression prediction
        output = self.regression_head(pooled)  # [B, 2]

        return output
