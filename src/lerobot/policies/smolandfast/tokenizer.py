import torch
import torch.nn as nn
from operator import mul
from functools import reduce

from lerobot.policies.smolandfast.vq import ResidualVectorQuantizer


# This is the core building block. Its most important feature is that the output shape
# is identical to the input shape, which makes the residual connection `x + block(x)` possible.
class ResBlock(nn.Module):
    """
    A simplified ResNet block inspired by SEANet.
    It uses a bottleneck design (dim -> dim/2 -> dim) to process the features.
    """
    def __init__(self, dim: int, kernel_size: int = 3):
        super().__init__()
        # The bottleneck halves the number of channels
        hidden_dim = dim // 2
        
        self.block = nn.Sequential(
            nn.ELU(),
            # First convolution: from dim to the hidden bottleneck dimension
            nn.Conv1d(dim, hidden_dim, kernel_size=kernel_size, padding=(kernel_size - 1) // 2),
            nn.ELU(),
            # Second convolution: from the bottleneck back to the original dimension
            nn.Conv1d(hidden_dim, dim, kernel_size=1)
        )
        # The shortcut connection simply passes the input through.
        self.shortcut = nn.Identity()

    def forward(self, x):
        return self.shortcut(x) + self.block(x)


# This uses the ResBlocks and separate downsampling convolution layers.
class Encoder(nn.Module):
    def __init__(self,
                 input_dims: int = 2,
                 base_features: int = 16,
                 ratios: list[int] = [1],
                 num_residual_layers: int = 1,
                 num_lstm_layers: int = 2):
        super().__init__()

        # Initial convolution to increase feature channels from 2 (x,y) to base_features
        self.conv_in = nn.Conv1d(input_dims, base_features, kernel_size=3, padding=1)

        conv_body = []
        mult = 1
        for i, ratio in enumerate(ratios):
            # Add residual blocks. These do NOT change the shape.
            for j in range(num_residual_layers):
                conv_body.append(ResBlock(dim=mult * base_features))

            # Add the downsampling layer. This HALVES sequence length and DOUBLES channels.
            conv_body.append(nn.ELU())
            conv_body.append(nn.Conv1d(
                in_channels=mult * base_features,
                out_channels=mult * base_features * 2,
                kernel_size=3,
                stride=ratio,
                padding=1,
            ))
            mult *= 2

        self.conv_body = nn.Sequential(*conv_body)

        # LSTM to process the sequence of features from the convolutions
        final_conv_channels = mult * base_features
        self.lstm = nn.LSTM(
            input_size=final_conv_channels,
            hidden_size=final_conv_channels,
            num_layers=num_lstm_layers,
            batch_first=True
        )
        
        out_layer = [
            nn.ELU(),
            nn.Linear(final_conv_channels, final_conv_channels, bias=False)
        ]
        self.out_layer = nn.Sequential(*out_layer)
        self.out_dim = final_conv_channels


    def forward(self, x):
        # Input: (B, 12, 2)
        x = x.permute(0, 2, 1)  # (B, 2, 12)
        x = self.conv_in(x)      # (B, 16, 12)
        x = self.conv_body(x)    # (B, 64, 3) after two downsampling steps (12 -> 6 -> 3)

        

        # _, (h_n, _) = self.lstm(x)
        # x = h_n[-1]  # Shape: (B, D_hidden), e.g., (B, 32)
        # x = x.unsqueeze(-1) # (B, 32, 1)
        
        x = x.permute(0, 2, 1)  # (B, T, C) for LSTM
        y, _ = self.lstm(x)
        x = y + x

        # Output layer
        x = self.out_layer(x)

        x = x.permute(0, 2, 1)  # (B, C, T) after LSTM

        return x


# --- 3. The Decoder ---
# This symmetrically reverses every operation from the encoder.
class Decoder(nn.Module):
    def __init__(self,
                 output_dims: int = 2,
                 base_features: int = 16,
                 ratios: list[int] = [1],
                 num_residual_layers: int = 1,
                 num_lstm_layers: int = 2):
        super().__init__()
        
        self.num_lstm_layers = num_lstm_layers
        
        # Calculate channel sizes (must mirror the encoder)
        mult = 2 ** len(ratios) # Starts at 4 (1 -> 2 -> 4)
        final_conv_channels = mult * base_features
        
        in_layer = [
            nn.Linear(final_conv_channels, final_conv_channels, bias=False),
            nn.ELU()
        ]
        self.in_layer = nn.Sequential(*in_layer)

        self.lstm = nn.LSTM(
            input_size=final_conv_channels,
            hidden_size=final_conv_channels,
            num_layers=num_lstm_layers,
            batch_first=True
        )

        deconv_body = []
        for i, ratio in enumerate(reversed(ratios)):
            # Add the upsampling layer. This DOUBLES sequence length and HALVES channels.
            in_channels = mult * base_features
            out_channels = mult * base_features // 2
            deconv_body.append(nn.ELU())
            deconv_body.append(nn.ConvTranspose1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=ratio,
                padding=1,
                output_padding=ratio - 1,
            ))

            # Add residual blocks. These operate on the new, smaller channel dimension.
            for j in range(num_residual_layers):
                deconv_body.append(ResBlock(dim=out_channels))

            mult //= 2

        self.deconv_body = nn.Sequential(*deconv_body)
        self.lstm_input_size = final_conv_channels

        # Reverse the initial convolution
        self.conv_out = nn.Conv1d(base_features, output_dims, kernel_size=3, padding=1)

    def forward(self, x):
        # x: [B, C, T]
        # x = x.squeeze(-1)

        x = x.permute(0, 2, 1)  # (B, T, C) for LSTM
        x = self.in_layer(x)
        
        # h_0 = x.unsqueeze(0).repeat(self.num_lstm_layers, 1, 1)
        # c_0 = torch.zeros_like(h_0) # Initial cell state
        
        # batch_size = x.shape[0]
        # dummy_input = torch.zeros((batch_size, target_seq_len, self.lstm_input_size), 
        #                           device=x.device)

        # x, _ = self.lstm(dummy_input, (h_0, c_0))

        y, _ = self.lstm(x)
        x = y + x
        x = x.permute(0, 2, 1)  # (B, C, T) after LSTM

        x = self.deconv_body(x) # (B, 16, 10)
        x = self.conv_out(x)    # (B, 2, 10)
        decoded = x.permute(0, 2, 1) # (B, 10, 2)

        return decoded

# --- 4. Full Autoencoder and Training Loop ---
class Autoencoder(nn.Module):
    def __init__(self, vocab_size: int, encoded_dim: int, **kwargs):
        super().__init__()
        self.seq_len_divider = reduce(mul, kwargs.get("ratios", [1]))
        self.vocab_size = vocab_size
        self.encoder = Encoder(**kwargs)
        self.decoder = Decoder(**kwargs)
        self.quantizer = ResidualVectorQuantizer(
            dimension=self.encoder.out_dim,
            n_q=encoded_dim,
            bins=vocab_size,
        )

    def forward(self, x):
        original_seq_len = x.shape[1]
        continuous_codes = self.encoder(x)
        quantized, codes, commit_loss = self.quantizer(continuous_codes)
        # codes: [n_q, B, T]
        # quantized: [B, C, T]
        reconstructed = self.decoder(quantized)
        codes = codes.permute(1, 2, 0) # Shape: (B, T, n_q)
        return reconstructed, codes, commit_loss

    def encode(self, x):
        x = self.encoder(x)
        codes = self.quantizer.encode(x)
        codes = codes.permute(1, 2, 0) # Shape: (B, T, n_q)
        return codes

    def decode(self, codes):
        codes = codes.permute(2, 0, 1) # Shape [n_q, B, T]
        quantized = self.quantizer.decode(codes)
        x = self.decoder(quantized)
        return x
