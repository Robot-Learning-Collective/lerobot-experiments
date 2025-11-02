import torch
import torch.nn as nn
from operator import mul
from functools import reduce
from typing import Optional

from lerobot.policies.transformer_diffusion import TransformerForDiffusion
from lerobot.policies.smolandfast.vq import ResidualVectorQuantizer
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler


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
                 emdedding_dim: int = 64,
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
            nn.Conv1d(
                in_channels=final_conv_channels,
                out_channels=emdedding_dim,
                kernel_size=3,
                padding=1,
            ),
        ]
        self.out_layer = nn.Sequential(*out_layer)
        self.latent_horizon_div = reduce(mul, ratios)


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
        x = x.permute(0, 2, 1)  # (B, C, T) after LSTM

        # Output layer
        x = self.out_layer(x)

        return x
    

class EncoderTransformer(nn.Module):
    def __init__(self,
                 input_dims: int = 2,
                 base_features: int = 16,
                 emdedding_dim: int = 64,
                 horizon: int = 8,
                 ratios: list[int] = [1],
                 num_residual_layers: int = 1,
                 num_lstm_layers: int = 2):
        super().__init__()
        
        self.latent_horizon_div = reduce(mul, ratios)

        self.patch_emb = nn.Conv1d(
            in_channels=input_dims,
            out_channels=emdedding_dim,
            kernel_size=self.latent_horizon_div + 1,
            stride=self.latent_horizon_div,
            padding=self.latent_horizon_div // 2,
        )

        T_codes = horizon // self.latent_horizon_div
        self.pos_emb = nn.Parameter(torch.zeros(1, T_codes, emdedding_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emdedding_dim,
            nhead=4,
            dim_feedforward=4 * emdedding_dim,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_lstm_layers
        )

        self.ln_f = nn.LayerNorm(emdedding_dim)


    def forward(self, x):
        # Input: [B, T, D_in]
        x = x.permute(0, 2, 1)  # [B, D_in, T]
        x = self.patch_emb(x)      # [B, C, T_codes]
        x = x.permute(0, 2, 1)    # [B, T_codes, C]

        x = x + self.pos_emb
        x = self.transformer(x) # [B, 3, 64]
        x = self.ln_f(x)

        # Permute back for VQ: [B, C, T_codes]
        x = x.permute(0, 2, 1)

        return x


class DiffusionAE(nn.Module):
    """
    1.  Encoder: [B, T, D_in] -> [B, D_latent, T_latent]
    2.  Quantizer: [B, D_latent, T_latent] -> [B, D_latent, T_latent] (quantized) + codes
    3.  DiffusionDecoder: Denoises noise [B, T, D_in] conditioned on [B, T_latent, D_latent]
    """
    def __init__(
        self,
        # Encoder params
        input_dims: int = 2,
        horizon: int = 12, # Original sequence length T
        base_features: int = 16,
        ratios: list[int] = [2, 2],
        num_residual_layers: int = 1,
        num_lstm_layers: int = 2,
        
        # VQ params
        vocab_size: int = 1024,
        encoded_dim: int = 4, # num quantizers (user's 'encoded_dim')
        emdedding_dim: int = 64, # Latent dim 'C'
        
        # Diffusion Transformer params
        n_layer: int = 12,
        n_head: int = 12,
        n_emb: int = 768,
        n_cond_layers: int = 4,
        
        # Scheduler params
        num_train_timesteps: int = 1000,
        prediction_type: str = 'epsilon' # 'epsilon' or 'sample'
    ):
        super().__init__()
        self.input_dims = input_dims
        self.horizon = horizon
        self.emdedding_dim = emdedding_dim

        # 1. Encoder
        self.encoder = EncoderTransformer(
            horizon=horizon,
            input_dims=input_dims,
            base_features=base_features,
            emdedding_dim=emdedding_dim,
            ratios=ratios,
            num_residual_layers=num_residual_layers,
            num_lstm_layers=num_lstm_layers
        )

        # self.encoder = Encoder(
        #     input_dims=input_dims,
        #     base_features=base_features,
        #     emdedding_dim=emdedding_dim,
        #     ratios=ratios,
        #     num_residual_layers=num_residual_layers,
        #     num_lstm_layers=num_lstm_layers
        # )
        
        # 2. Quantizer
        self.quantizer = ResidualVectorQuantizer(
            dimension=emdedding_dim,
            n_q=encoded_dim,
            bins=vocab_size,
        )

        latent_horizon = self.horizon // self.encoder.latent_horizon_div
        
        # Diffusion Decoder
        self.diffusion_decoder = TransformerForDiffusion(
            input_dim=self.input_dims,       # D_in
            output_dim=self.input_dims,      # D_in
            horizon=self.horizon,            # T
            n_obs_steps=latent_horizon,      # This is used to set T_cond
            cond_dim=self.emdedding_dim,     # C
            n_layer=n_layer,
            n_head=n_head,
            n_emb=n_emb,
            time_as_cond=True,
            obs_as_cond=True,
            n_cond_layers=n_cond_layers
        )

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule='squaredcos_cap_v2', # A common, effective schedule
            prediction_type=prediction_type,
        )
        self.prediction_type = prediction_type
        self.vocab_size = vocab_size

    def encode(self, x):
        x = self.encoder(x)
        codes = self.quantizer.encode(x)
        codes = codes.permute(1, 2, 0) # Shape: (B, T, n_q)
        return codes

    @torch.no_grad()
    def decode(self, 
               codes: torch.Tensor, 
               num_inference_steps: int = 50,
               generator: Optional[torch.Generator] = None
               ) -> torch.Tensor:
        # 1. Convert discrete codes to continuous quantized vectors
        # Ensure codes are [n_q, B, T_codes] for quantizer.decode
        if codes.shape[2] == self.quantizer.n_q:
            codes = codes.permute(2, 0, 1) # [B, T_codes, n_q] -> [n_q, B, T_codes]

        # quantized: [B, C, T_codes]
        quantized = self.quantizer.decode(codes)
        
        # 2. Prepare for Diffusion Sampling
        # Condition must be [B, T_cond, D_cond], so [B, T_codes, C]
        cond = quantized.permute(0, 2, 1)
        B = cond.shape[0]
        device = cond.device
        
        # Start with pure noise in the shape of the target output
        shape = (B, self.horizon, self.input_dims)
        sample = torch.randn(shape, device=device, generator=generator)

        # 3. Denoising Loop
        self.noise_scheduler.set_timesteps(num_inference_steps)
        
        for t in self.noise_scheduler.timesteps:
            # Predict the noise (or sample)
            pred = self.diffusion_decoder(
                sample=sample,
                timestep=t,
                cond=cond,
            )

            # Compute the previous noisy sample x_t-1
            sample = self.noise_scheduler.step(
                pred, t, sample, generator=generator
            ).prev_sample

        return sample
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass for training.
        
        Args:
            x (torch.Tensor): Input tensor, shape [B, T, D_in]
            
        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - total_loss (scalar)
            - diffusion_loss (scalar)
            - commit_loss (scalar)
        """
        B = x.shape[0]
        device = x.device

        # 1. Encode and Quantize
        # continuous_codes: [B, C, T_codes]
        continuous_codes = self.encoder(x)

        # quantized: [B, C, T_codes]
        # commit_loss: scalar tensor
        quantized, codes, commit_loss = self.quantizer(continuous_codes)

        # 2. Prepare for Diffusion
        # The target for diffusion is the original input `x`
        target = x

        # The condition is the permuted quantized vector
        # Shape: [B, T_codes, C]
        cond = quantized.permute(0, 2, 1)

        # 3. Diffusion Forward Process (Training)
        # Sample random noise
        noise = torch.randn(target.shape, device=device)

        # Sample random timesteps
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (B,), device=device
        ).long()

        # Add noise to the clean target (forward diffusion)
        noisy_target = self.noise_scheduler.add_noise(
            target, noise, timesteps
        )

        # 4. Predict Noise (or Sample)
        # The diffusion transformer predicts the noise `epsilon` or the original `x_0`
        pred = self.diffusion_decoder(
            sample=noisy_target,
            timestep=timesteps,
            cond=cond,
        )

        if self.prediction_type == 'epsilon':
            loss_target = noise
        elif self.prediction_type == 'sample':
            loss_target = target
        else:
            raise ValueError(f"Unsupported prediction type {self.prediction_type}")

        codes = codes.permute(1, 2, 0) # Shape: (B, T, n_q)
        return pred, codes, commit_loss.mean(), loss_target
