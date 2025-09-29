import torch
from torch import nn

from .cores import FeedforwardRecurrent
from .feedforwards import InputDense
from .recurrents import CvtLstm


class FrameDecoder(nn.Module):
    """Upsampling decoder mapping core features (C x 18 x 32) -> frame (1 x 144 x 256).

    Assumes input spatial size is 18x32 (from 144x256 downsampled by factor 8).
    """
    def __init__(self, in_channels=128, hidden_channels=(64, 32, 16), out_channels=1):
        super().__init__()
        layers = []
        c = in_channels
        # Upsample 18x32 -> 36x64 -> 72x128 -> 144x256
        for h in hidden_channels:
            layers.append(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False))
            layers.append(nn.Conv2d(c, h, 3, padding=1))
            layers.append(nn.GELU())
            c = h
        layers.append(nn.Conv2d(c, out_channels, 3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):  # x: [B, C, 18, 32]
        frame = self.net(x)  # [B, 1, 144, 256]
        return frame


class AutoregressiveVideoModel(nn.Module):
    """Sequence-to-multi-step autoregressive frame predictor.

    Usage:
        model = AutoregressiveVideoModel(pred_steps=5)
        preds = model(frames)  # frames: [B, T_in, 1, 64, 64]; preds: [B, pred_steps, 1, 64, 64]
    """

    def __init__(self, core: FeedforwardRecurrent, decoder: FrameDecoder, pred_steps: int = 5):
        super().__init__()
        self.core = core
        self.decoder = decoder
        self.pred_steps = int(pred_steps)
        # Learnable modulation embedding (acts as constant conditioning token)
        self.modulation_embed = nn.Parameter(torch.zeros(1, 1, 64, 64))

        self.core._init(perspectives=1, modulations=self.core.feedforward.channels, streams=1)

    def reset_state(self):
        self.core.reset()

    def encode_frame(self, frame: torch.Tensor):  # frame: [B, 1, 64, 64]
        # Feedforward
        f = self.core.feedforward([frame], stream=None)  # [B, F, 8, 8]
        z = torch.zeros_like(f)  # dummy modulation projection space

        features = self.core.recurrent([f, z], stream=0)  # [B, C, 8, 8]

        # NaN / Inf safeguard
        if torch.isnan(features).any() or torch.isinf(features).any():
            features = torch.nan_to_num(features, nan=0.0, posinf=1e4, neginf=-1e4)
        return features

    def forward(self, frames: torch.Tensor):
        """Forward pass.

        Parameters
        ----------
        frames : Tensor
            [B, T_in, 1, 144, 256]

        Returns
        -------
        Tensor
            [B, pred_steps, 1, 144, 256]
        """
        B, T_in, C, H, W = frames.shape
        assert C == 1 and H == 144 and W == 256, "Expected frames of shape [B, T, 1, 144, 256]"

        self.reset_state()

        # Encode observed frames (state accumulated in recurrent core)
        for t in range(T_in):
            self.encode_frame(frames[:, t])

        preds = []
        last_in = frames[:, -1]
        cur_frame = last_in
        for k in range(self.pred_steps):
            features = self.encode_frame(cur_frame)
            pred = self.decoder(features)
            preds.append(pred)
            cur_frame = pred
        return torch.stack(preds, dim=1)


def build_autoregressive_video_model(pred_steps: int = 5) -> AutoregressiveVideoModel:
    """Factory for a lightweight autoregressive frame prediction model using only
    the encoder (InputDense) + recurrent core + simple upsampling decoder.
    """
    feedforward = InputDense(
        input_spatial=6,
        input_stride=2,
        block_channels=[32, 64, 128],
        block_groups=[1, 2, 4],
        block_layers=[2, 2, 2],
        block_temporals=[3, 3, 3],
        block_spatials=[3, 3, 3],
        block_pools=[2, 2, 1],
        out_channels=128,
        nonlinear="gelu",
    )
    recurrent = CvtLstm(
        in_channels=256,
        out_channels=128,
        hidden_channels=256,
        common_channels=512,
        groups=8,
        spatial=3,
    )
    core = FeedforwardRecurrent(feedforward=feedforward, recurrent=recurrent)
    decoder = FrameDecoder(in_channels=recurrent.out_channels)
    return AutoregressiveVideoModel(core=core, decoder=decoder, pred_steps=pred_steps)
