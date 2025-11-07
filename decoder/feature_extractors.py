from random import sample
from typing import List, Optional

from pyparsing import Opt
import torch
import torchaudio
from torch import nn
import math
from decoder.modules import safe_log
from encoder.modules import SEANetEncoder, SEANetDecoder
from encoder import EncodecModel
from encoder.quantization import ResidualVectorQuantizer, vq


class FeatureExtractor(nn.Module):
    """Base class for feature extractors."""

    def forward(self, audio: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Extract features from the given audio.

        Args:
            audio (Tensor): Input audio waveform.

        Returns:
            Tensor: Extracted features of shape (B, C, L), where B is the batch size,
                    C denotes output features, and L is the sequence length.
        """
        raise NotImplementedError("Subclasses must implement the forward method.")


class MelSpectrogramFeatures(FeatureExtractor):
    def __init__(
        self,
        sample_rate=24000,
        n_fft=1024,
        hop_length=256,
        n_mels=100,
        padding="center",
    ):
        super().__init__()
        if padding not in ["center", "same"]:
            raise ValueError("Padding must be 'center' or 'same'.")
        self.padding = padding
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            center=padding == "center",
            power=1,
        )

    def forward(self, audio, **kwargs):
        if self.padding == "same":
            pad = self.mel_spec.win_length - self.mel_spec.hop_length
            audio = torch.nn.functional.pad(audio, (pad // 2, pad // 2), mode="reflect")
        mel = self.mel_spec(audio)
        features = safe_log(mel)
        return features


class EncodecFeatures(FeatureExtractor):
    def __init__(
        self,
        encodec_model: str = "encodec_24khz",
        bandwidths: Optional[List[float]] = None,
        train_codebooks: bool = False,
        num_quantizers: Optional[int] = None,
        downsamples: List[int] = [6, 5, 5, 4],
        vq_bins: int = 16384,
        vq_kmeans: int = 800,
        sample_rate: int = 24000,
    ):
        super().__init__()

        # breakpoint()
        if num_quantizers is None and bandwidths is None:
            raise ValueError(
                "Either 'num_quantizers' or 'bandwidths' must be provided."
            )
        if num_quantizers is not None and num_quantizers != 1:
            raise ValueError(
                "When 'num_quantizers' is provided, it must be set to 1 for single codebook usage."
            )

        self.sample_rate = sample_rate
        total_downsample = 1
        for r in downsamples:
            total_downsample *= r
        self.frame_rate = sample_rate / total_downsample

        self.n_q = (
            num_quantizers
            if num_quantizers is not None
            else int(bandwidths[-1] * 1000 / (math.log2(vq_bins) * self.frame_rate))
        )
        self.bandwidths = bandwidths # None or List[float]
        self.use_single_codebook = num_quantizers is not None and num_quantizers == 1
        
        # Define encoder, decoder, and quantizer
        encoder = SEANetEncoder(
            causal=False,
            n_residual_layers=1,
            norm="weight_norm",
            pad_mode="reflect",
            lstm=2,
            dimension=512,
            channels=1,
            n_filters=32,
            ratios=downsamples,
            activation="ELU",
            kernel_size=7,
            residual_kernel_size=3,
            last_kernel_size=7,
            dilation_base=2,
            true_skip=False,
            compress=2,
        )
        decoder = SEANetDecoder(
            causal=False,
            n_residual_layers=1,
            norm="weight_norm",
            pad_mode="reflect",
            lstm=2,
            dimension=512,
            channels=1,
            n_filters=32,
            ratios=[8, 5, 4, 2],
            activation="ELU",
            kernel_size=7,
            residual_kernel_size=3,
            last_kernel_size=7,
            dilation_base=2,
            true_skip=False,
            compress=2,
        )
        quantizer = ResidualVectorQuantizer(
            dimension=512,
            n_q=self.n_q,
            bins=vq_bins,
            kmeans_iters=vq_kmeans,
            decay=0.99,
            kmeans_init=True,
        )

        # breakpoint()
        if encodec_model == "encodec_24khz":
            self.encodec = EncodecModel(
                encoder=encoder,
                decoder=decoder,
                quantizer=quantizer,
                target_bandwidths=bandwidths,
                sample_rate=self.sample_rate,
                channels=1,
            )
        else:
            raise ValueError(
                f"Unsupported encodec_model: {encodec_model}. Supported options are 'encodec_24khz'."
            )
        for param in self.encodec.parameters():
            param.requires_grad = True
        

    def forward(self, audio: torch.Tensor, bandwidth_id: Optional[torch.Tensor]):
        if self.training:
            self.encodec.train()

        audio = audio.unsqueeze(1)  # audio(16,24000) -> audio(16,1,24000)

        # breakpoint()

        emb = self.encodec.encoder(audio)
        if self.use_single_codebook:
            q_res = self.encodec.quantizer(emb, self.frame_rate)
        else:
            if bandwidth_id is None:
                raise ValueError(
                    "bandwidth_id must be provided when using multiple codebooks."
                )
            q_res = self.encodec.quantizer(
                emb, self.frame_rate, bandwidth=self.bandwidths[bandwidth_id]
            )
        quantized = q_res.quantized
        codes = q_res.codes
        commit_loss = q_res.penalty  # codes(8,16,75),features(16,128,75)

        return quantized, codes, commit_loss


    def infer(self, audio: torch.Tensor, bandwidth_id: Optional[torch.Tensor]):
        if self.training:
            self.encodec.train()

        audio = audio.unsqueeze(1)  # audio(16,24000)
        emb = self.encodec.encoder(audio)
        if self.use_single_codebook:
            q_res = self.encodec.quantizer.infer(emb, self.frame_rate)
        else:
            if bandwidth_id is None:
                raise ValueError(
                    "bandwidth_id must be provided when using multiple codebooks."
                )
            q_res = self.encodec.quantizer.infer(
                emb, self.frame_rate, bandwidth=self.bandwidths[bandwidth_id]
            )
        quantized = q_res.quantized
        codes = q_res.codes
        commit_loss = q_res.penalty  # codes(8,16,75),features(16,128,75)

        return quantized, codes, commit_loss
