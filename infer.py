# --coding:utf-8--
import os

import librosa
from librosa import display
from scipy import signal

from sklearn import metrics
import torchaudio
import torch
from decoder.pretrained import WavTokenizer


import logging
from dotenv import load_dotenv

load_dotenv()

import pathlib

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

input_path = "./data/test/test-clean.txt"
out_folder = "./result/infer"
# os.system("rm -r %s"%(out_folder))
# os.system("mkdir -p %s"%(out_folder))
# ll="libritts_testclean500_large"
ll = "wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn_testclean"

tmptmp = out_folder + "/" + ll

if os.path.exists(tmptmp):
    os.system("rm -r %s" % (tmptmp))
os.makedirs(tmptmp)


logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(pathlib.Path(out_folder) / ll / "infer.log")
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


device = torch.device("cuda:0")


# 自己数据模型加载
config_path = "./configs/wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
# model_path = "./WavTokenizer/result/train/wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn/lightning_logs/version_3/checkpoints/wavtokenizer_checkpoint_epoch=24_step=137150_val_loss=5.6731.ckpt"
model_path = pathlib.Path(os.getenv("MODEL_PATH_40"))
wavtokenizer = WavTokenizer.from_pretrained0802(config_path, model_path)
wavtokenizer = wavtokenizer.to(device)


with open(input_path, "r") as fin:
    x = fin.readlines()

x = [i.strip() for i in x]

# 完成一些加速处理


def plot_waveform_comparison(
    original_wav,
    reconstructed_wav,
    original_sr,
    save_path,
    min_length=None,
    max_length=None,
):
    """
    绘制原始音频和重建音频的波形对比图
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 10))

    # 转换为numpy数组
    original_np = original_wav.squeeze().numpy()
    reconstructed_np = reconstructed_wav.squeeze().numpy()

    # 截取相同长度
    # min_length = min(len(original_np), len(reconstructed_np))
    min_length = 0 if min_length is None else min_length
    max_length = (
        min(len(original_np), len(reconstructed_np))
        if max_length is None
        else max_length
    )

    original_cropped = original_np[min_length:max_length]
    reconstructed_cropped = reconstructed_np[min_length:max_length]

    # 时间轴
    time_axis = np.linspace(
        min_length / original_sr, max_length / original_sr, max_length - min_length
    )

    # 绘制原始音频
    ax1.plot(time_axis, original_cropped, color="blue", alpha=0.7, linewidth=0.5)
    ax1.set_title(f"Original Audio (SR: {original_sr} Hz)", fontsize=14)
    ax1.set_ylabel("Amplitude")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(min_length / original_sr, max_length / original_sr)

    # 绘制重建音频
    ax2.plot(time_axis, reconstructed_cropped, color="red", alpha=0.7, linewidth=0.5)
    ax2.set_title("Reconstructed Audio (SR: 24000 Hz)", fontsize=14)
    ax2.set_ylabel("Amplitude")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(min_length / original_sr, max_length / original_sr)

    # 计算差异
    difference = original_cropped - reconstructed_cropped

    # 绘制差异
    ax3.plot(time_axis, difference, color="green", alpha=0.7, linewidth=0.5)
    ax3.set_title("Difference (Original - Reconstructed)", fontsize=14)
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Amplitude")
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(min_length / original_sr, max_length / original_sr)

    # 添加统计信息
    mse = np.mean(difference**2)
    max_diff = np.max(np.abs(difference))
    ax3.text(
        0.02,
        0.98,
        f"MSE: {mse:.6f}\nMax |diff|: {max_diff:.6f}",
        transform=ax3.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Waveform comparison saved: {save_path}")


def plot_perceptual_comparison(
    original_wav, reconstructed_wav, original_sr, save_path, filename
):
    """
    绘制符合人类听感的音频对比图
    """
    # 转换为numpy数组并重采样到相同采样率
    original_np = original_wav.squeeze().numpy()
    reconstructed_np = reconstructed_wav.squeeze().numpy()

    # 重采样原始音频到24kHz以便比较
    if original_sr != 24000:
        original_resampled = librosa.resample(
            original_np, orig_sr=original_sr, target_sr=24000
        )
    else:
        original_resampled = original_np

    # 截取相同长度
    min_length = min(len(original_resampled), len(reconstructed_np))
    original_cropped = original_resampled[:min_length]
    reconstructed_cropped = reconstructed_np[:min_length]

    fig, axes = plt.subplots(3, 2, figsize=(20, 15))

    # 1. Mel频谱图对比
    # 原始音频Mel频谱图
    mel_orig = librosa.feature.melspectrogram(
        y=original_cropped, sr=24000, n_mels=128, fmax=8000
    )
    mel_orig_db = librosa.power_to_db(mel_orig, ref=np.max)

    im1 = librosa.display.specshow(
        mel_orig_db, y_axis="mel", x_axis="time", sr=24000, fmax=8000, ax=axes[0, 0]
    )
    axes[0, 0].set_title("Original - Mel Spectrogram", fontsize=14)
    plt.colorbar(im1, ax=axes[0, 0], format="%+2.0f dB")

    # 重建音频Mel频谱图
    mel_recon = librosa.feature.melspectrogram(
        y=reconstructed_cropped, sr=24000, n_mels=128, fmax=8000
    )
    mel_recon_db = librosa.power_to_db(mel_recon, ref=np.max)

    im2 = librosa.display.specshow(
        mel_recon_db, y_axis="mel", x_axis="time", sr=24000, fmax=8000, ax=axes[0, 1]
    )
    axes[0, 1].set_title("Reconstructed - Mel Spectrogram", fontsize=14)
    plt.colorbar(im2, ax=axes[0, 1], format="%+2.0f dB")

    # 2. MFCC特征对比
    # 原始音频MFCC
    mfcc_orig = librosa.feature.mfcc(y=original_cropped, sr=24000, n_mfcc=13)
    im3 = librosa.display.specshow(mfcc_orig, x_axis="time", ax=axes[1, 0])
    axes[1, 0].set_title("Original - MFCC", fontsize=14)
    axes[1, 0].set_ylabel("MFCC Coefficients")
    plt.colorbar(im3, ax=axes[1, 0])

    # 重建音频MFCC
    mfcc_recon = librosa.feature.mfcc(y=reconstructed_cropped, sr=24000, n_mfcc=13)
    im4 = librosa.display.specshow(mfcc_recon, x_axis="time", ax=axes[1, 1])
    axes[1, 1].set_title("Reconstructed - MFCC", fontsize=14)
    axes[1, 1].set_ylabel("MFCC Coefficients")
    plt.colorbar(im4, ax=axes[1, 1])

    # 3. 频谱包络对比和感知差异
    # 计算频谱
    freqs_orig, times_orig, Sxx_orig = signal.spectrogram(
        original_cropped, fs=24000, nperseg=1024
    )
    freqs_recon, times_recon, Sxx_recon = signal.spectrogram(
        reconstructed_cropped, fs=24000, nperseg=1024
    )

    # 频谱包络对比
    axes[2, 0].semilogy(
        freqs_orig, np.mean(Sxx_orig, axis=1), "b-", labe0l="Original", alpha=0.7
    )
    axes[2, 0].semilogy(
        freqs_recon, np.mean(Sxx_recon, axis=1), "r-", label="Reconstructed", alpha=0.7
    )
    axes[2, 0].set_xlabel("Frequency (Hz)")
    axes[2, 0].set_ylabel("Power Spectral Density")
    axes[2, 0].set_title("Average Power Spectral Density Comparison", fontsize=14)
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].set_xlim(0, 12000)

    # 感知指标
    # Mel频谱图差异
    mel_diff = np.abs(mel_orig_db - mel_recon_db)
    im5 = librosa.display.specshow(
        mel_diff, y_axis="mel", x_axis="time", sr=24000, fmax=8000, ax=axes[2, 1]
    )
    axes[2, 1].set_title("Mel Spectrogram Difference", fontsize=14)
    plt.colorbar(im5, ax=axes[2, 1], format="%+2.0f dB")

    # 计算感知指标
    mel_mse = np.mean(mel_diff**2)
    mfcc_mse = np.mean((mfcc_orig - mfcc_recon) ** 2)

    # 频谱质心对比
    spectral_centroids_orig = librosa.feature.spectral_centroid(
        y=original_cropped, sr=24000
    )[0]
    spectral_centroids_recon = librosa.feature.spectral_centroid(
        y=reconstructed_cropped, sr=24000
    )[0]
    centroid_diff = np.mean(np.abs(spectral_centroids_orig - spectral_centroids_recon))

    # 在图上添加感知指标
    metrics_text = f"""Perceptual Metrics:
        Mel MSE: {mel_mse:.4f}
        MFCC MSE: {mfcc_mse:.4f}
        Spectral Centroid Diff: {centroid_diff:.2f} Hz
        Max Mel Diff: {np.max(mel_diff):.2f} dB"""

    axes[2, 1].text(
        0.02,
        0.98,
        metrics_text,
        transform=axes[2, 1].transAxes,
        verticalalignment="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Perceptual comparison saved: {save_path}")

    return {
        "mel_mse": mel_mse,
        "mfcc_mse": mfcc_mse,
        "centroid_diff": centroid_diff,
        "max_mel_diff": np.max(mel_diff),
    }


features_all = []
original_wavs = []

max_len = 1
for i in range(max_len):

    wav, sr = torchaudio.load(x[i])
    # print("***:",x[i])
    # wav = convert_audio(wav, sr, 24000, 1)                             # (1,131040)
    # audio_path = out_folder + '/' + ll + '/' + x[i].split('/')[-1]
    original_filename = x[i].split("/")[-1]
    name_without_ext = pathlib.Path(original_filename).stem
    new_filename = f"{name_without_ext}_original.wav"
    audio_path = pathlib.Path(out_folder) / ll / new_filename
    torchaudio.save(
        audio_path, wav, sample_rate=sr, encoding="PCM_S", bits_per_sample=16
    )

    bandwidth_id = torch.tensor([0])
    wav = wav.to(device)
    logger.info(f"Processing {x[i]} with original sample rate {sr}")

    features, discrete_code = wavtokenizer.encode_infer(wav, bandwidth_id=bandwidth_id)

    logger.info(
        f"Original waveform shape: {wav.shape}, Extracted features shape: {features.shape}, Discrete codes shape: {discrete_code.shape}"
    )
    features_all.append(features)
    original_wavs.append((wav.cpu(), sr))


for i in range(max_len):

    bandwidth_id = torch.tensor([0])
    bandwidth_id = bandwidth_id.to(device)

    logger.info(
        f"Decoding features of shape: {features_all[i].shape} with bandwidth_id: {bandwidth_id}"
    )
    audio_out = wavtokenizer.decode(features_all[i], bandwidth_id=bandwidth_id)
    # print(i,time.time())
    # breakpoint()                        # (1, 131200)
    # audio_path = out_folder + "/" + ll + "/" + x[i].split("/")[-1]
    original_filename = x[i].split("/")[-1]
    name_without_ext = pathlib.Path(original_filename).stem
    new_filename = f"{name_without_ext}_reconstruction.wav"
    audio_path = pathlib.Path(out_folder) / ll / new_filename
    # os.makedirs(out_folder + '/' + ll, exist_ok=True)
    torchaudio.save(
        audio_path,
        audio_out.cpu(),
        sample_rate=24000,
        encoding="PCM_S",
        bits_per_sample=16,
    )

    # 生成波形对比图
    # plot_filename = f"{name_without_ext}_waveform_comparison.png"
    # plot_path = pathlib.Path(out_folder) / ll / plot_filename

    # original_wav, original_sr = original_wavs[i]
    # plot_waveform_comparison(
    #     original_wav, audio_out.cpu(), original_sr, plot_path, name_without_ext
    # )

    # 生成感知对比图
    plot_filename = f"{name_without_ext}_perceptual_comparison.png"
    plot_path = pathlib.Path(out_folder) / ll / plot_filename

    original_wav, original_sr = original_wavs[i]
    metrics_ = plot_perceptual_comparison(
        original_wav, audio_out.cpu(), original_sr, plot_path, name_without_ext
    )

    logger.info(f"Perceptual metrics for {name_without_ext}: {metrics_}")
