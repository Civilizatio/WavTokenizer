# Metrics

主要对训练过程中一些指标进行总结


## Discriminator Loss

使用 hinge loss 作为判别器的损失函数，代码位于 `decoder/loss.py`。

论文中数学原理为：
$$
\begin{aligned}
\mathcal{L}_{dis}(X, \tilde{X}) = \frac{1}{K} \sum_{k=1}^{K} & \left\{\max \left(0,1-D_{k}(X)\right) + \right. \\
& \left. \max \left(0,1+D_{k}(\tilde{X})\right)\right\}
\end{aligned}
$$

实际上，应该为 SoundStream 论文中的公式：
$$
\begin{aligned}
\mathcal{L}_{dis}(X, \tilde{X}) = \frac{1}{K} \sum_{k=1}^{K} & \left\{\frac{1}{T_k}\sum_t\max \left(0,1-D_{k,t}(X)\right) + \right. \\
& \frac{1}{T_k}\sum_t\left. \max \left(0,1+D_{k,t}(\tilde{X})\right)\right\}
\end{aligned}
$$

> 这里的 $k$ 表示第 $k$ 个判别器，$t$ 表示时间步，$T_k$ 是第 $k$ 个判别器的时间步总数。不同于一般的判别器输出 $0\sim 1$ 的分数，这里输出的是一段长度为 $T_k$ 的序列，代表每一段的分数。

代码为：
```python
loss = 0
r_losses = []
g_losses = []
for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
    r_loss = torch.mean(torch.clamp(1 - dr, min=0))
    g_loss = torch.mean(torch.clamp(1 + dg, min=0))
    loss += r_loss + g_loss
    r_losses.append(r_loss.item())
    g_losses.append(g_loss.item())
```

应用这一段代码的是 `decoder/discriminators.py` 中的 `MultiResolutionSTFTDiscriminator`。

### DAC loss

针对 DAC （Descript Audio Codec） 论文中提出的判别器损失函数，代码位于 `decoder/loss.py`。

主要的部分也是 `MPD` 和 `MSD`。区别之处在于这里使用的损失是 MSE loss，而不是 hinge loss。

数学公式为：

$$
\begin{aligned}
\mathcal{L}_{D A C}(X, \tilde{X}) = \frac{1}{K} \sum_{k=1}^{K} & \left\{\frac{1}{T_k} \sum_t \|D_{k,t}(\tilde{X})\|^2 + \right. \\
& \left. \frac{1}{T_k} \sum_t \|1-D_{k,t}(X)\|^2\right\}
\end{aligned}
$$

代码如下：
```python
def discriminator_loss(self, fake, real):
    d_fake, d_real = self.forward(fake.clone().detach(), real)

    loss_d = 0
    for x_fake, x_real in zip(d_fake, d_real):
        loss_d += torch.mean(x_fake[-1] ** 2)
        loss_d += torch.mean((1 - x_real[-1]) ** 2)
    return loss_d
```

最后总的损失为：
```python
loss = loss_mp + self.hparams.mrd_loss_coeff * loss_mrd + loss_dac
```

## Generator Loss

同 Discriminator Loss 一样，使用 hinge loss 作为生成器的损失函数，代码位于 `decoder/loss.py`。

公式为：
$$
\begin{aligned}
\mathcal{L}_{gen}(X, \tilde{X}) = \frac{1}{K} \sum_{k=1}^{K} & \left\{\frac{1}{T_k}\sum_t \max (0, 1- D_{k,t}(\tilde{X}))\right\}
\end{aligned}
$$

代码为：

```python
loss = 0
gen_losses = []
for dg in disc_outputs:
    l = torch.mean(torch.clamp(1 - dg, min=0))
    gen_losses.append(l)
    loss += l
```

应用这段代码的也是 `decoder/discriminators.py` 中的 `MultiPeriodDiscriminator` 和 `MultiScaleDiscriminator`。

### DAC loss

同 Discriminator Loss 一样，使用 MSE loss 作为生成器的损失函数，代码位于 `decoder/loss.py`。

数学公式为：
$$
\begin{aligned}
\mathcal{L}_{D A C}^{gen}(X, \tilde{X}) = \frac{1}{K} \sum_{k=1}^{K} & \left\{\frac{1}{T_k} \sum_t \|D_{k,t}(\tilde{X}) - 1\|^2\right\}
\end{aligned}
$$

### Feature Matching Loss

主要是针对不同鉴别器的中间层输出进行 $l_1$ 范数匹配，代码位于 `decoder/loss.py`。

数学公式为：
$$
\begin{aligned}
\mathcal{L}_{F M}(X, \tilde{X}) = & \frac{1}{K} \sum_{k=1}^{K} \frac{1}{L_k}  \sum_l \|D_{k,l}(X) - D_{k,l}(\tilde{X})\|^2
\end{aligned}
$$

代码为：
```python
loss = 0
for dr, dg in zip(fmap_r, fmap_g):
    for rl, gl in zip(dr, dg):
        loss += torch.mean(torch.abs(rl - gl))
```

应用这段代码的也是 `decoder/discriminators.py` 中的 `MultiPeriodDiscriminator` 和 `MultiScaleDiscriminator`。以及 DAC 的生成损失。

### Mel-Spectrogram Loss

使用 $l_1$ 范数作为 Mel 频谱的损失函数，代码位于 `decoder/loss.py`。

数学公式为：
$$
\mathcal{L}_{mel}(X, \tilde{X}) = \|
Mel(X) - Mel(\tilde{X})\|_1
$$

### Commit Loss

量化误差，表示模型输出的离散化表示与目标离散化表示之间的差异。这个损失通常用于训练VQ-VAE模型。
代码位于 `encoder/quantization/core_vq.py`；

```python
if self.training: 
    if self.commitment_weight > 0:
        commit_loss = F.mse_loss(quantize.detach(), x)
        loss = loss + commit_loss * self.commitment_weight
```
数学公式为：
$$
\text{commitment\_loss} =  \beta\|z_e(x) - \text{sg}[e] \|^2
$$
VQ-VAE 原文中说明 $\beta$ 的取值范围为 0.25 到 2.0 之间，但是影响不大。这里默认应该是 $1$。

### Summary of Total Loss

最终的总损失为：

```python
loss = (
    loss_gen_mp # multiperiod discriminator loss
    + self.hparams.mrd_loss_coeff * loss_gen_mrd # multiscale discriminator loss
    + loss_fm_mp # multiperiod discriminator feature matching loss
    + self.hparams.mrd_loss_coeff * loss_fm_mrd # multiscale discriminator feature matching loss
    + self.mel_loss_coeff * mel_loss # mel-spectrogram loss
    + 1000 * commit_loss # commit loss
    + loss_dac_1 # dac discriminator loss
    + loss_dac_2 # dac feature matching loss
)
```

## Validation Metrics


### periodicity loss

- **原理**：衡量预测音频和真实音频的周期性（periodicity）差异。
- **计算方法**：
  1. 用 `torchcrepe.predict` 得到每帧的周期性分数（`periodicity`），范围一般在 [0, 1]。
  2. 计算预测和真实周期性分数的均方误差（MSE），再开根号（RMSE）。
  3. 对所有 batch 求均值。
- **公式**：
  $$
  \text{periodicity\_loss} = \text{mean}_\text{batch}\left(\sqrt{\frac{1}{N}\sum_{i=1}^N (\text{pred}_i - \text{true}_i)^2}\right)
  $$
  其中 $N$ 为帧数。


### pitch loss

- **原理**：只在“清音”帧（voiced）上，衡量预测音高和真实音高的差异（单位为cents，音高对数刻度）。
- **计算方法**：
  1. 只在预测和真实都为“清音”的帧上，取出音高值。
  2. 计算音高的对数差异（以cents为单位）：$1200 \times (\log_2(\text{true}) - \log_2(\text{pred}))$
  3. 计算这些差异的均方根（RMSE）。
- **公式**：
  $$
  \text{pitch\_loss} = \sqrt{\frac{1}{M}\sum_{j=1}^M \left[1200 \cdot (\log_2(\text{true}_j) - \log_2(\text{pred}_j))\right]^2}
  $$
  其中 $M$ 为“清音”帧数。

### 3. f1

- **原理**：衡量预测和真实音频在“清音/非清音”判别上的一致性（F1分数）。
- **计算方法**：
  1. 统计预测和真实都为“清音”的帧数（True Positive）。
  2. 统计预测为“清音”但真实为“非清音”的帧数（False Positive）。
  3. 统计真实为“清音”但预测为“非清音”的帧数（False Negative）。
  4. 计算精确率（Precision）、召回率（Recall）和 F1 分数。
- **公式**：
  $$
  \text{Precision} = \frac{TP}{TP + FP}
  $$
  $$
  \text{Recall} = \frac{TP}{TP + FN}
  $$
  $$
  \text{F1} = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
  $$



### UTMOS Score

UTMOS（**Universal Text-to-Speech MOS**）分数是一种**自动化的音频主观质量评估指标**，用于模拟人类对语音合成、语音增强等音频的主观打分（MOS, Mean Opinion Score，均值意见分数）。

- **UTMOS** 是由日本 NTT 提出的基于深度学习的 MOS 预测模型。
- 它输入一段音频，输出一个分数（通常在 1~5 之间），分数越高表示音频质量越好，越接近真实人声。
- UTMOS 通过大量真实主观打分数据训练，能较好地反映人类听感。

`utmos_score = self.utmos_model.score(audio_hat_16khz.unsqueeze(1)).mean()`  
这表示用 UTMOS 模型对合成音频进行自动评分，作为模型生成音频质量的一个评估指标。

### PESQ Score

**PESQ score**（Perceptual Evaluation of Speech Quality，感知语音质量评估）是一种**自动化的语音质量客观评价指标**，广泛用于语音合成、语音增强、语音编码等领域。

- **范围**：PESQ 分数通常在 1.0（极差）到 4.5（极好）之间，分数越高表示音频质量越接近原始参考音频。
- **原理**：PESQ 通过模拟人耳听觉模型，比较参考音频（原始）和待评估音频（合成/增强后），输出一个反映主观听感的分数。
- **用途**：常用于衡量语音合成、语音增强、降噪等模型的音频质量，越高越好。

在 experiment.py 的 `validation_step` 里，`pesq_score = pesq(16000, ref, deg, "wb", on_error=1)`  
这表示用 PESQ 工具对合成音频和真实音频进行质量对比打分。


### 总结
- **periodicity_loss**：周期性分数的RMSE，衡量周期性结构相似度。
- **pitch_loss**：清音帧音高的RMSE（cents），衡量音高准确性。
- **f1**：清音/非清音判别的一致性，衡量清浊音结构相似度。
- **UTMOS** 分数就是用深度学习模型自动预测的“听感质量分”，数值越高，音频越自然、越接近真实人声。
- **PESQ score** 是衡量合成音频与真实音频听感接近程度的国际标准客观分数，分数越高，音频质量越好。
