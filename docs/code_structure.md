# Code Structure

主要介绍一下项目的代码结构。由于是标准的 encoder-vq-decoder 结构，代码结构也比较清晰。
阅读的顺序如下：

1. `encoder` 部分：论文中提到这里参考自 SoundStream 的设计，主要是一个 1D-CNN 的结构。因此略看。也可以将SoundStream的代码作为参考。
2. `vq` 部分：是一个单码本，可能需要和多码本对比
3. `decoder` 部分：可能是重点看的部分，增加了 attention 机制

