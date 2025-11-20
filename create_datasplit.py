from dotenv import load_dotenv
import os
import sys
from torch import Tensor

from decoder.titw_dataset import TITW

load_dotenv()

import torchaudio
import pathlib
import enum
from tqdm import tqdm


class LIBRITTSStructure(enum.Enum):
    WAVEFORM = 0
    SAMPLE_RATE = 1
    ORIGINAL_SCRIPT = 2
    NORMALIZED_SCRIPT = 3
    ID = 4
    SPEAKER_ID = 5
    CHAPTER_ID = 6


class LIBRITTSSubset(enum.Enum):
    TRAIN_CLEAN_100 = "train-clean-100"
    TRAIN_CLEAN_360 = "train-clean-360"
    TRAIN_OTHER_500 = "train-other-500"
    DEV_CLEAN = "dev-clean"
    DEV_OTHER = "dev-other"
    TEST_CLEAN = "test-clean"
    TEST_OTHER = "test-other"
    
class TITWStructure(enum.Enum):
    WAVEFORM = 0
    SAMPLE_RATE = 1
    UTT_ID = 2
    TEXT = 3
    SPEAKER_ID = 4
    
class TITWSubset(enum.Enum):
    EASY_TEST = "easy_test"
    EASY_DEV = "easy_dev"
    HARD_TEST = "hard_test"
    HARD_DEV = "hard_dev"
    


def print_dataset_info(dataset, index, dataset_structure):
    """
    打印数据集指定索引的结构化信息

    Args:
        dataset: LIBRITTS 数据集对象
        index: 数据索引
        dataset_structure: 数据集结构枚举类
            - LIBRITTSStructure
            - TITWStructure
    """
    
    print(f"Dataset length: {len(dataset)}")
    print(f"=== Dataset Info for Index {index} ===")
    print(f"Length of each example: {len(dataset[index])}")
    
    for field in dataset_structure:
        field_name = field.name
        field_value = dataset[index][field.value]
        
        if isinstance(field_value, Tensor):
            print(f"{field_name}: shape={field_value.shape}")
        else:
            print(f"{field_name}: {field_value}")
    print("="*40)


def get_wav_path(dataset, index, base_path, url):
    """
    根据数据集索引获取对应的 WAV 文件路径

    Args:
        dataset: LIBRITTS 数据集对象 or TITW 数据集对象
        index: 数据索引
        base_path: 数据集根路径
        url: 数据集子集名称 (如 "test-clean")

    Returns:
        pathlib.Path: WAV 文件的完整路径
    """

    # /mnt/nas3_workspace/spmiData/LibriTTS/test-clean/1089/134686/1089_134686_000001_000001.wav
    if isinstance(dataset, TITW):
        # /mnt/nas3_workspace/spmiData/TITW/titw_easy_audio/test/id10270-5r0dWxy17C8-00001-001.wav
        return dataset.data[index]["audio"]
    elif isinstance(dataset, torchaudio.datasets.LIBRITTS):
        return (
            pathlib.Path(base_path)
            / "LibriTTS"
            / url
            / str(dataset[index][LIBRITTSStructure.ID.value])  # ID
            / str(dataset[index][LIBRITTSStructure.SPEAKER_ID.value])  # Speaker ID
            / (
                str(dataset[index][LIBRITTSStructure.CHAPTER_ID.value]) + ".wav"
            )  # Chapter ID.wav
        )
        
    else:
        raise TypeError("Unsupported dataset type")


def generate_wav_paths_txt(
    subset_enum, base_path, max_len=None, output_dir="./data/test", dataset_type="LIBRITTS"
):
    """
    为指定的数据集子集生成 WAV 路径文件

    Args:
        subset_enum: LIBRITTSSubset 枚举值
        base_path: 数据集根路径
        max_len: 最大处理样本数，None 表示处理全部样本
        output_dir: 输出目录
        dataset_type: 数据集类型 ("LIBRITTS" 或 "TITW")
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 加载数据集
    url = subset_enum.value
    print(f"Processing {url}...")

    try:
        if dataset_type == "TITW":
            from decoder.titw_dataset import TITW
            dataset = TITW(root=base_path, url=url, sampling_rate=None)
        else:
            dataset = torchaudio.datasets.LIBRITTS(root=base_path, url=url, download=False)
    except Exception as e:
        print(f"Error loading {url}: {e}")
        return

    # 生成输出文件路径
    output_file = pathlib.Path(output_dir) / f"{url}.txt"

    # 写入 WAV 路径
    actual_len = min(len(dataset), max_len) if max_len is not None else len(dataset)

    with open(output_file, "w", encoding="utf-8") as f:
        for i in tqdm(range(actual_len), desc=f"Processing {url}", unit="files"):
            wav_path = get_wav_path(dataset, i, base_path, url)
            f.write(str(wav_path) + "\n")

    print(f"Generated {output_file} with {actual_len} paths")


def test():
    # LibriTTS 数据集测试
    url = LIBRITTSSubset.TEST_CLEAN.value
    dataset = torchaudio.datasets.LIBRITTS(
        root=os.getenv("DATASET_PATH"), url=url, download=False
    )
    index = 423
    print_dataset_info(dataset, index, LIBRITTSStructure)
   
    # TITW 数据集测试
    # url = TITWSubset.EASY_TEST.value
    # dataset = TITW(root=os.getenv("DATASET_PATH"), url=url)
    # print_dataset_info(dataset, 0, TITWStructure)
    
    
    wav, sample_rate, *rest = dataset[index]
    print(f"WAV shape: {wav.shape}, Sample Rate: {sample_rate}")
    
    # save the audio
    torchaudio.save(f"test_{sample_rate}.wav", wav, sample_rate)
    
    # Resample to 24kHz
    target_sample_rate = 24_000
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=target_sample_rate
        )
        wav = resampler(wav)
        print(f"Resampled WAV shape: {wav.shape}, New Sample Rate: {target_sample_rate}")
        torchaudio.save(f"test_{target_sample_rate}.wav", wav, target_sample_rate)
    
    # Dataset length: 4837
    # === Dataset Info for Index 0 ===
    # Length of each example: 7
    # Waveform shape: torch.Size([1, 307920])
    # Sample Rate: 24000
    # Original script: He hoped there would be stew for dinner, turnips and carrots and bruised potatoes and fat mutton pieces to be ladled out in thick peppered flour-fattened sauce. Stuff it into you, his belly counselled him.
    # Normalized script: He hoped there would be stew for dinner, turnips and carrots and bruised potatoes and fat mutton pieces to be ladled out in thick peppered flour fattened sauce. Stuff it into you, his belly counselled him.
    # ID: 1089
    # Speaker ID: 134686
    # Chapter ID: 1089_134686_000001_000001
    # ========================================
    # real_path = get_wav_path(dataset, 0, os.getenv("DATASET_PATH"), url)
    # print(f"WAV 文件路径: {real_path}")
    # WAV 文件路径: /mnt/nas3_workspace/spmiData/LibriTTS/test-clean/1089/134686/1089_134686_000001_000001.wav

def cal_mean_duration(dataset, dataset_structure, max_len=1000):
    total_duration = 0.0
    
    # randomly sample 1000 items if dataset is too large
    if len(dataset) > max_len:
        import numpy as np
        indices = np.random.choice(len(dataset), size=max_len, replace=False)
        for i in indices:
            waveform = dataset[i][dataset_structure.WAVEFORM.value]
            sample_rate = dataset[i][dataset_structure.SAMPLE_RATE.value]
            duration = waveform.size(1) / sample_rate
            total_duration += duration
        mean_duration = total_duration / max_len
    else:
        for i in range(len(dataset)):
            waveform = dataset[i][dataset_structure.WAVEFORM.value]
            sample_rate = dataset[i][dataset_structure.SAMPLE_RATE.value]
            duration = waveform.size(1) / sample_rate
            total_duration += duration
        mean_duration = total_duration / len(dataset)
    return mean_duration

if __name__ == "__main__":
    # 打印数据集信息示例
    # test()
    # 生成指定子集的 WAV 路径文件
    # GENERATE_PATH_DICT = {
    #     LIBRITTSSubset.TRAIN_CLEAN_100: "./data/train",
    #     # LIBRITTSSubset.TRAIN_CLEAN_360: "./data/train",
    #     # LIBRITTSSubset.TRAIN_OTHER_500: "./data/train",
    #     LIBRITTSSubset.DEV_CLEAN: "./data/dev",
    #     # LIBRITTSSubset.DEV_OTHER: "./data/dev",
    #     LIBRITTSSubset.TEST_CLEAN: "./data/test",
    #     # LIBRITTSSubset.TEST_OTHER: "./data/test",
    # }
    # max_len = 4000
    # for subset, out_dir in GENERATE_PATH_DICT.items():
    #     generate_wav_paths_txt(subset, os.getenv("DATASET_PATH"), max_len, out_dir)

    
    # GENERTE_PATH_DICT = {
    #     # TITWSubset.EASY_TEST: "./data/titw_test",
    #     # TITWSubset.EASY_DEV: "./data/titw_dev",
    #     TITWSubset.HARD_TEST: "./data/titw_test",
    #     TITWSubset.HARD_DEV: "./data/titw_dev",
    # }
    
    # max_len = None
    # for subset, out_dir in GENERTE_PATH_DICT.items():
    #     generate_wav_paths_txt(
    #         subset_enum=subset,
    #         base_path=os.getenv("DATASET_PATH"),
    #         max_len=max_len,
    #         output_dir=out_dir,
    #         dataset_type="TITW"
    #     )
    # test()
    # 计算数据集平均时长
    url = LIBRITTSSubset.TEST_CLEAN.value
    dataset = torchaudio.datasets.LIBRITTS(
        root=os.getenv("DATASET_PATH"), url=url, download=False
    )
    mean_duration = cal_mean_duration(dataset, LIBRITTSStructure)
    print(f"Mean duration of {url}: {mean_duration:.2f} seconds")
    
    url = TITWSubset.EASY_TEST.value
    dataset = TITW(root=os.getenv("DATASET_PATH"), url=url, sampling_rate=None)
    mean_duration = cal_mean_duration(dataset, TITWStructure)
    print(f"Mean duration of {url}: {mean_duration:.2f} seconds")
    