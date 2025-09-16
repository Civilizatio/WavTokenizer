from dotenv import load_dotenv
import os

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


def print_dataset_info(dataset, index):
    """
    打印数据集指定索引的结构化信息

    Args:
        dataset: LIBRITTS 数据集对象
        index: 数据索引
    """
    print(f"Dataset length: {len(dataset)}")
    print(f"=== Dataset Info for Index {index} ===")
    print(f"Length of each example: {len(dataset[index])}")
    print(f"Waveform shape: {dataset[index][LIBRITTSStructure.WAVEFORM.value].shape}")
    print(f"Sample Rate: {dataset[index][LIBRITTSStructure.SAMPLE_RATE.value]}")
    print(f"Original script: {dataset[index][LIBRITTSStructure.ORIGINAL_SCRIPT.value]}")
    print(
        f"Normalized script: {dataset[index][LIBRITTSStructure.NORMALIZED_SCRIPT.value]}"
    )
    print(f"ID: {dataset[index][LIBRITTSStructure.ID.value]}")
    print(f"Speaker ID: {dataset[index][LIBRITTSStructure.SPEAKER_ID.value]}")
    print(f"Chapter ID: {dataset[index][LIBRITTSStructure.CHAPTER_ID.value]}")
    print("=" * 40)


def get_wav_path(dataset, index, base_path, url):
    """
    根据数据集索引获取对应的 WAV 文件路径

    Args:
        dataset: LIBRITTS 数据集对象
        index: 数据索引
        base_path: 数据集根路径
        url: 数据集子集名称 (如 "test-clean")

    Returns:
        pathlib.Path: WAV 文件的完整路径
    """

    # /mnt/nas3_workspace/spmiData/LibriTTS/test-clean/1089/134686/1089_134686_000001_000001.wav
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


def generate_wav_paths_txt(
    subset_enum, base_path, max_len=None, output_dir="./data/test"
):
    """
    为指定的数据集子集生成 WAV 路径文件

    Args:
        subset_enum: LIBRITTSSubset 枚举值
        base_path: 数据集根路径
        output_dir: 输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 加载数据集
    url = subset_enum.value
    print(f"Processing {url}...")

    try:
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
    url = LIBRITTSSubset.TEST_CLEAN.value
    dataset = torchaudio.datasets.LIBRITTS(
        root=os.getenv("DATASET_PATH"), url=url, download=False
    )
    print_dataset_info(dataset, 0)
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
    real_path = get_wav_path(dataset, 0, os.getenv("DATASET_PATH"), url)
    print(f"WAV 文件路径: {real_path}")
    # WAV 文件路径: /mnt/nas3_workspace/spmiData/LibriTTS/test-clean/1089/134686/1089_134686_000001_000001.wav

if __name__ == "__main__":
    # 打印数据集信息示例
    test()
    # 生成指定子集的 WAV 路径文件
    GENERATE_PATH_DICT = {
        LIBRITTSSubset.TRAIN_CLEAN_100: "./data/train",
        # LIBRITTSSubset.TRAIN_CLEAN_360: "./data/train",
        # LIBRITTSSubset.TRAIN_OTHER_500: "./data/train",
        LIBRITTSSubset.DEV_CLEAN: "./data/dev",
        # LIBRITTSSubset.DEV_OTHER: "./data/dev",
        LIBRITTSSubset.TEST_CLEAN: "./data/test",
        # LIBRITTSSubset.TEST_OTHER: "./data/test",
    }
    max_len = 4000
    for subset, out_dir in GENERATE_PATH_DICT.items():
        generate_wav_paths_txt(subset, os.getenv("DATASET_PATH"), max_len, out_dir)

    
