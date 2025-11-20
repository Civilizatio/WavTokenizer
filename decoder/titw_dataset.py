# Import necessary libraries
import json
import os
from pathlib import Path
import librosa
from typing import Tuple, Union
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset

FOLDER_IN_ARCHIVE = "TITW"

VALID_SPLITS = {
    "easy_test": "titw_easy_test.jsonl",
    "easy_dev": "titw_easy_dev.jsonl",
    "hard_test": "titw_hard_test.jsonl",
    "hard_dev": "titw_hard_dev.jsonl",
}


class TITW(Dataset):
    """TITW dataset class, supports automatic loading of multiple subsets and sample access"""

    def __init__(self, root: Union[str, Path], url: str, folder: str = FOLDER_IN_ARCHIVE, sampling_rate: int = None):
        """
        Initialize the TITW dataset
        :param data_dir: Root directory of the dataset
        :param splits: Dictionary of subset names and file paths, e.g., {"easy_dev": "TITW/titw_easy_dev.jsonl"}
        :param sampling_rate: Audio sampling rate, None means using the original sampling rate
        """
        root = os.fspath(root)

        if url not in VALID_SPLITS.keys():
            raise ValueError(
                f"Invalid split name: {url}, valid splits are: {VALID_SPLITS.keys()}"
            )

        self.url = url
        self.root = os.path.join(root, folder)

        self.jsonl_path = os.path.join(self.root, VALID_SPLITS[self.url])
        if not Path(self.jsonl_path).exists():
            raise FileNotFoundError(f"File does not exist: {self.jsonl_path}")

        self.data = self._load_jsonl(self.jsonl_path)
        self.sampling_rate = sampling_rate

    def _load_jsonl(self, file_path):
        """Load a JSONL file"""
        if not Path(file_path).exists():
            raise FileNotFoundError(f"File does not exist: {file_path}")

        data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data

    def __len__(self):
        """Get the number of samples in the dataset"""
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Tensor, int, str, str, str]:
        """Get a sample from the dataset

        Args:
            idx (int): Index of the sample to retrieve

        Returns:
            Tuple[Tensor, int, str, str, str]: A tuple containing:
                - audio (Tensor): The audio waveform tensor
                - sampling_rate (int): The sampling rate of the audio
                - utt_id (str): The utterance ID
                - text (str): The transcription text
                - speaker (str): The speaker ID
        """
        if idx < 0 or idx >= len(self.data):
            raise IndexError(f"Index out of range: {idx}")

        sample = self.data[idx].copy()
        audio_path = sample["audio"]
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file does not exist: {audio_path}")

        # Load audio file
        waveform, sr = torchaudio.load(audio_path)

        # Resample if a target sampling rate is specified
        if self.sampling_rate is not None and sr != self.sampling_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=self.sampling_rate
            )
            waveform = resampler(waveform)
            sr = self.sampling_rate

        return waveform, sr, sample["utt_id"], sample["text"], sample["speaker"]

    def __repr__(self):
        """Print dataset information"""
        return (
            f"TITW Dataset\n"
            f"Root directory: {self.root}\n"
            f"Subset: {self.url}\n"
            f"Number of samples: {len(self)}\n"
            f"Sampling rate: {self.sampling_rate if self.sampling_rate else 'Original'}"
        )


# Example usage
if __name__ == "__main__":
    # Configuration parameters
    DATASET_PATH = "/mnt/nas3_workspace/spmiData/TITW"  # Root directory of the dataset
    SAMPLING_RATE = (
        16000  # Target sampling rate, None means using the original sampling rate
    )

    # Define subset names and corresponding file paths
    url = "easy_test"

    # Initialize the TITW dataset (automatically load all subsets)
    titw = TITW(root=DATASET_PATH, url=url, sampling_rate=SAMPLING_RATE)

    # View dataset information
    print(titw)

    # Get a sample from a specific subset
    waveform, sr, utt_id, text, speaker = titw[0]

    print("\nFirst sample information:")
    print(f"utt_id: {utt_id}")
    print(f"audio type: {type(waveform)}")
    print(f"Audio array shape: {waveform.shape}")
    print(f"Sampling rate: {sr}")
    print(f"text: {text}")
    print(f"speaker: {speaker}")
