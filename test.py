import pathlib
from dotenv import load_dotenv
import os

load_dotenv()
from datasets import load_dataset, Audio, Features, Value, ClassLabel

DATASET_PATH = os.getenv("DATASET_PATH", "/mnt/nas3_workspace/spmiData/")


# ---------------------- 4. 测试加载（验证数据完整性） ----------------------
if __name__ == "__main__":
   
    file_path = os.path.join(DATASET_PATH, "TITW/titw_easy_test.jsonl")
    assert pathlib.Path(file_path).exists(), f"{file_path} 不存在，请检查 DATASET_PATH 是否正确"
    
    features = Features({
        "audio": Value("string"),  # Audio(sampling_rate=16000)
        "text": Value("string"),
        "utt_id": Value("string"),
        "speaker": Value("string"),
        # "emotion": ClassLabel(names=["angry", "happy", "neutral", "sad"]),
    })
    ds = load_dataset(
        "json",
        data_files={
            "easy_test": file_path,
        },
        features=features
    )
    ds.cast_column("audio", Audio())
    print(ds)
    print(ds["easy_test"][0])
    print(ds["easy_test"][0]["audio"].keys())  # 现在audio是字典
    # print(ds["easy_test"][0]["audio"]["array"].shape)
    # print(ds["easy_test"][0]["audio"]["sampling_rate"])
    # print(ds["easy_test"][0]["text"])
    # print(ds["easy_test"][0]["utt_id"])
    # print(ds["easy_test"][0]["speaker"])