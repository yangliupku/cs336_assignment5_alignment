import pathlib
from datasets import load_dataset
import json
import random

DATASETS_PATH = (pathlib.Path(__file__).resolve().parent.parent) / "data"


def create_math_dataset(random_seed=0):
    """split MATH dataset to validation (5k) and sft (7k)"""
    random.seed(random_seed)
    validation_fname = DATASETS_PATH / "MATH" / "validation.jsonl"
    sft_fname = DATASETS_PATH / "MATH" / "sft.jsonl"
    ds = list(load_dataset("qwedsacf/competition_math")["train"])

    random.shuffle(ds)
    validation_samples = ds[:5000]
    sft_samples = ds[5000:]
    print("validation_samples:", len(validation_samples))
    print("sft_samples:", len(sft_samples))
    with open(validation_fname, "w") as f:
        for i in range(len(validation_samples)):
            f.write(json.dumps(validation_samples[i]) + "\n")
    with open(sft_fname, "w") as f:
        for i in range(len(sft_samples)):
            f.write(json.dumps(sft_samples[i]) + "\n")


def load_jsonl(file_name):
    results = []
    with open(file_name, "r") as f:
        for line in f:
            results.append(json.loads(line.strip()))
    return results


def save_jsonl(data, filename):
    with open(filename, "w") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")


if __name__ == "__main__":
    # create_math_dataset()
    ds = load_jsonl(DATASETS_PATH / "MATH" / "validation.jsonl")
    print(len(ds))
    print(ds[45])
