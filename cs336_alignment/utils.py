import pathlib
from datasets import load_dataset
import json
import random

DATASETS_PATH = (pathlib.Path(__file__).resolve().parent.parent) / "data"


def create_math_dataset(random_seed=0):
    """split MATH dataset to validation (5k) and sft (7k)"""
    random.seed(random_seed)
    validation_fname = DATASETS_PATH / "MATH" / "validation.jsonl"
    training_fname = DATASETS_PATH / "MATH" / "train.jsonl"
    ds = list(load_dataset("qwedsacf/competition_math")["train"])

    random.shuffle(ds)
    validation_samples = ds[:5000]
    training_samples = ds[5000:]
    print("validation_samples:", len(validation_samples))
    print("training_samples:", len(training_samples))
    save_jsonl(validation_samples, validation_fname)
    save_jsonl(training_samples, training_fname)


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
