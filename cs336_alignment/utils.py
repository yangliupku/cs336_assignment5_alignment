import pathlib
from datasets import load_dataset
import json
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel
from vllm import LLM, SamplingParams
from unittest.mock import patch
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn


DATASETS_PATH = (pathlib.Path(__file__).resolve().parent.parent) / "data"
MODEL_PATH = (pathlib.Path(__file__).resolve().parent.parent) / "models" / "Qwen2.5-Math-1.5"


def get_prompt(question):
    return f"""A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
  User: {question}
  Assistant: <think>"""


sampling_params = SamplingParams(
    temperature=1.0,
    top_p=1.0,
    max_tokens=1024,
    stop=["</answer>"],
    include_stop_str_in_output=True,
    min_tokens=4,
)


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


def load_math_validation_set():
    return load_jsonl(DATASETS_PATH / "MATH" / "validation.jsonl")


def load_base_model(device):
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    return (model, tokenizer)


def init_vllm():
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None,
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=str(MODEL_PATH),
            device=torch.device("cuda:1"),
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=0.85,
        )


def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


def get_validation_accuracy(ds, llm):
    prompts = [get_prompt(example["problem"]) for example in ds]
    results = []
    responses = llm.generate(prompts, sampling_params)
    for i in range(len(ds)):
        solution = ds[i]["solution"]
        model_response = responses[i].outputs[0].text
        r1_reward = r1_zero_reward_fn(model_response, solution)
        results.append(r1_reward["reward"])
    return sum(results) / len(results)


if __name__ == "__main__":
    # create_math_dataset()
    ds = load_jsonl(DATASETS_PATH / "MATH" / "validation.jsonl")
    print(len(ds))
    print(ds[45])
