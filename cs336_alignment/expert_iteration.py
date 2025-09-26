import random
import torch
import pathlib
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel
from vllm.model_executer import set_random_seed as vllm_set_random_seed
from vllm import LLM, SamplingParams
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.utils import load_jsonl
from cs336_alignment.sft_helpers import (
    tokentize_prompt_and_output,
    get_response_log_probs,
    sft_microbatch_train_step,
)
from unittest.mock import patch
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

DATASETS_PATH = (pathlib.Path(__file__).resolve().parent.parent) / "data" / "MATH"
MODEL_PATH = (pathlib.Path(__file__).resolve().parent.parent) / "models" / "Qwen2.5-Math-1.5"

EI_SAMPLE_SIZE = 32  # number of questions to sample in each EI step
GROUP_SIZE = 5  # number of responses for each question
NUM_EI_STEPS = 100
LR = 1e-3
GRAD_STEPS = 4

device = torch.device("cuda:0")

sampling_params = SamplingParams(
    temperature=1.0,
    top_p=1.0,
    max_tokens=1024,
    stop=["</answer>"],
    include_stop_str_in_output=True,
    min_tokens=4,
)


def get_prompt(question):
    return f"""A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
  User: {question}
  Assistant: <think>"""


def set_all_seed():
    random.seed(0)
    vllm_set_random_seed(0)


def load_base_model():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_type=torch.bfloat16, attention_implementation="flash_attention_2"
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


def get_correct_prompt_and_output(prompts: list[str], solutions: list[str], llm: LLM):
    responses = llm.generate(prompts, sampling_params)
    correct_prompts = []
    correct_outputs = []
    for i in range(len(prompts)):
        model_response = responses[i].outputs[0].text
        reward = r1_zero_reward_fn(model_response, solutions[i])
        if reward["r1_reward"] == 1:
            correct_outputs.append(model_response)
            correct_prompts.append(prompts[i])
    return correct_prompts, correct_outputs


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


set_all_seed()
model, tokenizer = load_base_model()
llm = init_vllm()
opt = AdamW(model.parameters(), lr=LR)

train_ds = load_jsonl(DATASETS_PATH / "train.jsonl")
validation_ds = load_jsonl(DATASETS_PATH / "validation.jsonl")
validation_ds = validation_ds[:1024]
random.shuffle(train_ds)

for ei_step_i in range(NUM_EI_STEPS):
    print("----------> EI iteration:", ei_step_i)
    ei_sample_data = train_ds[ei_step_i * EI_SAMPLE_SIZE : (1 + ei_step_i) * EI_SAMPLE_SIZE]
    ei_repeated_questions = [s["problem"] for s in ei_sample_data for _ in range(GROUP_SIZE)]
    ei_repeated_solutions = [s["solution"] for s in ei_sample_data for _ in range(GROUP_SIZE)]
    ei_repeated_prompts = [get_prompt(q) for q in ei_repeated_questions]
    load_policy_into_vllm_instance(model, llm)
    validation_acc = get_validation_accuracy(llm, validation_ds)
    print("---------> validation acc:", validation_acc)
    training_prompts, training_outputs = get_correct_prompt_and_output(
        ei_repeated_prompts, ei_repeated_solutions, llm
    )
    print("--------> number of training samples:", len(training_outputs))
    tokenized_training_data = tokentize_prompt_and_output(
        training_prompts, training_outputs, tokenizer
    )
    dataset = TensorDataset(
        tokenized_training_data["input_ids"],
        tokenized_training_data["labels"],
        tokenized_training_data["response_mask"],
    )
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    opt.zero_grad()

    for idx, (batch_inputs, batch_labels, batch_masks) in enumerate(dataloader):
        batch_inputs = batch_inputs.to(device)
        bach_labels = batch_labels.to(device)
        batch_masks = batch_masks.to(device)
        response_log_probs = get_response_log_probs(model, batch_inputs, batch_labels)
        loss, _ = sft_microbatch_train_step(
            response_log_probs["log_probs"],
            batch_masks,
            gradient_accumulation_steps=GRAD_STEPS,
        )
        if (idx + 1) % GRAD_STEPS == 0:
            opt.step()
            opt.zero_grad()
