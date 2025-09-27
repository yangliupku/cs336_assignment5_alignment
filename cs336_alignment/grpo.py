import random
import torch
import pathlib
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from vllm import LLM, SamplingParams
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.utils import load_jsonl
from cs336_alignment.sft_helpers import (
    tokentize_prompt_and_output,
    get_response_log_probs,
    get_old_policy_log_probs_in_batches,
)
from cs336_alignment.grpo_helpers import (
    compute_group_normalized_rewards,
    grpo_microbatch_train_step,
)
from unittest.mock import patch
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

DATASETS_PATH = (pathlib.Path(__file__).resolve().parent.parent) / "data" / "MATH"
MODEL_PATH = (pathlib.Path(__file__).resolve().parent.parent) / "models" / "Qwen2.5-Math-1.5"

ROLLOUT_BATCH_SIZE = 256
TRAIN_BATCH_SIZE = 256
EPOCHS_PER_RLLOUT_BATCH = 4
# ROLLOUT_BATCH_SIZE / TRAIN_BATCH_SIZE * EPOCH_PER_ROLLOUT = number of gradient updats per rollout
GROUP_SIZE = 8
GRADIENT_ACC_STEPS = 128
LR = 1e-5
N_GRPO_STEPS = 200

MICRO_TRAIN_BATCH_SIZE = TRAIN_BATCH_SIZE // GRADIENT_ACC_STEPS
N_PROMPTS_PER_ROLLOUT_BATCH = ROLLOUT_BATCH_SIZE // GROUP_SIZE
N_MICRO_BATCHES_PER_ROLLOUT_BATCH = ROLLOUT_BATCH_SIZE // MICRO_TRAIN_BATCH_SIZE

NORMALIZE_BY_STD = True
# LOSS_TYPE = "reinforce_with_baseline"
LOSS_TYPE = "grpo_clip"
CLIPRANGE = 0.2
MAX_GRAD_NORM = 1.0

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


set_all_seed()
model, tokenizer = load_base_model()
llm = init_vllm()
opt = AdamW(model.parameters(), lr=LR, weight_decay=0, betas=(0.9, 0.95))

train_ds = load_jsonl(DATASETS_PATH / "train.jsonl")
validation_ds = load_jsonl(DATASETS_PATH / "validation.jsonl")
validation_ds = validation_ds[:1024]
random.shuffle(train_ds)

for grpo_step in range(N_GRPO_STEPS):
    rollout_sample_data = train_ds[
        grpo_step * N_PROMPTS_PER_ROLLOUT_BATCH : (1 + grpo_step) * N_PROMPTS_PER_ROLLOUT_BATCH
    ]
    rollout_repeated_questions = [
        s["problem"] for s in rollout_sample_data for _ in range(GROUP_SIZE)
    ]
    rollout_repeated_solutions = [
        s["solution"] for s in rollout_sample_data for _ in range(GROUP_SIZE)
    ]
    rollout_repeated_prompts = [get_prompt(q) for q in rollout_repeated_questions]
    load_policy_into_vllm_instance(model, llm)
    validation_acc = get_validation_accuracy(validation_ds, llm)
    print("---------> validation acc:", validation_acc)
    llm_response = llm.generate(rollout_repeated_prompts, sampling_params)
    rollout_responses = [r.outputs[0].text for r in llm_response]
    advantages, raw_rewards, _ = compute_group_normalized_rewards(
        reward_fn=r1_zero_reward_fn,
        rollout_responses=rollout_responses,
        repeated_ground_truths=rollout_repeated_solutions,
        group_size=GROUP_SIZE,
        advantage_eps=1e-6,
        normalize_by_std=NORMALIZE_BY_STD,
    )
    print("---------> total raw rewards:", torch.sum(raw_rewards))

    tokenized_rollout_batch = tokentize_prompt_and_output(
        rollout_repeated_prompts, rollout_responses, tokenizer
    )
    rollout_batch_input_ids = tokenized_rollout_batch["input_ids"]
    rollout_batch_labels = tokenized_rollout_batch["labels"]
    rollout_batch_response_masks = tokenized_rollout_batch["response_mask"]
    old_log_prob_res = get_old_policy_log_probs_in_batches(model, rollout_batch_input_ids, rollout_batch_labels)
    rollout_batch_old_log_probs = old_log_prob_res["log_probs"]
    dataset = TensorDataset(
        rollout_batch_input_ids,
        rollout_batch_labels,
        rollout_batch_response_masks,
        advantages.unsqueeze(-1),
        raw_rewards.unsqueeze(-1),
        rollout_batch_old_log_probs,
    )
    dataloader = DataLoader(dataset, batch_size=MICRO_TRAIN_BATCH_SIZE, shuffle=True)
    for epoch in range(EPOCHS_PER_RLLOUT_BATCH):
        opt.zero_grad()
        for idx, (
            batch_inputs,
            batch_labels,
            batch_masks,
            batch_advantages,
            batch_raw_rewards,
            batch_old_log_probs,
        ) in enumerate(dataloader):
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)
            batch_masks = batch_masks.to(device)
            batch_advantages = batch_advantages.to(device)
            batch_raw_rewards = batch_raw_rewards.to(device)
            batch_old_log_probs = batch_old_log_probs.to(device)
            response_log_probs = get_response_log_probs(model, batch_inputs, batch_labels)
            policy_log_probs = response_log_probs["log_probs"]
            loss, _ = grpo_microbatch_train_step(
                policy_log_probs=policy_log_probs,
                response_mask=batch_masks,
                gradient_accumulation_steps=GRADIENT_ACC_STEPS,
                loss_type=LOSS_TYPE,
                raw_rewards=batch_raw_rewards,
                advantages=batch_advantages,
                old_log_prob=batch_old_log_probs,
                cliprange=CLIPRANGE,
            )
            if (idx + 1) % GRADIENT_ACC_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                opt.step()
                opt.zero_grad()
