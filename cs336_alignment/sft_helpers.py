from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedModel, AutoModelForCausalLM
import pathlib
import torch
from torch.nn.utils.rnn import pad_sequence

MODEL_PATH = (pathlib.Path(__file__).resolve().parent.parent) / "models" / "Qwen2.5-Math-1.5"


def tokentize_prompt_and_output(
    prompt_strs: list[str], output_strs: list[str], tokenizer: PreTrainedTokenizer
):
    assert len(prompt_strs) == len(output_strs)
    tokenized_prompts = tokenizer(prompt_strs)["input_ids"]
    tokenized_outputs = tokenizer(output_strs)["input_ids"]
    token_tensors = []
    mask_tensors = []
    for i in range(len(tokenized_prompts)):
        token_tensors.append(
            torch.tensor(tokenized_prompts[i] + tokenized_outputs[i], dtype=torch.long)
        )
        mask_tensors.append(
            torch.tensor(
                [False] * len(tokenized_prompts[i]) + [True] * len(tokenized_outputs[i]),
                dtype=torch.bool,
            )
        )
    padded_tensors = pad_sequence(
        token_tensors, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    padded_masks = pad_sequence(mask_tensors, batch_first=True, padding_value=False)
    inputs = padded_tensors[:, :-1]
    labels = padded_tensors[:, 1:]
    masks = padded_masks[:, 1:]
    return {"input_ids": inputs, "labels": labels, "response_mask": masks}


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    logp = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    p = torch.exp(logp)
    return torch.sum(-p * logp, dim=-1)


def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    logits = model(input_ids).logits
    logp = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    log_probs = torch.gather(logp, -1, labels.unsqueeze(-1)).squeeze(-1)
    token_entropy = compute_entropy(logits) if return_token_entropy else None
    return {"log_probs": log_probs, "token_entropy": token_entropy}


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
    normalize_constant: float = 1.0,
):
    return torch.sum(tensor * mask, dim=dim) / normalize_constant


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss = masked_normalize(
        policy_log_probs,
        response_mask,
        dim=-1,
        normalize_constant=normalize_constant,
    )
    loss = -1.0 * torch.mean(loss) / gradient_accumulation_steps
    loss.backward()
    return (loss, {})


if __name__ == "__main__":
    # tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    # print(tokenizer.pad_token_id)
    # prompt_strs = [
    #     "Hello, world!",
    #     "This is a test.",
    #     "This is another test.",
    # ]
    # output_strs = [
    #     "Hello, world!",
    #     "This is a test.",
    #     "This is another test.",
    # ]
    # x = tokentize_prompt_and_output(prompt_strs, output_strs, tokenizer)
    # print(x)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
    )
    input_ids = torch.randint(0, 1024, (2, 3))
    labels = torch.randint(0, 10000, (2, 3))
    x = get_response_log_probs(model, input_ids, labels, return_token_entropy=True)
    print(x)
