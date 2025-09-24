from transformers import AutoTokenizer, PreTrainedTokenizer
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
    x = torch.rand(3, 5, 8)
    print(compute_entropy(x))
