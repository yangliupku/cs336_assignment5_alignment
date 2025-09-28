from transformers import PreTrainedTokenizer, PreTrainedModel, AutoModelForCausalLM
import pathlib
import torch
from torch.nn.utils.rnn import pad_sequence
from jaxtyping import Float, Int, Bool
from typing import Callable, Literal

MODEL_PATH = (pathlib.Path(__file__).resolve().parent.parent) / "models" / "Qwen2.5-Math-1.5"


def tokentize_prompt_and_output(
    prompt_strs: list[str], output_strs: list[str], tokenizer: PreTrainedTokenizer
):
    """Tokenize the prompt and output strings, and construct a mask that is 1
    for the response tokens and 0 for other tokens (prompt or padding).

    Args:
        prompt_strs: list[str], the prompt strings.
        output_strs: list[str], the output strings.
        tokenizer: PreTrainedTokenizer, the tokenizer to use.

    Returns:
        dict[str, torch.Tensor]:
            "input_ids": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                the tokenized prompt and output strings, with the final token sliced off.
            "labels": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                shifted input_ids (i.e., the input_ids without the first token).
            "response_mask": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                a mask on the response tokens in `labels`.
    """
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


def compute_entropy(
    logits: Float[torch.Tensor, "batch_size seq_len vocab"],
) -> Float[torch.Tensor, "batch_size seq_len"]:
    """Get the entropy of the logits (i.e., entropy of the final dimension)."""
    logp = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    p = torch.exp(logp)
    return torch.sum(-p * logp, dim=-1)


def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: Int[torch.Tensor, "batch_size seq_len"],
    labels: Int[torch.Tensor, "batch_size seq_len"],
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """Get the conditional log-probs of the response given the prompt,
        and optionally the entropy of the next token predictions.

    Args:
        model: PreTrainedModel, the model to score.
        input_ids: torch.Tensor of shape (batch_size, sequence_length):
            the tokenized prompt and output.
        labels: torch.Tensor of shape (batch_size, sequence_length):
            shifted input_ids.
        return_token_entropy: bool, whether to return the entropy of the
            next token predictions.

    Returns:
        dict[str, torch.Tensor]:
            "log_probs": torch.Tensor of shape (batch_size, sequence_length):
                the conditional log-probs of the response given the prompt.
                Note that we have not masked out the token indices corresponding
                to the prompt or padding; that is done in the train loop.
            "token_entropy": Optional[torch.Tensor] of shape (batch_size, sequence_length):
                the entropy of the next token predictions. As with the log-probs,
                we have not masked out the token indices corresponding to the prompt
                or padding; that is done in the train loop.
    """
    logits = model(input_ids).logits
    logp = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    log_probs = torch.gather(logp, -1, labels.unsqueeze(-1)).squeeze(-1)
    token_entropy = compute_entropy(logits) if return_token_entropy else None
    return {"log_probs": log_probs, "token_entropy": token_entropy}


def get_old_policy_log_probs_in_batches(
    model: PreTrainedModel,
    input_ids: Int[torch.Tensor, "batch_size seq_len"],
    labels: Int[torch.Tensor, "batch_size seq_len"],
    batch_size: int = 8,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """Get the conditional log-probs of the response given the prompt,
        and optionally the entropy of the next token predictions.
    This function runs the model inference in detach mode, should only be used for
    getting log prob for old policy model, where we don't want the gradient to flow through.

    Args:
        model: PreTrainedModel, the model to score.
        input_ids: torch.Tensor of shape (batch_size, sequence_length):
            the tokenized prompt and output.
        labels: torch.Tensor of shape (batch_size, sequence_length):
            shifted input_ids.
        return_token_entropy: bool, whether to return the entropy of the
            next token predictions.

    Returns:
        dict[str, torch.Tensor]:
            "log_probs": torch.Tensor of shape (batch_size, sequence_length):
                the conditional log-probs of the response given the prompt.
                Note that we have not masked out the token indices corresponding
                to the prompt or padding; that is done in the train loop.
            "token_entropy": Optional[torch.Tensor] of shape (batch_size, sequence_length):
                the entropy of the next token predictions. As with the log-probs,
                we have not masked out the token indices corresponding to the prompt
                or padding; that is done in the train loop.
    """
    all_log_probs = []
    all_token_entropy = []
    model_device = next(model.parameters()).device
    for i in range(0, input_ids.shape[0], batch_size):
        batch_inputs = input_ids[i : i + batch_size].to(model_device)
        batch_labels = labels[i : i + batch_size].to(model_device)
        with torch.no_grad():
            batch_logits = model(batch_inputs).logits
        batch_logp = batch_logits - torch.logsumexp(batch_logits, dim=-1, keepdim=True)
        batch_log_probs = torch.gather(batch_logp, -1, batch_labels.unsqueeze(-1)).squeeze(-1)
        batch_token_entropy = compute_entropy(batch_logits)
        all_log_probs.append(batch_log_probs.cpu())
        all_token_entropy.append(batch_token_entropy.cpu())

    log_probs = torch.concat(all_log_probs, dim=0)
    token_entropy = torch.concat(all_token_entropy, dim=0) if return_token_entropy else None
    return {"log_probs": log_probs, "token_entropy": token_entropy}


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
    normalize_constant: float = 1.0,
):
    """Sum over a dimension and normalize by a constant,
    considering only the elements with mask value 1.

    Args:
        tensor: torch.Tensor, the tensor to sum and normalize.
        mask: torch.Tensor, the mask. We only consider elements
            with mask value 1.
        dim: int | None, the dimension to sum along before
            normalization. If None, sum over all dimensions.
        normalize_constant: float, the constant to divide by
            for normalization.

    Returns:
        torch.Tensor, the normalized sum, where masked elements
            (mask=0) don't contribute to the sum.
    """
    return torch.sum(tensor * mask, dim=dim) / normalize_constant


def sft_microbatch_train_step(
    policy_log_probs: Float[torch.Tensor, "batch_size seq_len"],
    response_mask: Bool[torch.Tensor, "batch_size seq_len"],
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy gradient loss and backprop its gradients for a microbatch."""

    loss = masked_normalize(
        policy_log_probs,
        response_mask,
        dim=-1,
        normalize_constant=normalize_constant,
    )
    loss = -1.0 * torch.mean(loss) / gradient_accumulation_steps
    loss.backward()
    return (loss, {})


def compute_group_normalized_rewards(
    reward_fn: Callable,
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool = True,
):
    """
    Compute rewards for each group of rollout responses,
    normalized by the group size.

    For more on GRPO, see:
        DeepSeekMath: https://arxiv.org/abs/2402.03300
        DeepSeek-R1: https://arxiv.org/abs/2501.12948

    Args:
        reward_fn: Callable[[str, str], dict[str, float]],
            scores the rollout responses against the ground truths,
            producing a dict with keys
            "reward", "format_reward", and "answer_reward".
        rollout_responses: list[str], rollouts from the policy.
            The length of this list is
            `rollout_batch_size = n_prompts_per_rollout_batch * group_size`.
        repeated_ground_truths: list[str], the ground truths for the examples.
            The length of this list is `rollout_batch_size`,
            because the ground truth for each example is repeated `group_size` times.
        group_size: int, number of rollouts per group.
        advantage_eps: float, epsilon to avoid division by zero
            during group normalization.
        normalize_by_std: bool, whether to normalize the rewards by
            std(rewards).

    Returns:
        tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
            torch.Tensor of shape (rollout_batch_size,):
                group-normalized rewards for each rollout response.
            torch.Tensor of shape (rollout_batch_size,):
                raw rewards for each rollout response.
            dict[str, float]: metadata for the rewards of the rollout batch.
                You may choose what you wish to log here
                (some statistics of the rewards, etc.).
    """
    rewards = []
    rollout_batch_size = len(rollout_responses)
    n_prompts_per_rollout_batch = rollout_batch_size // group_size
    for i in range(rollout_batch_size):
        grade = reward_fn(rollout_responses[i], repeated_ground_truths[i])
        rewards.append(grade["reward"])
    raw_rewards = torch.tensor(rewards).view(n_prompts_per_rollout_batch, group_size)
    advantages = raw_rewards - torch.mean(raw_rewards, dim=-1, keepdim=True)
    if normalize_by_std:
        advantages = advantages / (torch.std(raw_rewards, dim=-1, keepdim=True) + advantage_eps)
    advantages = advantages.view(-1)
    raw_rewards = raw_rewards.view(-1)
    return (advantages, raw_rewards, {})


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: Float[torch.Tensor, "batch_size 1"],
    policy_log_probs: Float[torch.Tensor, "batch_size seq_len"],
):
    """Compute policy gradient loss using either raw rewards or advantages.

    Args:
        raw_rewards_or_advantages: torch.Tensor of shape (batch_size, 1):
            the raw rewards or advantages for each rollout response.
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length):
            the log-probs of the policy.

    Returns:
        torch.Tensor of shape (batch_size, sequence_length):
            the policy gradient per-token loss.
    """
    return -1.0 * raw_rewards_or_advantages * policy_log_probs


def compute_grpo_clip_loss(
    advantages: Float[torch.Tensor, "batch_size 1"],
    policy_log_probs: Float[torch.Tensor, "batch_size seq_len"],
    old_log_probs: Float[torch.Tensor, "batch_size seq_len"],
    cliprange: float,
):
    """Compute the GRPO-Clip loss.

    Args:
        advantages: torch.Tensor of shape (batch_size, 1):
            the advantages for each rollout response.
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length):
            the log-probs of the policy.
        old_log_probs: torch.Tensor of shape (batch_size, sequence_length):
            the log-probs of the old policy.
        cliprange: float, the clip range for the ratio.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            torch.Tensor of shape (batch_size, sequence_length):
                the GRPO-Clip per-token loss.
            dict[str, torch.Tensor]: metadata for the GRPO-Clip loss
                (used to compute clip fraction).
    """
    probs_ratio = torch.exp(policy_log_probs - old_log_probs)
    c = torch.clip(probs_ratio, 1 - cliprange, 1 + cliprange)
    loss = -1 * torch.minimum(advantages * probs_ratio, advantages * c)
    return (loss, {})


def compute_policy_gradient_loss(
    policy_log_probs: Float[torch.Tensor, "batch_size seq_len"],
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: Float[torch.Tensor, "batch_size 1"],
    advantages: Float[torch.Tensor, "batch_size 1"],
    old_log_prob: Float[torch.Tensor, "batch_size seq_len"],
    cliprange: float,
):
    """
    Wrapper that delegates to the appropriate policy gradient loss function above.
    """
    if loss_type == "no_baseline":
        loss = compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)
        return (loss, {})
    elif loss_type == "reinforce_with_baseline":
        loss = compute_naive_policy_gradient_loss(advantages, policy_log_probs)
        return (loss, {})
    elif loss_type == "grpo_clip":
        return compute_grpo_clip_loss(advantages, policy_log_probs, old_log_prob, cliprange)
    else:
        raise NotImplementedError


def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
):
    """Compute the mean of the tensor along a dimension,
    considering only the elements with mask value 1.

    Args:
        tensor: torch.Tensor, the tensor to compute the mean of.
        mask: torch.Tensor, the mask. We only take the mean over
            the elements with mask value 1.
        dim: int | None, the dimension to compute the mean along.
            If None, sum over all non-masked elements and average
            by their total count.

    Returns:
        torch.Tensor, the mean of the tensor along the specified
            dimension, considering only the elements with mask value 1.
    """
    return torch.sum(tensor * mask, dim=dim) / (torch.sum(mask, dim=dim))


def grpo_microbatch_train_step(
    policy_log_probs: Float[torch.Tensor, "batch_size seq_len"],
    response_mask: Int[torch.Tensor, "batch_size seq_len"],
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: Float[torch.Tensor, "batch_size 1"],
    advantages: Float[torch.Tensor, "batch_size 1"],
    old_log_prob: Float[torch.Tensor, "batch_size seq_len"],
    cliprange: float,
):
    """Compute the policy gradient loss and backprop its gradients for a microbatch.

    Args:
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length):
            the log-probs of the policy.
        response_mask: torch.Tensor of shape (batch_size, sequence_length):
            the mask for the response.
        gradient_accumulation_steps: int, the number of gradient accumulation steps.
        loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
            the type of loss function to use.
        raw_rewards: torch.Tensor | None, the raw rewards for each rollout response.
            Needed for loss_type="no_baseline".
        advantages: torch.Tensor | None, the advantages for each rollout response.
            Needed for loss_type in {"reinforce_with_baseline", "grpo_clip"}.
        old_log_probs: torch.Tensor | None, the log-probs of the old policy.
            Needed for loss_type="grpo_clip".
        cliprange: float | None, the clip range for the ratio.
            Needed for loss_type="grpo_clip".
        constant_normalize_factor: int | None, provided if we want to sum over
            the sequence dimension and normalize by this constant factor
            (as in Dr. GRPO).

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            the policy gradient loss and its metadata.
    """
    loss, metadata = compute_policy_gradient_loss(
        policy_log_probs,
        loss_type,
        raw_rewards,
        advantages,
        old_log_prob,
        cliprange,
    )
    loss = masked_mean(loss, response_mask) / gradient_accumulation_steps
    loss.backward()
    return (loss, metadata)


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
