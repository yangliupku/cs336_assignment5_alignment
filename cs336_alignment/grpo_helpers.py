from typing import Literal
import torch


def compute_group_normalized_rewards(
    reward_fn,
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool = True,
):
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
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
):
    return -1.0 * raw_rewards_or_advantages * policy_log_probs


def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
):
    probs_ratio = torch.exp(policy_log_probs - old_log_probs)
    c = torch.clip(probs_ratio, 1 - cliprange, 1 + cliprange)
    loss = -1 * torch.minimum(advantages * probs_ratio, advantages * c)
    return (loss, {})


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor,
    advantages: torch.Tensor,
    old_log_prob: torch.Tensor,
    cliprange: float,
):
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
    return torch.sum(tensor * mask, dim=dim) / (torch.sum(mask, dim=dim))
