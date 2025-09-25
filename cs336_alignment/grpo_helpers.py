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
