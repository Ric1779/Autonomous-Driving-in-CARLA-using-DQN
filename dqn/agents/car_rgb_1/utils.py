
from typing import Tuple
import torch
from torch import Tensor
from replay_memory import Batch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device in utils:{}'.format(device))

def preprocess_observation(obs):
    """Performs necessary observation preprocessing."""
    tensor_obs = torch.tensor(obs, device=device).float()
    tensor_obs = tensor_obs.permute(2,0,1)
    return tensor_obs


def preprocess_sampled_batch(
    batch: Batch
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Pre-processes a batch of samples from the memory.

    Args:
        batch (Batch): A batch of raw samples.

    Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]: A batch of pre-processed samples;
    """
    obs = torch.stack(batch.obs).to(device)
    next_obs = torch.stack(batch.next_obs).to(device)
    actions = torch.Tensor(batch.actions).long().unsqueeze(1).to(device)
    rewards = torch.Tensor(batch.rewards).long().unsqueeze(1).to(device)
    dones = torch.Tensor(batch.dones).long().unsqueeze(1).to(device)
    return obs, next_obs, actions, rewards, dones