"""
Replay buffer for storing and sampling experiences.
"""
from typing import Tuple

import torch
import numpy as np


class ReplayBuffer:
    """A simple replay buffer."""

    def __init__(
        self,
        buffer_size: int,
        state_dim: int,
        action_dim: int,
        device: torch.device | None,
    ):
        self.device = device

        self.state = torch.zeros((buffer_size, state_dim)).to(self.device)
        self.action = torch.zeros((buffer_size, action_dim)).to(self.device)
        self.reward = torch.zeros(buffer_size).to(self.device)
        self.next_state = torch.zeros((buffer_size, state_dim)).to(self.device)
        self.done = torch.zeros(buffer_size, dtype=torch.int64).to(self.device)

        self.ptr = 0
        self.size = 0
        self.buffer_size = buffer_size

    def push(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor,
    ):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.done[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self, batch_size: int) -> Tuple:
        batch_idx = torch.randint(0, self.size, size=batch_size)

        state = self.state[batch_idx]
        action = self.action[batch_idx]
        reward = self.reward[batch_idx]
        next_state = self.next_state[batch_idx]
        done = self.done[batch_idx]

        return (state, action, reward, next_state, done)
