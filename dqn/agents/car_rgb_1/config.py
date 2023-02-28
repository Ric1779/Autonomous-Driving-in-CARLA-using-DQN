from dataclasses import dataclass
import os
from pathlib import Path

@dataclass
class TrainParams:
    """Parameters used for training the agent.
    """

    memory_size: int = 50000
    episodes: int = 1000
    batch_size: int = 1
    target_update_frequency: int = 100
    frequency: int = 1
    gamma: float = 0.95
    lr: float = 1e-4
    epsilon: float = 0.1
    epsilon_end: float = 0.05
    anneal_length: int = 10 ** 4
    num_actions: int = 3
    obs_stack_size = 4


@dataclass
class EvaluateParams:
    """Parameters used for evaluating the agent.
    """

    frequency: int = 25
    episodes: int = 5


@dataclass
class CarConfig:
    """Configuration for the CartPole agent.
    """

    train: TrainParams = TrainParams()
    evaluate: EvaluateParams = EvaluateParams()
    env: str = "car_env"
    model_path: str = os.path.join(
        Path.cwd(), "models/car_best.pt"
    )

@dataclass
class CarEnvConfig:
    """
    Configuration for the CARLA env
    """
    SHOW_PREVIEW = False
    IM_WIDTH = 640
    IM_HEIGHT = 480
    SECONDS_PER_EPISODE = 10
    STEER_AMT = 1.0
    SECONDS_PER_EPISODE = 10
