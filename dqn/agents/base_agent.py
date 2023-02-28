from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """A DQN agent that can learn to play various Atari games (Pong, CartPole etc.).
    """

    @abstractmethod
    def train(self) -> None:
        """Trains the specified agent using a DQN.
        """
        pass

    @abstractmethod
    def optimize(self) -> float:
        """Samples a batch from the replay buffer and optimizes the Q-network.

        Returns:
            float: The loss of the DQN.
        """
        pass

    @abstractmethod
    def evaluate(self, render: bool = False) -> float:
        """Runs self.cfg.evaluate.episodes episodes to evaluate the current policy.

        Args:
            render (bool, optional): A flag to determine if the environment should be rendered. Defaults to False.

        Returns:
            float: The mean return after running the current policy for self.cfg.evaluate.episodes episodes.
        """
        pass

    @abstractmethod
    def simulate(self) -> None:
        """Runs a simulation of the agent playing the game.
        """
        pass
