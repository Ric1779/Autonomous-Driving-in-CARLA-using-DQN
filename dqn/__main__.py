import argparse
from config import agent_map
import sys
    
def run(args: argparse.Namespace) -> None:
    agent = agent_map[args.agent]()
    if args.mode == "train":
        agent.train()
    elif args.mode == "simulate":
        agent.simulate()


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--agent",
        type=str,
        help="The Carla agent.",
        choices=["car-rgb-1"],
        required=True,
    )

    parser.add_argument(
        "--mode",
        type=str,
        help="Train or Simulate (render) the RL agent.",
        choices=["train", "simulate"],
        required=True,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    run(args)
