# Deep Q-Learning agent in CARLA

This Python Package can be used to train an agent to drive in [CARLA](https://carla.org/) using [Deep Q-Learning](https://www.tensorflow.org/agents/tutorials/0_intro_rl).


## 📝 Table of Contents

- [Getting Started](#getting_started)
- [Agents](#agents)


## 🏁 Getting Started <a name = "getting_started"></a>

### Install all dependencies:
```
$ pip install -r requirements.txt
```
### Basic usage:
```
$ python -m __main__.py --agent <agent_name> --mode <train/simulate>
```

#### Parameters
  - `--agent`: Specify the agent to be used (Refer to [Agents](#agents)).
  - `--mode`: Train an agent (`train`) or simulate a trained agent (`simulate`).


## Agents <a name = "agents"></a>

### Car-RGB-1 (`--agent car-rgb-1`)

A simple CNN based DQN agent trained on single RGB camera on the front of the car and the steering amount is automatically controlled by the agent.

#### To Train:
```
$ python -m __main__.py --agent car-rgb-1 --mode train
```

#### To Simulate:
```
$ python -m __main__.py --agent car-rgb-1 --mode simulate
```

### Pong-V0 (`--agent car-lidar-1`)
> In-progress
