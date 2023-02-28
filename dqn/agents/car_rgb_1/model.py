import torch
from torch import Tensor
import numpy as np
from agents.car_rgb_1.config import CarConfig

class DQN(torch.nn.Module):
    def __init__(self, cfg: CarConfig):
        super(DQN,self).__init__()

        self.batch_size = cfg.train.batch_size
        self.gamma = cfg.train.gamma
        self.epsilon = cfg.train.epsilon
        self.epsilon_end = cfg.train.epsilon_end
        self.anneal_length = cfg.train.anneal_length
        self.num_actions = cfg.train.num_actions

        self.conv1 = torch.nn.Conv2d(3,64,3,1,1)
        self.conv2 = torch.nn.Conv2d(64,128,3,1,1)
        self.conv3 = torch.nn.Conv2d(128,256,3,1,1)

        self.fc1 = torch.nn.Linear(256 * 60 * 80, 120)  
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 3)

        self.relu = torch.nn.ReLU()
        self.avg_pool2d = torch.nn.AvgPool2d(2,2)

    def forward(self,x):
        x = self.avg_pool2d(self.relu(self.conv1(x)))
        x = self.avg_pool2d(self.relu(self.conv2(x)))
        x = self.avg_pool2d(self.relu(self.conv3(x)))
        
        x = x.view(-1, self.num_flat_features(x))
        
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))

        return x

    def num_flat_features(self,x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def act(self, observation: Tensor) -> int:
        """ Selects and action using epsilon-greedy exploration strategy.
            0: Steer left
            1: Move straight
            2: Steer right

            Args: observation (Tensor): The current observation
            Returns (int): The action taken by the DQN based on the observation. 
        
        """

        if np.random.uniform(low=0.0, high=1.0) <= self.epsilon:
            action = np.random.randint(low=0, high=self.num_actions)
        else:
            prediction = self(observation)
            action = int(torch.argmax(prediction,1).item())
        
        return action