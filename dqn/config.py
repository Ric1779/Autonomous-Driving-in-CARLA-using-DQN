from typing import Dict
from agents.base_agent import BaseAgent
from agents.car_rgb_1.agent import CarAgent

agent_map: Dict[str, BaseAgent] = {"car-rgb-1": CarAgent}