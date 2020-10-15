import gym
import numpy as np
import os

os.system("")

env = gym.make("Taxi-v3").env

env.reset()
env.render()