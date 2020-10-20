import gym
import random
import numpy as np
import pandas as pd

# from IPython.display import clear_output

env = gym.make("Taxi-v3").env

# Initialize q-table
q_table = np.zeros([env.observation_space.n, env.action_space.n])

episode = 1000000

# Hyperparameters
alpha = 0.1  # Learning rate - the extent to which our Q-values are being updated in every iteration.
gamma = 0.6  # Discount factor - determines how much importance we want to give to future rewards
epsilon = 0.1  # Probability of exploration

# For plotting metrics
all_epochs = []
all_penalties = []

for i in range(1, episode):
    # Initialize environment
    state = env.reset()
    epochs, penalties, reward, = 0, 0, 0
    done = False

    while not done:
        # Exploit or explore
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        next_state, reward, done, info = env.step(action)

        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])

        # Q-learning
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        # Q-table update
        q_table[state, action] = new_value

        # Detect wrong dropoffs
        if reward == -10:
            penalties += 1

        # State update
        state = next_state
        epochs += 1

print("Training finished.\n")

# Visualize result to see if it looks reasonable
state = env.reset()
for i in range(15):
    env.render()
    action = np.argmax(q_table[state])
    next_state, reward, done, info = env.step(action)
    state = next_state

# Save results in csv-file
index = np.arange(1, 3001)
value = q_table.ravel()
df = pd.DataFrame({"Id": index, "Value": value})
df.to_csv("submission.csv", index=False)

