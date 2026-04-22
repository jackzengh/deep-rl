import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import gymnasium as gym
from IPython.display import Video

# initialize our policy network
class PolicyNetwork(nn.Module): 

    def __init__(self, input_features=8, output_features=4, hidden_features=128):
        super(PolicyNetwork, self).__init__()

        self.fc1 = nn.Linear(input_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, hidden_features)
        self.fc3 = nn.Linear(hidden_features, output_features)

    def forward(self, x): 

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.self.fc3(x)
        pi = F.softmax(x, dim=-1)

        return pi


# Compute returns for the reward trajectory 
def compute_returns(reward_trajectory, gamma): 

    G = 0
    returns = []

    # of shape (state, next_state, action, reward, done)
    for step in reversed(reward_trajectory):

        G = reward + gamma * G

        # insert at the beginning since we're walking the trajectory backwards 
        returns.insert(0, G)

    # return in a tensor form since we'll use this for the loss
    return torch.tensor(returns, dtype=torch.float32)


# Training - where we run the game, calculate returns, calculate loss - entropy, then step the model
def train(
    env,
    entropy_weight=0.01,
    gamma=0.99,
    learning_rate=0.01
    input_features=8,
    output_features=4,
    hidden_features=128,
    num_episodes=5000,
    ): 
    
    env = gym.make("LunarLander-v3", render_mode="rgb_array")

    # initialize our policy
    policy = PolicyNetwork(input_features, output_features, hidden_features)

    optimizer = optim.Adam(policy.parameters(), lr = learning_rate)

    log = {
        "scores": [],
        "running_avg_scores" = []
    }

    for episode in range(num_episodes): 

        state, _ = env.reset()
        log_probs = []

        done = False
        
        while not done: 

            # lets create our trajectory
            states = []
            next_states = []
            actions = []
            rewards = []

            possible_actions = policy(state) # outputs probability of taking said action, shape of (1,4)
            selected_action = torch.multinomial(possible_actions, 1).item() # returns index, action chosen according to output probability
            log_action_probs = torch.log(possible_actions[0][selected_action]) # find the log probability that we selected the action that we selected

            log_probs.append(log_action_probs)

            next_state, reward, terminal, truncated = env.step(selected_action) # you need to pass in an integer not 
            done = terminal or truncated

            states.append(state)
            next_states.append(next_state)
            actions.append(action)

            if done: 
                returns = compute_rewards(rewards)

                # compute our entropy = -log(probs of each action) * probs of each action
                entropy = torch.sum(torch.log(possible_actions)*possible_actions)

                # compute our loss which is -objective - entropy
                loss = -(torch.sum(log_probs*returns)) - entropy