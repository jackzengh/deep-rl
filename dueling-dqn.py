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
import ffmpeg

### DOUBLE DQN ### 
# How double DQN differs is that we take the max_action prescribed by the newly trained model DQN (that changes on every step in an episode)
# But 

### Make the Game ###
env = gym.make("LunarLander-v3", render_mode="rgb_array")

# Create our DQN nn.Module (3x linear layer with RELU)
class DuelingDQN(nn.Module): 

    def __init__(self, input_state_features=8, hidden_features=64, num_actions=4):
        
        super(DQN, self).__init__()

        self.fc1 = nn.Linear(input_state_features, hidden_features) # transform from 8 to hidden_features
        self.fc2 = nn.Linear(hidden_features, hidden_features)

        self.value = nn.Linear(hidden_features, 1)
        self.advantage = nn.Linear(hidden_features, num_actions) 

    def forward(self, x): # where x is the input, shaped as (B, input_state_features)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        value = self.value(x)
        advantage = self.advantage(x) # output tensor (B,4)

        advantage_mean = torch.mean(advantage, dim=-1, keepdim=True) # need to unsqueeze
        advantage = advantage - advantage_mean

        x = advantage + value # (B,4) + (B,1) which is broadcast across all 

        return x # tensor of actions for each batch and the associated values


# Create timecapsule replay buffer (storing memory of our steps and associated rewards), access_memory, insert_memory
class TimeCapsule:

    def __init__(self, max_memory=100_000, num_state_features=8):

        self.max_memory = max_memory
        self.num_state_features = num_state_features
        # we want a replay buffer that stores: state, action, reward, done
        self.state_memory = torch.zeros((self.max_memory, self.num_state_features), dtype=torch.float32)
        self.next_state_memory = torch.zeros((self.max_memory, self.num_state_features), dtype=torch.float32)
        
        # we only want to store the action we took at that state
        self.action_memory = torch.zeros((self.max_memory, ), dtype=torch.long)

        self.reward_memory = torch.zeros((self.max_memory, ), dtype=torch.float32)
        self.terminal_memory = torch.zeros((self.max_memory,), dtype=torch.bool)

        self.current_memories_counter = 0

    def add_memory(self, state, next_state, action, reward, terminal): 

        # we want to add to each from the start of their respective arrays, [1,2,3,4,5], [6,2,3,4,5]

        # in the beginning we just append, then once we get to max_memory, we must start popping from the front and adding there
        
        # we can use a trick where if we take the modulus of how many we've added into this array, by the max_memory, we'll get the index we should index into 
        # take for example we have 5 slots, current memories counter = 8, that means we should insert at the 9th index (i.e. memory[8])
        idx = self.current_memories_counter % self.max_memory

        self.state_memory[idx] = torch.tensor(state, dtype=self.state_memory.dtype)
        self.next_state_memory[idx] = torch.tensor(next_state, dtype=self.next_state_memory.dtype)
        self.action_memory[idx] = torch.tensor(action, dtype=self.action_memory.dtype)
        self.reward_memory[idx] = torch.tensor(reward, dtype=self.reward_memory.dtype)
        self.terminal_memory[idx] = torch.tensor(terminal, dtype=self.terminal_memory.dtype)

        # add to current_memories_counter
        self.current_memories_counter += 1


    def access_memory(self, batch_size, device="mps"):

        # now lets retrieve batch_size memories from the replay buffer, we want to return them in a tuple form

        # first check that we have enough memories to retrieve
        total_memories = min(self.current_memories_counter, self.max_memory) # since len(state_memory) is always max_memory given we initialized with zeros, we want to know how many we actually have
        if total_memories < batch_size: 
            return None # back out, we don't have enough memories to retrieve yet

        # sample memories at random, we want to first get an array of random array indexes from 0 -> total_memories, then we index into the arrays with those indexes
        batch_indices = np.random.choice(total_memories, batch_size, replace=False)
        batch_indices = torch.tensor(batch_indices, dtype=torch.long)

        # whenever you access data you must move it to your device to then perform training on 
        batch = {"states": self.state_memory[batch_indices].to(device),
                 "next_states": self.next_state_memory[batch_indices].to(device),
                 "actions": self.action_memory[batch_indices].to(device),
                 "rewards": self.reward_memory[batch_indices].to(device),
                 "terminal": self.terminal_memory[batch_indices].to(device)}

        return batch

        # to make these ready for ingest in an LLM we have to unsqueeze and turn them into (1, X) tensors

# Create our agent that walks the environment
class Agent:
    def __init__(
        self,
        max_memories=100_000,
        epsilon=1.0, # in the beginning it should take lots of exploratory steps
        epsilon_decay=0.9995, # decay as the policy matures
        min_epsilon=0.05,
        learning_rate=5e-4,
        time_decay = 0.9, # AKA gamma
        input_state_features=8,
        num_actions=4,
        hidden_features=128,
        device = "mps",
    ):
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.learning_rate = learning_rate
        self.max_memories = max_memories
        self.hidden_features = hidden_features
        self.input_state_features = input_state_features
        self.num_actions = num_actions
        self.time_decay = time_decay
        # Save the device so every method (select_action, train_step) can move tensors onto the same
        # device as the model. Mixing CPU and MPS/CUDA tensors in one op raises a RuntimeError.
        self.device = device

        self.loss_fn = nn.MSELoss()

        self.time_capsule = TimeCapsule(max_memory=max_memories, num_state_features=input_state_features)


        self.DQN = DuelingDQN(input_state_features, hidden_features, num_actions).to(device)
        
        # Create a new model so that we don't have crazy moving targets, that just copies the current weights from DQN
        self.DQN_NEXT = DuelingDQN(input_state_features, hidden_features, num_actions).to(device)
        self.DQN_NEXT.load_state_dict(self.DQN.state_dict())
        
        # we never update this model
        self.DQN_NEXT.eval()

        # optimizer needs the learning rate and parameters 
        self.optimizer = optim.Adam(self.DQN.parameters(), lr = self.learning_rate)

    def update_target(self): 
        
        # re-initialize DQN_NEXT with the new weights of DQN
        self.DQN_NEXT.load_state_dict(self.DQN.state_dict())

    
    # traverse the environment and return an action
    def select_action(self, state): 
        
        # check that state is a torch tensor, with dtype: torch.float32
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32) # save state as tensor of floats

        if state.dim() == 1:
            state = state.unsqueeze(dim=0) # unsqueeze doesn't change in place, so you have to copy

        # Make sure the input is on the same device as the model. gym gives us CPU numpy arrays,
        # and torch.tensor(...) defaults to CPU too — but self.DQN lives on self.device (e.g. "mps"),
        # so calling self.DQN(state) while state is on CPU raises a device-mismatch RuntimeError.
        state = state.to(self.device)

        if np.random.rand() < self.epsilon:
            action = env.action_space.sample()
        else:
            # self.DQN(state) returns a (1, num_actions) tensor of Q-values. argmax picks the best action's index.
            # .item() turns a 1-element tensor into a plain Python int, which is what env.step(...) expects.
            with torch.no_grad():
                action = self.DQN(state).argmax(dim=-1).item()

        return action

    # Create our training algorithm, lr_scheduler
    def update_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)

    def inference(self, state, device="cpu"):

        ### Quick Inference Script to Get Action from State ###
        self.DQN = self.DQN.to(device)
        self.DQN.eval()
        
        with torch.no_grad():
            Q_s_a = self.DQN(state.to(device))
            
        return torch.argmax(Q_s_a).item()
    
    # Create train_model
    def train_step(self, batch_size): 

        # we sample from the batch and get training data, then we use that training data to update our model.
        # Pass self.device so the batch tensors end up on the same device as the model.
        batch = self.time_capsule.access_memory(batch_size, device=self.device)
        
        if batch is None:
            return None
        
        # Q values for the 4 possible actions we might take at this state across every batch (B, 4)
        q_estimate = self.DQN(batch["states"])

        # now we want to update based on the actions we actually took 
        max_q_estimate = torch.gather(q_estimate, index=batch["actions"].unsqueeze(dim=1), dim=1).squeeze(1)

        with torch.no_grad(): 
            # self.DQN.eval() # not needed because DQN_NEXT is already a frozen model
            q_next_action_estimate = self.DQN(batch["next_states"]) # input a (B,8) tensor, output a (B,4) tensor
            action_index = q_next_action_estimate.argmax(axis=-1, keepdim=True) # outputs a (B,1) tensor with the index of the action
            qnext_next_action_estimate = self.DQN_NEXT(batch["next_states"]) # output type is (B,4) tensor
            max_qnext_estimate = torch.gather(qnext_next_action_estimate, index=action_index, dim=1) # index into each batches action array, pluck out the DQN-selected action index
            # self.DQN.train()

        # compare the Q value of the TD error with the current Q value for that state action pair
        # max_q_next_estimate = q_next_estimate.max(axis=-1).values # returning (B, 1) which is the value of the action you take in the next state for every batch


        # compute the TD target (which is the TD error, but you don't subtract what the current Q_sa value is)
        td_target = batch["rewards"] + self.time_decay * max_q_next_estimate * (~batch["terminal"])

        # compute loss
        loss = self.loss_fn(td_target, max_q_estimate)

        # ensure there are no old gradients in the optimizer
        self.optimizer.zero_grad()
        
        # do loss.backward() 
        loss.backward()
        
        self.optimizer.step()

# Create our trainer which allows the agent we built to interact with the environment through a number of episodes, with sequential updates
def trainer(
    env, 
    num_episodes = 3000,
    min_reward=200,
    max_memories=100_000,
    epsilon=1.0, 
    epsilon_decay=0.999, 
    min_epsilon=0.05,
    learning_rate=0.01,
    input_state_features=8,
    num_actions=4,
    hidden_features=128,
    device = "mps", 
    game_tolerance=10,
    batch_size=64,
    log_freq=3,
    update_target_freq=3,
    running_avg_steps=25,
): 

    # initialize the agent
    agent = Agent(
        max_memories=max_memories,
        epsilon=epsilon, 
        epsilon_decay=epsilon_decay, 
        min_epsilon=min_epsilon,
        learning_rate=learning_rate,
        input_state_features=input_state_features,
        num_actions=num_actions,
        hidden_features=hidden_features,
        device=device
    )

    log = {
        "scores" : [],
        "running_avg_scores": []
    }

    # stop training after 10 consecutive wins, record how many wins we get 
    ending_tol = 0

    for i in range(num_episodes): 

        state, _ = env.reset()
        done = False

        score = 0

        step = 0
        
        while not done: 

            action = agent.select_action(state)

            next_state, reward, terminal, truncated, _ = env.step(action)
            done = terminal or truncated

            # add to our replay buffer for training later
            agent.time_capsule.add_memory(state, next_state, action, reward, terminal)

            state = next_state


            # train our agent at every step
            agent.train_step(batch_size=batch_size)

            score += reward
            step += 1

        if step % update_target_freq == 0: 
            agent.update_target()

        agent.update_epsilon()

        if i % 1000 == 0:
            visualize_agent(agent, save_path=f"episode_{i}.mp4")
            
        log["scores"].append(score)
        running_avg_scores = np.mean(log["scores"][-running_avg_steps:])
        log["running_avg_scores"].append(running_avg_scores)

        # how frequently to log (every 3 episodes)
        if i % log_freq == 0:
            print(f"Game #: {i} | Score: {score} | Moving Avg Scores: {running_avg_scores} | Epsilon: {agent.epsilon}")
        
        # stop if we get game_tolerance wins in a row
        if score > min_reward:
            ending_tol += 1
        
            if ending_tol >= game_tolerance: 
                break
        
        else: 
            ending_tol = 0 # since we want to see consecutive wins, if we lose even one game, reset the win tolerance

    print("Completed Training")
    return agent, log

# Visualize a trained agent playing one episode of LunarLander.
#
# How it works: we create a *fresh* env with render_mode="rgb_array" so every env.render() call
# gives us back a numpy array of pixels (the current game frame, shape roughly (400, 600, 3)).
# We run the agent for one episode with epsilon=0 (pure greedy — we want to watch what it has
# actually learned, not random noise), collecting a frame each step into a list. Then we stitch
# those frames into an MP4 using matplotlib's animation module.
def visualize_agent(agent, save_path="lunarlander.mp4", max_steps=1000, fps=30):

    # Build a dedicated rendering env. We don't reuse the training env because we want a clean
    # episode and the same env object may be mid-episode elsewhere.
    render_env = gym.make("LunarLander-v3", render_mode="rgb_array")
    observation, _ = render_env.reset()

    # Temporarily force greedy behavior (no exploration). Save old epsilon so we can restore it
    # afterwards — otherwise calling visualize_agent() during training would permanently wipe
    # exploration.
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0

    frames = []
    total_reward = 0.0
    done = False
    step = 0

    # Put the model in eval mode. For this simple MLP it doesn't matter (there's no Dropout or
    # BatchNorm), but it's a good habit — it tells any layer that behaves differently during
    # training vs inference to use its inference behavior.
    agent.DQN.eval()

    while not done and step < max_steps:
        # Capture the current frame *before* we act, so the first frame is the initial state.
        frames.append(render_env.render())

        action = agent.select_action(observation)
        observation, reward, terminal, truncated, _ = render_env.step(action)
        total_reward += reward
        done = terminal or truncated
        step += 1

    # Capture the final frame after the episode ends so the video shows the landing/crash.
    frames.append(render_env.render())
    render_env.close()

    # Put the model back into training mode and restore epsilon so training can continue cleanly.
    agent.DQN.train()
    agent.epsilon = old_epsilon

    print(f"Episode finished in {step} steps with total reward {total_reward:.1f}")

    # --- Turn the list of frames into an MP4 ---
    # We build a matplotlib figure whose only content is an image. Each animation frame, we
    # swap the image data for the next captured frame. FuncAnimation plays them in sequence.
    height, width, _ = frames[0].shape
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    ax.axis("off")  # hide axis ticks/labels — we want a clean video
    img = ax.imshow(frames[0])

    def update(frame_idx):
        img.set_data(frames[frame_idx])
        return [img]

    # interval is milliseconds between frames; 1000/fps gives us the requested frame rate.
    anim = animation.FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps, blit=True)

    # Try saving as MP4 (needs ffmpeg installed). Fall back to GIF if ffmpeg isn't available.
    try:
        anim.save(save_path, writer=animation.FFMpegWriter(fps=fps))
        print(f"Saved video to {save_path}")
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        # ffmpeg not found → save as GIF instead. Swap the .mp4 extension for .gif.
        gif_path = save_path.rsplit(".", 1)[0] + ".gif"
        anim.save(gif_path, writer="pillow", fps=fps)
        print(f"ffmpeg not available ({e}). Saved GIF to {gif_path} instead.")

    plt.close(fig)
    return total_reward


device = "cuda" if torch.cuda.is_available() else "mps"
agent, log = trainer(env, device=device)

# After training, watch the trained agent play one episode.
visualize_agent(agent)
