import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import matplotlib
matplotlib.use('Agg')  # Headless backend
import matplotlib.pyplot as plt
from tetris_game import TetrisGame
import os
import shutil

class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU()
        )
        conv_out = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out, 512), nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        x = self.conv(x).view(x.size(0), -1)
        return self.fc(x)

class DQNAgent:
    def __init__(self, state_shape, n_actions, device):
        self.device = device
        self.n_actions = n_actions
        self.policy_net = DQN(state_shape, n_actions).to(device)
        self.target_net = DQN(state_shape, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.0005)
        self.memory = deque(maxlen=50000)
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.target_update = 10
        self.criterion = nn.SmoothL1Loss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.policy_net(state).argmax().item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        next_q = self.target_net(next_states).max(1)[0]
        target_q = rewards + self.gamma * next_q * (1 - dones)

        loss = self.criterion(current_q, target_q.detach())
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

def plot_progress(episode_numbers, episode_scores, episode_lines, episode):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(episode_numbers, episode_scores)
    plt.title('Episode Scores')
    plt.xlabel('Episode')
    plt.ylabel('Score')

    plt.subplot(1, 2, 2)
    plt.plot(episode_numbers, episode_lines)
    plt.title('Lines Cleared')
    plt.xlabel('Episode')
    plt.ylabel('Lines')

    plt.tight_layout()
    plt.savefig(f'progress_plots/progress_episode_{episode}.png')
    plt.close()

def train(episodes=500):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = TetrisGame(render=True)
    state_shape = (2, env.GRID_HEIGHT, env.GRID_WIDTH)
    agent = DQNAgent(state_shape, 3, device)

    # Clean up or create plot folder
    plot_dir = 'progress_plots'
    if os.path.exists(plot_dir):
        shutil.rmtree(plot_dir)
    os.makedirs(plot_dir, exist_ok=True)

    episode_scores = []
    episode_lines = []
    episode_numbers = []

    for ep in range(episodes):
        state, done, score = env.reset_game(), False, 0
        step_count = 0
        while not done and step_count < 5000:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            state, score = next_state, score + reward
            step_count += 1
            env.render_game()

        episode_scores.append(score)
        episode_lines.append(env.lines_cleared)
        episode_numbers.append(ep + 1)

        if ep % agent.target_update == 0:
            agent.update_target_network()
            plot_progress(episode_numbers, episode_scores, episode_lines, ep + 1)

        print(f"Episode {ep+1}, Score: {score:.2f}, Epsilon: {agent.epsilon:.3f}, Lines Cleared: {env.lines_cleared}")

    torch.save(agent.policy_net.state_dict(), 'tetris_model.pth')

if __name__ == "__main__":
    train(50000)