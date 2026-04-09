import torch
import torch.optim as optim
import numpy as np
from network import DQNNetwork
import random

class DQNAgent:
    def __init__(self, input_dim, output_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQNNetwork(input_dim, 128, output_dim).to(self.device)
        self.target_net = DQNNetwork(input_dim, 128, output_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.epsilon = 1.0  # Initial exploration
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

    def select_action(self, state, legal_moves):
        if np.random.rand() < self.epsilon:
            return random.choice(legal_moves)  
        state_t = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_t)
            # Masking illegal moves would happen here
            return torch.argmax(q_values).item()

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)