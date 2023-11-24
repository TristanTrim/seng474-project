# Actor Critic Architecture 
# NN with two outputs -> policy and value

import torch
import torch.nn as nn
import ptan
import numpy as np 

FIRST_HID_SIZE = 32
SECOND_HID_SIZE = 64
THIRD_HID_SIZE = 64
GAMMA = 0.9 # Gamma in general should be between 0.9 and 0.99 (lower value encourages short term thinking)


class ModelA2C(nn.Module):
    def __init__(self, input_shape, act_size):
        super(ModelA2C, self).__init__()

        # Inital Evaluation Head - will later be split into policy and value heads
        self.base = nn.Sequential(
            nn.Conv2d(input_shape[0], FIRST_HID_SIZE, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(FIRST_HID_SIZE, SECOND_HID_SIZE, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(SECOND_HID_SIZE, THIRD_HID_SIZE, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # Policy Head (Actor) - outputs probabilities for taking each action (beacuse continuous action space output mean and variance)
        self.policy_mean = nn.Sequential(
            nn.Linear(THIRD_HID_SIZE, act_size),
            nn.tanh() # allows for final values between -1 and 1 
        )
        self.policy_var = nn.Sequential(
            nn.Linear(THIRD_HID_SIZE, act_size),
            nn.Softplus() # allows for only final values above 0
        )

        # Value Head (Critic) - outputs single number that is the value of current state
        self.value = nn.Sequential(
            nn.Linear(THIRD_HID_SIZE, 1)
        )

        # forward function to provide all outputs of the network
        def forward(self, x):
            base_out = self.base(x)
            return self.policy_mean(base_out), self.policy_var(base_out), self.value(base_out)


# Agent function based on the following: https://github.com/colinskow/move37/blob/master/actor_critic/a2c_continuous.py#L6
class AgentA2C(ptan.agent.BaseAgent):
    def __init__(self, net, device="cpu"):
        self.net = net
        self.device = device

    def __call__(self, states, agent_states):
        states_v = ptan.agent.float32_preprocessor(states).to(self.device)

        mu_v, var_v, _ = self.net(states_v) # get actions by sampling the normal distribution
        mu = mu_v.data.cpu().numpy()
        sigma = torch.sqrt(var_v).data.cpu().numpy()
        actions = np.random.normal(mu, sigma)
        actions = np.clip(actions, -1, 1)
        return actions, agent_states

# log probability from a normal distribution (used for policy loss)
def calc_logprob(mu_v, var_v, actions_v):
    distribution = torch.distributions.Normal(mu_v, torch.sqrt(var_v))
    return distribution.log_prob(actions_v)


# TO DOs

# training loop & testing function & visualizations 