# Actor Critic Architecture 
# NN with two outputs -> policy and value

import torch
import torch.nn as nn
import ptan
import numpy as np 

HID_SIZE = 128


class ModelA2C(nn.Module):

    def __init__(self, obs_size, act_size):
        super(ModelA2C, self).__init__()

        self.base = nn.Sequential(
            nn.Linear(obs_size, HID_SIZE),
            nn.ReLU(),
        )
        self.mu = nn.Sequential(
            nn.Linear(HID_SIZE, act_size),
            nn.Tanh(),
        )
        self.var = nn.Sequential(
            nn.Linear(HID_SIZE, act_size),
            nn.Softplus(),
        )
        self.value = nn.Linear(HID_SIZE, 1)

    def forward(self, x):
        base_out = self.base(x)
        return self.mu(base_out), self.var(base_out), self.value(base_out)



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
