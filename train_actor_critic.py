import argparse
import gym

from agents.a2c_continuous import ModelA2C
from dj_env.dj_env import DJEnv

import numpy as np
import torch


ENV_ID = "DJEnv"
# ENV_NAME = 'HalfCheetahBulletEnv-v0'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=False, help="Model file to load")
    parser.add_argument("-e", "--env", default=ENV_ID, help="Environment name to use, default=" + ENV_ID)
    parser.add_argument("-r", "--record", help="If specified, sets the recording dir, default=Disabled")
    args = parser.parse_args()

    env = DJEnv()
    if args.record:
        env = gym.wrappers.Monitor(env, args.record)

    net = ModelA2C(
            env.observation_space.shape[0],
            env.action_space.shape[0])
    if args.model:
        net.load_state_dict(torch.load(args.model))
    else:
        #TODO URGENT INIT RANDOM PROBABLY
        pass

    obs = env.reset()
    total_reward = 0.0
    total_steps = 0
    while True:
        obs_v = torch.FloatTensor([obs])
        mu_v, var_v, val_v = net(obs_v)
        action = mu_v.squeeze(dim=0).data.numpy()
        action = np.clip(action, -1, 1)
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        total_steps += 1
        if done:
            break
    print("In %d steps we got %.3f reward" % (total_steps, total_reward))
