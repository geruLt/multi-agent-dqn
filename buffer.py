from collections import deque
import random
import numpy as np



class Buffer:
    def __init__(self,n_agents,buffer_size,batch_size):
        self.n_agents = n_agents
        self.batch_size = batch_size
        self.replay_buffers = []
        for agent_idx in range(n_agents):
            self.replay_buffers.append(deque(maxlen=buffer_size))

    def store(self,transition):
        for agent_idx in range(self.n_agents):
            obs = transition[0][agent_idx]
            actions = transition[1][agent_idx]
            rewards = transition[2][agent_idx]
            dones = transition[3][agent_idx]
            new_obs = transition[4][agent_idx]
            agent_transition = (obs, actions, rewards, dones, new_obs)
            self.replay_buffers[agent_idx].append(agent_transition)

    def sample(self):
        samples = []
        for agent_idx in range(self.n_agents): 
            samples.append(random.sample(self.replay_buffers[agent_idx], self.batch_size))
        return samples
