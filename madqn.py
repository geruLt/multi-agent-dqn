import torch
from torch import nn
import numpy as np

from dqn import Network


class maDQN:
    def __init__(self,n_agents,num_actions,color_channels,learning_rate,gamma,env_name):
        self.agents = []
        self.target_agents = []
        self.optimizers = []
        self.n_agents = n_agents
        self.gamma = gamma
        self.env_name = env_name
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        for agent_idx in range(self.n_agents):
            self.agents.append(Network(gamma,num_actions,color_channels))
            self.agents[agent_idx].to(self.device)
            self.target_agents.append(Network(gamma,num_actions,color_channels))
            self.target_agents[agent_idx].to(self.device)
            self.target_agents[agent_idx].load_state_dict(self.agents[agent_idx].state_dict())
            self.optimizers.append(torch.optim.Adam(self.agents[agent_idx].parameters(), lr=learning_rate))

    def target_update(self):
        for agent_idx in range(self.n_agents):
            self.target_agents[agent_idx].load_state_dict(self.agents[agent_idx].state_dict())


    def save_checkpoint(self, score):
        print('... saving checkpoint ...')
        agent_save_paths = []
        target_agent_save_paths = []

        for i in range(self.n_agents):
            agent_save_paths.append(self.env_name + '/agent'+str(i)+'-score-{:.2f}.pack'.format(score)) 
            target_agent_save_paths.append(self.env_name + '/target-agent'+str(i)+'-score-{:.2f}.pack'.format(score)) 

        for agent_idx in range(self.n_agents):
            self.target_agents[agent_idx].load_state_dict(self.agents[agent_idx].state_dict())
            self.agents[agent_idx].save(agent_save_paths[agent_idx])
            self.target_agents[agent_idx].save(target_agent_save_paths[agent_idx])

    def load_checkpoint(self,load_score):
        print('... loading checkpoint ...')
        for agent_idx in range(self.n_agents):
            path_a = self.env_name + '/agent' + str(agent_idx) + '-score-' + load_score + '.pack'
            path_ta = self.env_name + '/agent' + str(agent_idx) + '-score-' + load_score + '.pack'
            self.agents[agent_idx].load(path_a)
            self.target_agents[agent_idx].load(path_ta)

    def choose_actions(self, raw_obs):
        actions = []
        for agent_idx, agent in enumerate(self.agents):
            action = agent.act(raw_obs[agent_idx])
            actions.append(action)
        return actions

    def learn(self, batch):
        for agent_idx in range(self.n_agents):
            agent_batch = batch[agent_idx]
            loss = self.compute_loss(agent_batch,self.agents[agent_idx],self.target_agents[agent_idx])
            # Gradient Descent
            self.optimizers[agent_idx].zero_grad()
            loss.backward()
            self.optimizers[agent_idx].step()

    def compute_loss(self, transitions, online_net, target_net):
        obses = np.asarray([t[0] for t in transitions])
        actions = np.asarray([t[1] for t in transitions])
        rews = np.asarray([t[2] for t in transitions])
        dones = np.asarray([t[3] for t in transitions])
        new_obses = np.asarray([t[4] for t in transitions])

        obses_t = torch.as_tensor(obses, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(-1)
        rews_t = torch.as_tensor(rews, dtype=torch.float32, device=self.device).unsqueeze(-1)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(-1)
        new_obses_t = torch.as_tensor(new_obses, dtype=torch.float32, device=self.device)

        # Compute Targets
        target_q_values = target_net(new_obses_t)
        max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]
        targets = rews_t + self.gamma * (1 - dones_t) * max_target_q_values

        # Compute Loss
        q_values = online_net(obses_t)
        action_q_values = torch.gather(input=q_values, dim=1, index=actions_t)

        loss = nn.functional.smooth_l1_loss(action_q_values, targets)
        return loss