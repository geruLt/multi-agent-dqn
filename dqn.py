from torch import nn
import torch
import numpy as np
import os
import msgpack
from msgpack_numpy import patch as msgpack_numpy_patch
msgpack_numpy_patch()

class Network(nn.Module):
    def __init__(self, gamma,num_actions,color_channels):
        super().__init__()
        self.gamma = gamma
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        conv_net = self.make_nature_cnn(color_channels)
        self.net = nn.Sequential(conv_net, nn.Linear(512, num_actions))

    def make_nature_cnn(self,n_input_channels,depths=(32,32,64),kernels=(8,4,3),strides=(4,2,1),final_layer=512):
        cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, depths[0], kernel_size=kernels[0],stride=strides[0]),
            nn.ReLU(),
            nn.Conv2d(depths[0], depths[1], kernel_size=kernels[1],stride=strides[1]),
            nn.ReLU(),
            nn.Conv2d(depths[1], depths[2], kernel_size=kernels[2],stride=strides[2]),
            nn.ReLU(),
            nn.Flatten())

        with torch.no_grad():
            n_flatten = cnn(torch.as_tensor(np.empty(shape=(1,84,84))[None]).float()).shape[1]

        out = nn.Sequential(cnn,
            nn.Linear(n_flatten,final_layer,
            nn.ReLU()))

        return out

    def forward(self, x):
        return self.net(x)

    def act(self, obs):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        q_values = self(obs_t.unsqueeze(0))
        max_q_index = torch.argmax(q_values, dim=1)[0]
        action = max_q_index.detach().item()
        return action

    def save(self, save_path):
        params = {k: t.detach().cpu().numpy() for k, t in self.state_dict().items()}
        params_data = msgpack.dumps(params)
        os.makedirs(os.path.dirname(save_path),exist_ok=True)
        with open(save_path,'wb') as f:
            f.write(params_data)

    def load(self,load_path):
        if not os.path.exists(load_path):
            raise FileNotFoundError(load_path)
        with open(load_path, 'rb') as f:
            params_numpy = msgpack.loads(f.read())
        params = {k: torch.as_tensor(v, device=self.device) for k,v in params_numpy.items()}

        self.load_state_dict(params)