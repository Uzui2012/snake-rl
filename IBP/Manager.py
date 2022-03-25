import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import util

class ManagerModel(nn.Module):
    def __init__(self, state_size, context_size, hidden_size=(100,100), output_size=3, 
                       cuda_flag=True):
        super().__init__()
        self.cuda_flag = cuda_flag
        self.linear1 = nn.Linear(context_size + state_size, hidden_size[0])
        self.linear2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.linear3 = nn.Linear(hidden_size[1], output_size)
        self.activation = nn.ReLU()

        self.saved_log_probs = []
        self.rewards = []
        
    def forward(self, state, context):
        seq = []
        seq.append(context)
        seq.append(torch.flatten(state))

        if self.cuda_flag:
            seq[1] = seq[1].cuda()

        seq_tensor = util.tensor_from(seq).float()

        if self.cuda_flag:
            seq_tensor = seq_tensor.cuda()

        X = F.relu(self.linear1(seq_tensor))
        X = F.relu(self.linear2(X))
        X = F.relu(self.linear3(X))
        return X

    def discount_rewards(self, rewards_in, gamma=0.99):
        rewards = np.array([gamma**i * rewards_in[i] 
                    for i in range(len(rewards_in))])
        # Reverse the array direction for cumsum and then
        # revert back to the original order
        rewards = rewards[::-1].cumsum()[::-1]
        return rewards - rewards.mean()

    
        
    

