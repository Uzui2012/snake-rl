from json import load
import random
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import sys
from torch.nn.functional import mse_loss
from DQN.replay_memory import replay_memory
from IBP.Memory import LSTMModel
sys.path.append("..")

class Controller_Memory():
    def __init__(self, action_number, frames, context_size, learning_rate, 
                       discount_factor, batch_size, epsilon, save_model,
                       path, epsilon_speed, load_model=False, cuda_flag=True):

        self.cuda_flag = cuda_flag
        self.save_model = save_model
        self.load_model = load_model
        self.epsilon_speed = epsilon_speed
        self.context_size = context_size
        self.controller_network = ControllerModel(context_size=context_size, 
                                       output_size=action_number)
        self.target_network = ControllerModel(context_size=context_size, 
                                              output_size=action_number)

        self.memory_network = LSTMModel(context_size=self.context_size, 
                                cuda_flag=cuda_flag)

        if self.cuda_flag:
            self.controller_network = ControllerModel(context_size=context_size, 
                                           output_size=action_number).cuda()
            self.target_network = ControllerModel(context_size=context_size,
                                            output_size=action_number).cuda()
        else:
            self.controller_network = ControllerModel(context_size=context_size,
                                           output_size=action_number)
            self.target_network = ControllerModel(context_size=context_size,
                                                  output_size=action_number)

        if self.load_model:
            self.controller_network.load_state_dict(torch.load(path, map_location=('cpu')))

        self.controller_optimizer_network = optim.Adam(self.controller_network.parameters(), lr=learning_rate)
        self.memory_optimizer_network = optim.Adam(self.controller_network.parameters(), lr=learning_rate)

        self.discount_factor = discount_factor
        self.target_network.load_state_dict(self.controller_network.state_dict())
        self.frames = frames
        self.previous_action = None
        self.previous_state = 0
        self.previous_context = 0
        self.batch_size = batch_size
        self.memory = replay_memory(10000)
        self.sync_counter = 0
        self.epsilon = epsilon
        self.flag = False
        self.previous_reward = None

    def optimize(self):
        print(len(self.memory.memory))
        if len(self.memory.memory) > self.batch_size + 1:
            print("optimising controller and memory")
            with torch.enable_grad():
                batch = self.memory.sample(self.batch_size)
                if self.cuda_flag:
                    states = torch.empty((self.batch_size, self.frames, 84, 84), requires_grad=True).cuda()
                    contexts = torch.empty((self.batch_size, 100), requires_grad=True).cuda()
                    actions = torch.empty((self.batch_size), requires_grad=True).cuda()
                    rewards = torch.empty((self.batch_size), requires_grad=True).cuda()
                    future_states = torch.empty((self.batch_size, self.frames, 84, 84),requires_grad=True).cuda()
                    terminals = torch.empty((self.batch_size), requires_grad=True).cuda()
                    terminals_reward = torch.empty((self.batch_size), requires_grad=True).cuda()
                                                                                           
                else:
                    states = torch.empty((self.batch_size, self.frames, 84, 84), requires_grad=True)
                    contexts = torch.empty((self.batch_size, 100), requires_grad=True)
                    actions = torch.empty((self.batch_size), requires_grad=True)
                    rewards = torch.empty((self.batch_size), requires_grad=True)
                    future_states = torch.empty((self.batch_size, self.frames, 84, 84),requires_grad=True)
                    terminals = torch.empty((self.batch_size), requires_grad=True)
                    terminals_reward = torch.empty((self.batch_size), requires_grad=True)
                    
                if self.cuda_flag:
                    for i in range(len(batch)):
                        states[i] = batch[i][0]
                        contexts[i] = batch[i][1]
                        actions[i] = batch[i][2] 
                        rewards[i] = batch[i][3] 
                        future_states[i] = batch[i][4] 
                        terminals[i] = batch[i][5] 
                        terminals_reward[i] = batch[i][6] 
                    
                else:
                    with torch.no_grad():
                        for i in range(len(batch)):
                            states[i] = batch[i][0]
                            contexts[i] = batch[i][1]
                            actions[i] = batch[i][2] 
                            rewards[i] = batch[i][3] 
                            future_states[i] = batch[i][4] 
                            terminals[i] = batch[i][5] 
                            terminals_reward[i] = batch[i][6] 
                    
                if self.cuda_flag:
                    future_states = future_states.cuda()
                
                self.controller_network.train()
                self.controller_optimizer_network.zero_grad()
                self.memory_optimizer_network.zero_grad()
                response = self.controller_network(states, contexts, True)
                loss_input = response
                loss_target = loss_input.clone()

                new_values = rewards + torch.mul(self.target_network(future_states).max(dim=1).values[0] * (1 - terminals) + terminals * terminals_reward,
                                                 self.discount_factor)
                
                if self.cuda_flag:
                    new_values = new_values.cuda()
                    idx = torch.cat((torch.arange(self.batch_size).float().cuda(), actions)).cuda()
                else:
                    idx = torch.cat((torch.arange(self.batch_size).float(), actions))
                
                
                idx = idx.reshape(2, self.batch_size).cpu().long().numpy()
                loss_target[idx] = new_values

                loss = mse_loss(input=loss_input, target=loss_target) 
                print(loss)
                loss.backward()
                self.controller_optimizer_network.step()
                self.memory_optimizer_network.step()

    def update_memory(self, reward, terminal, state):
        if self.flag:
            self.memory.append(
                (self.previous_state, self.previous_context, self.previous_action, self.previous_reward, state, terminal,
                 reward))

            self.optimize()

    def make_action(self, state, context, reward, terminal):
        self.controller_network.eval()
        with torch.no_grad():
            state = torch.from_numpy(state.copy()).unsqueeze(0).unsqueeze(0)
            #context is already Tensor
            if self.cuda_flag:
                state = state.float().cuda()
            else:
                state = state.float()

            network_output = self.controller_network(state, context, False)
            values, indices = network_output.max(dim=0)
            randy_random = random.uniform(0, 1)
            if randy_random > self.epsilon:
                
                if self.previous_action != None:
                    forbidden_move = self.forbidden_action()
                    network_output[0][forbidden_move] = -99
                    possible_actions = network_output[0]
                    values, indices = possible_actions.max(dim=0)
                    action = indices.item()
                else:
                    action = indices.item()

            else:
                actions = [0, 1, 2, 3]
                if self.previous_action != None:
                    forbidden_move = self.forbidden_action()
                    actions.remove(forbidden_move)
                action = random.choice(actions)

            #self.update_memory(reward, terminal, state)
            
            self.flag = True
            self.previous_action = action
            self.previous_state = state.clone()
            self.previous_context = context.clone()
            self.previous_reward = reward

            if terminal:
                if reward == -1:
                    self.previous_action = None
                    self.previous_state = None
                    self.previous_context = None
                    self.previous_reward = None
                self.flag = False
            return action

    def forbidden_action(self):
        if self.previous_action == 0:
            forbidden_move = 1
        elif self.previous_action == 1:
            forbidden_move = 0
        elif self.previous_action == 2:
            forbidden_move = 3
        elif self.previous_action == 3:
            forbidden_move = 2
        return forbidden_move

    

class ControllerModel(nn.Module):
    def __init__(self, context_size, state_size=84, hidden_size=512,
                       output_size=1):
        super().__init__()
        # state_size = 
        self.conv1 = nn.Conv2d(1, 32, 8, stride=4) #out=20
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2) #out=9
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1) #out=7
        self.linear1 = nn.Linear(7 * 7 * 64 + context_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, state, context, batch_flag):
        X = F.relu(self.conv1(state))
        X = F.relu(self.conv2(X))
        X = F.relu(self.conv3(X))
        X = X.view(X.size(0), 7 * 7 * 64)
        
        context = context.unsqueeze(0)
        '''print(f"Unsqueezed Context: {context.shape}")
        print(f"Squeezed Context: {context.squeeze().shape}")
        print(f"X Shape: {X.shape}")
        if not batch_flag:
            print(torch.cat((X.squeeze(), context.squeeze()), 0).shape)
            X = F.relu(self.linear1(torch.cat((X.squeeze(), context.squeeze()), 0)))

        else:
            print("AAAAAA")
            print(X.squeeze().shape)
            print(context.squeeze().shape)
            print(torch.cat((X.squeeze(), context.squeeze()), 0).shape)
            X = F.relu(self.linear1(torch.cat((X.squeeze(), context.squeeze()), 0)))'''
        X = F.relu(self.linear1(torch.cat((X.squeeze(), context.squeeze()), 0)))
        X = F.relu(self.linear2(X))
        return X

        
        

