import imp
import sys
from tkinter.tix import IMMEDIATE
from matplotlib import pyplot as plt
from matplotlib.style import context
import numpy as np
from GAN.model import UNet
from GAN.reward_model import reward_model
import util
sys.path.append("..")
from DQN.DQN_agent import DQN_agent
import torch.nn as nn
import torch
import torch.optim as optim
from torch.distributions import Categorical
from IBP.Manager import ManagerModel
from IBP.Controller import ControllerAgent, ControllerModel
from IBP.Memory import LSTMModel
from IBP.Controller_Memory import Controller_Memory
from GAN.model import UNet
import time
from pygame.locals import (
    K_UP,
    K_DOWN,
    K_LEFT,
    K_RIGHT,
    K_ESCAPE,
    KEYDOWN,
    QUIT,
)

# torch.save(model.state_dict(), 
# proj_path + 
# f"new_models\\GAN13_{generator_amplifier}_{discriminator_deamplifier}_new.pt")

class IBP(object):
    def __init__(self, proj_path, environment, cuda_flag=True):
        self.context = torch.zeros(3136) #CONTEXT SEQUENCE IS OF LENGTH 100 
        self.proj_path = proj_path
        self.cuda_flag = cuda_flag
        if cuda_flag:
            self.context= self.context.cuda()
        self.env = environment
        #print(torch.flatten(torch.from_numpy(self.env.get_state())).shape)
        state_size = torch.from_numpy(self.env.get_state())
        state_size = torch.flatten(state_size).shape[0]
        #Fresh Manager
        self.manager = ManagerModel(state_size=state_size, 
                                    context_size=self.context.shape[0],
                                    cuda_flag=cuda_flag)
        #Fresh Controller Agent
        self.controller_memory = Controller_Memory(context_size=self.context.shape[0],
                                          action_number=4,
                                          frames=1, 
                                          learning_rate=0.0001,
                                          discount_factor=0.99, 
                                          batch_size=1,
                                          epsilon=1,
                                          save_model=False,
                                          load_model=False,
                                          path=proj_path+"DQN_trained_model\\1"+
                                          "0x10_model_with_tail.pt",
                                          epsilon_speed=1e-4,
                                          cuda_flag=cuda_flag)
        '''self.controller = ControllerAgent(context_size=self.context.shape[0],
                                          action_number=4,
                                          frames=1, 
                                          learning_rate=0.0001,
                                          discount_factor=0.99, 
                                          batch_size=8,
                                          epsilon=1,
                                          save_model=False,
                                          load_model=False,
                                          path=proj_path+"DQN_trained_model\\1"+
                                          "0x10_model_with_tail.pt",
                                          epsilon_speed=1e-4,
                                          cuda_flag=cuda_flag)'''
        #Fresh Memory
        #self.memory = LSTMModel(context_size=self.context.shape[0], 
         #                       cuda_flag=cuda_flag)
        self.GAN = UNet(5, 1).cuda()
        self.reward_predictor = reward_model(5).cuda()
        gan_path = proj_path
        gan_path += f"IBP_GAN_Folder_2\\IBP_GAN_models\\GAN_Unique_5x5_3_15_num1_epoch24.pt"
        rew_pred_path = proj_path + f"IBP_GAN_Folder_2\\IBP_GAN_Reward_Predictor\\5x5_reward_predictor.pt"
        if not cuda_flag:
            self.GAN.load_state_dict(torch.load(gan_path, map_location=torch.device('cpu')))
            self.GAN.eval() 
            self.reward_predictor.load_state_dict(torch.load(rew_pred_path, map_location=torch.device('cpu')))
            self.reward_predictor.eval()
        else:
            #self.controller_memory = self.controller_memory.cuda()
            self.manager = self.manager.cuda()
            #self.memory = self.memory.cuda()
            self.GAN.load_state_dict(torch.load(gan_path))
            self.GAN.eval() 
            self.reward_predictor.load_state_dict(torch.load(rew_pred_path))
            self.reward_predictor.eval()
        #print(self.GAN)
        # self.reward_predictor = ### Our best rew.prededitor
        # this is likely to just be our controller but excluding everything but 
        # the reward - may not need it?

    def select_action(self, state, context, reward):
        return self.controller_memory.make_action(state, context, reward,
                                True if reward == -1 or reward == 10 else False)

    def plot_results(self, scores):
        plt.figure(figsize=(12,5))
        plt.title("Rewards")
        plt.plot(scores, alpha=0.6, color='red')
        plt.savefig("Snake_Rewards_plot.png")
        plt.close()

    def select_route(self, state, context):
        state = torch.from_numpy(state.copy()).unsqueeze(0).unsqueeze(0)
        #context is already Tensor
        if self.cuda_flag:
            state = state.float().cuda()
        else:
            state = state.float()
        route = self.manager(state, context)
        route = route + 0.00001
        #print(route)
        probabilities = Categorical(route)
        #print(probabilities.probs)
        chosen_route = torch.argmax(route).item()
        tensor_chosen_route = torch.tensor(chosen_route)
        #print(chosen_route)
        if self.cuda_flag:
            tensor_chosen_route = tensor_chosen_route.cuda()
        self.manager.saved_log_probs.append(probabilities.log_prob(tensor_chosen_route))
        return chosen_route

    def get_state_action(self, action, real_state):
        if action == None:
            action = np.zeros(4)
        else:
            action_vec = np.zeros(4)
            action_vec[action] = 1
            action = action_vec
        action = torch.ones_like(torch.from_numpy(real_state)).repeat(4, 1, 1) * torch.from_numpy(action) \
                    .unsqueeze(1) \
                    .unsqueeze(2)
        state_action = torch.cat([torch.from_numpy(real_state).unsqueeze(0), action], dim=0)
        state_action = state_action.unsqueeze(0).float()
        return state_action
        

    def run(self, env, num_eps):
        scores = []
        '''
        NOTING DOWN IBP ALGORITHM FOR EASY REFERENCE:

        funciton a^M (x, x*)
            h <- ()
            n_real <- 0
            n_imagined <- 0

            x_real <- x
            x_imagined <- x

            while n_real < n_max_real_steps
                r <- policy_m (x_real, x*, h_n)

                if r == 0 OR n_imagined < n_max_imagined_steps
                    c <- policy_c (x_real, x*, h_n)
                    x_real <- World(x_real, c)
                    n_real += 1
                    n_imagined = 0
                    x_imagined <- x_real

                else if r == 1 
                    c <- policy_c (x_real, x*, h_n)
                    x_imagined <- I(x_real, x*, h_n)
                    n_imagined += 1

                else if r == 2
                    c <- policy_c (x_imagined, x*, h_n)
                    x_imagined <- I(x_imagined, c)
                    n_imagined += 1

                h <- u(h, c, r, x_real, x_imagined, n_real, n_imagined) 
        '''
        train_manager_flag = True
        manager_optimizer = optim.Adam(self.manager.parameters(), lr=0.01)
        #memory_criterion = nn.MSELoss()
        #memory_optimizer = torch.optim.Adam(self.memory.parameters(), lr=0.001)
        manager_gamma = 0.99
        times = []
        running_manager_loss = []
        running_controller_memory_loss = []
        for ep in range(100):
            
            num_real = 0
            num_imagined = 0
            n_max_imagined_steps = 4
            score = 0
            
            real_reward = env.collision.return_reward(env.height, env.width)
            
            imagined_reward = real_reward
            real_state = env.get_state()
            imagined_state = env.get_state()
            
            start1 = time.time()
            action = self.select_action(state=real_state,
                                            context=self.context, 
                                            reward=real_reward)
            
            flag_first = True
            
            
            while True:
                #self.memory.zero_grad()
                #self.controller.network.zero_grad()
                done_flag = False
                route = self.select_route(real_state, self.context)
                
                #print(f"Chosen Route = {route}")
                route = 0
                if route == 0 or num_imagined > n_max_imagined_steps:
                    action = self.select_action(state=real_state,
                                            context=self.context, 
                                            reward=real_reward)
                    
                    real_state, real_reward, done = env.run_step(action,
                                                            apple_crawl=False)
                    done_flag = done
                    num_real += 1
                    num_imagined = 0
                    imagined_state = real_state
                    imagined_reward = real_reward
                    print(f"Real Action Chosen: {action}, Resultant Reward: {real_reward}")
                    
                elif route == 1:
                    action = self.select_action(state=real_state,
                                            context=self.context, 
                                            reward=real_reward)
                    temp_action = action
                    
                    state_action = self.get_state_action(action, real_state)

                    if self.cuda_flag:
                        state_action = state_action.cuda()
                    #print(state_action.unsqueeze(0).is_cuda())
                    with torch.no_grad():
                        imagined_state = self.GAN(state_action)
                        imagined_reward = self.reward_predictor(imagined_state)
                        imagined_state = imagined_state.cpu().detach().numpy()
                        imagined_state = imagined_state[ 0:, :]
                        
                    num_imagined += 1
                    action = temp_action
                elif route == 2:
                    action = self.select_action(state=imagined_state,
                                            context=self.context, 
                                            reward=imagined_reward)
                    temp_action = action
                    
                    state_action = self.get_state_action(action, real_state)

                    if self.cuda_flag:
                        state_action = state_action.cuda()
                    with torch.no_grad():
                        imagined_state = self.GAN(state_action)
                        imagined_reward = self.reward_predictor(imagined_state)
                        imagined_state = imagined_state.squeeze().squeeze()
                        imagined_state = imagined_state.cpu().detach().numpy()
                        
                    num_imagined += 1
                    action = temp_action

                '''
                    # Push "plan context" through LSTM
                    # LSTM takes:

                    # (After Imagining)
                    # manager output (route) p_j_k
                    # current (real_state) s_j
                    # current (imagined_state, given route - can be s_j again) s_j_pjk
                    # action decided (action) a_j,k
                    # state imagined (next_imagined_state) s_j_k+1
                    # resultant reward (reward) r_j_k
                    # j
                    # k
                    # c_i-1
                    #
                    # OR
                    #
                    # (After Acting)
                    # manager output (route) p_j_k
                    # current (real_state) s_j
                    # current (imagined_state "base", just s_j again) s_j_0
                    # action decided (action) a_j
                    # resultant world state (next_state) s_j+1
                    # resultant reward (reward) r_j
                    # j
                    # k
                    # c_i-1 .
                '''
                #print(num_real + num_imagined)
                self.context = self.controller_memory.memory_network(route=route, 
                                    actual_state=real_state, 
                                    last_imagined_state=imagined_state,
                                    action=action,
                                    new_state=real_state,
                                    reward=real_reward,
                                    j=0,
                                    k=0,
                                    prev_c=self.context)
                self.manager.rewards.append(real_reward)
                

                if real_reward == 10:
                    score += 1            

                
                if done_flag: # or (num_real==20):
                    if train_manager_flag:
                        
                        R = 0
                        manager_loss = []
                        returns = []
                        #print(f"manager rewards:  {self.manager.rewards}")
                        for r in self.manager.rewards[::-1]:
                            R = r + manager_gamma * R
                            returns.insert(0, R)
                        returns = torch.tensor(returns)
                        returns = (returns - returns.mean()) / (returns.std() + ep)
                        #print(f"returns:    {len(returns)}")
                        #print(f"saved log probs:     {len(self.manager.saved_log_probs)}")
                        for log_prob, R in zip(self.manager.saved_log_probs, returns):
                            #print(log_prob)
                            #print(R)
                            manager_loss.append(log_prob * R)
                        
                        manager_optimizer.zero_grad()
                        
                        manager_loss = torch.stack(manager_loss).sum()
                        print(f"manager_loss =   {manager_loss.item()}")
                        running_manager_loss.append(manager_loss.item())
                        print(f"returns:    {returns}")
                        manager_loss.backward(retain_graph=True)
                        manager_optimizer.step()
                        del self.manager.rewards[:]
                        del self.manager.saved_log_probs[:]
                    plt.clf()
                    plt.plot(running_manager_loss)
                    plt.savefig(self.proj_path + f"IBP_results\\manager_loss\\manager_loss")
                    plt.clf()
                    plt.plot(self.controller_memory.running_loss)
                    plt.savefig(self.proj_path + f"IBP_results\\controller_memory_loss\\controller_memory_loss")
                    scores.append(score)
                    end = time.time()
                    times.append(end - start1)
                    print(f"Time taken to take entire episode: {end - start1}")
                    break

        plt.clf()
        plt.plot(running_manager_loss)
        plt.savefig(self.proj_path + f"IBP_results\\manager_loss\\manager_loss")
        plt.clf()
        plt.plot(self.controller_memory.running_loss)
        plt.savefig(self.proj_path + f"IBP_results\\controller_memory_loss\\controller_memory_loss")
        return scores
                    
            

