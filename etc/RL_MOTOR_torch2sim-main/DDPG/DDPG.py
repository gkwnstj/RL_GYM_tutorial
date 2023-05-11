import torch
from torch import nn, cat
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from replaybuffer import ReplayBuffer
import numpy as np


class Actor(nn.Module):

    def __init__(self, state_dim, action_dim, action_bound):   # (3, 1, 2)
        super(Actor, self).__init__()

        self.action_bound = action_bound

        self.h1 = nn.Linear(state_dim, 64)
        self.h1a = nn.ReLU()
        self.h2 = nn.Linear(64,32)
        self.h2a = nn.ReLU()
        self.h3 = nn.Linear(32,16)
        self.h3a = nn.ReLU()
        self.action = nn.Linear(16, action_dim)
        self.actiona = nn.Tanh()

    def forward(self, state):
        x = self.h1(state)
        x = self.h1a(x)
        x = self.h2(x)
        x = self.h2a(x)
        x = self.h3(x)
        x = self.h3a(x)
        a = self.action(x)
        a = self.actiona(a)


        a = torch.mul(a, self.action_bound)

        return a
    



## 크리틱 신경망
class Critic(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.x1 = nn.Linear(state_dim, 32)
        self.x1a = nn.ReLU()
        
        self.a1 = nn.Linear(action_dim, 32)
        self.a1a = nn.ReLU()
        
        self.h2 = nn.Linear(64,32)
        self.h2a = nn.ReLU()
        self.h3 = nn.Linear(32,16)
        self.h3a = nn.ReLU()
        self.q = nn.Linear(16,1)


    def forward(self, state_action):
        state = state_action[0]
        action = state_action[1]
        x = self.x1(state)
        x = self.x1a(x)
        a = self.a1(action)
        a = self.a1a(a)

        h = torch.cat([x,a], dim=-1)

        x = self.h2(h)
        x = self.h2a(x)
        x = self.h3(x)
        x = self.h3a(x)
        q = self.q(x)
        return q
    


    ## DDPG 에이전트
class DDPGagent(object):

    def __init__(self, parameters):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"


        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {self.device} device")


        self.state_dim = parameters['state_dim']
        self.action_dim = parameters['action_dim']
        self.action_bound = parameters['action_bound']

        self.GAMMA = parameters["gamma"]
        self.BATCH_SIZE = parameters["batch_size"]
        self.BUFFER_SIZE = parameters["buffer_size"]
        self.ACTOR_LEARNING_RATE = parameters['learning_rate']['actor']     # 0.0003
        self.CRITIC_LEARNING_RATE = parameters['learning_rate']['critic']   # 0.0003
        self.TAU = parameters["tau"]



        self.actor = Actor(self.state_dim, self.action_dim, self.action_bound).to(self.device)  # (3, 1, 2)
        self.target_actor = Actor(self.state_dim, self.action_dim, self.action_bound).to(self.device) # (3, 1. 2)
        self.critic = Critic(self.state_dim, self.action_dim).to(self.device)
        self.target_critic = Critic(self.state_dim, self.action_dim).to(self.device)

        self.Actor_optimizer =  torch.optim.Adam(self.actor.parameters(), lr=self.ACTOR_LEARNING_RATE)
        self.Critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.CRITIC_LEARNING_RATE)

        self.buffer = ReplayBuffer(self.BUFFER_SIZE)

        self.save_epi_reward = []



    ## 신경망의 파라미터값을 타깃 신경망으로 복사
    def update_target_network(self, TAU):
        theta_paramter = self.actor.state_dict()
        theta_idx = theta_paramter.keys()
        theta = list(theta_paramter.values())

        target_theta = self.target_actor.state_dict()
        target_theta = list(target_theta.values())

        for i in range(len(theta)):
            target_theta[i] = TAU * theta[i] + (1 - TAU) * target_theta[i]
        target_theta = dict(zip(theta_idx, target_theta))
        self.target_actor.load_state_dict(target_theta)


        phi_parameter = self.critic.state_dict()
        phi_idx = phi_parameter.keys()
        phi = list(phi_parameter.values())
        target_phi = self.target_critic.state_dict()
        target_phi = list(target_phi.values())

        for i in range(len(phi)):
            target_phi[i] = TAU * phi[i] + (1 - TAU) * target_phi[i]
        target_phi = dict(zip(phi_idx, target_phi))
        self.target_critic.load_state_dict(target_phi)

    ## 크리틱 신경망 학습
    def critic_learn(self, states, actions, td_targets):
        self.critic.train()
        q = self.critic([states, actions])

        loss = torch.mean((q-td_targets)**2)
        self.Critic_optimizer.zero_grad()
        loss.backward()
        self.Critic_optimizer.step()


    ## 액터 신경망 학습
    def actor_learn(self, states):
        self.actor.train()

        actions = self.actor(states)

        critic_q = self.critic([states, actions])
        loss = -torch.mean(critic_q)

        self.Actor_optimizer.zero_grad()
        loss.backward()
        self.Actor_optimizer.step()

    ## Ornstein Uhlenbeck 노이즈
    def ou_noise(self, x, rho=0.15, mu=0, dt=1e-1, sigma=0.2, dim=1):
        return x + rho*(mu - x)*dt + sigma*np.sqrt(dt)*np.random.normal(size=dim)

    ## TD 타깃 계산
    def td_target(self, rewards, q_values, dones):
        y_k = np.asarray(q_values)
        for i in range(q_values.shape[0]): # number of batch
            if dones[i]:
                y_k[i] = rewards[i]
            else:
                y_k[i] = rewards[i] + self.GAMMA * q_values[i]
        return y_k
    

