import random
import gym
from collections import deque
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn, cat
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from replaybuffer import ReplayBuffer


## 액터 신경망
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

    def __init__(self, env):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"


        self.GAMMA = 0.95
        self.BATCH_SIZE = 32
        self.BUFFER_SIZE = 20000
        self.ACTOR_LEARNING_RATE = 0.0001
        self.CRITIC_LEARNING_RATE = 0.001
        self.TAU = 0.001

        self.env = env

        self.state_dim = env.observation_space.shape[0]


        self.action_dim = env.action_space.shape[0]


        self.action_bound = env.action_space.high[0]
        print("self.action_bound : ", self.action_bound)  # self.action_bound :  2.0


        self.actor = Actor(self.state_dim, self.action_dim, self.action_bound).to(self.device)  # (3, 1, 2)
        self.target_actor = Actor(self.state_dim, self.action_dim, self.action_bound).to(self.device) # (3, 1. 2)
        self.critic = Critic(self.state_dim, self.action_dim).to(self.device)
        self.target_critic = Critic(self.state_dim, self.action_dim).to(self.device)

        self.Actor_optimizer =  torch.optim.Adam(self.actor.parameters(), lr=self.ACTOR_LEARNING_RATE)
        self.Critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.CRITIC_LEARNING_RATE)

        self.buffer = ReplayBuffer(self.BUFFER_SIZE)

        self.save_epi_reward = [-1e4]



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

    ## 신경망 파라미터 로드
    def load_weights(self, path):
        self.actor.load_weights(path + 'pendulum_actor.h5')
        self.critic.load_weights(path + 'pendulum_critic.h5')


    ## 에이전트 학습
    def train(self, max_episode_num):

        # 타깃 신경망 초기화
        self.update_target_network(1.0)

        # 에피소드마다 다음을 반복
        for ep in range(int(max_episode_num)):
            # OU 노이즈 초기화
            pre_noise = np.zeros(self.action_dim)
            # 에피소드 초기화
            time, episode_reward, done = 0, 0, False
            # 환경 초기화 및 초기 상태 관측
            state = self.env.reset()

            while not done:

                action = self.actor(torch.as_tensor(state, device=self.device).type(torch.float32))

                action = action.detach().cpu().numpy()
                noise = self.ou_noise(pre_noise, dim=self.action_dim)
                action = np.clip(action + noise, -self.action_bound, self.action_bound)
                next_state, reward, done, _ = self.env.step(action)
                #print("reward : ", reward)
                train_reward = (reward + 8) / 8
                self.buffer.add_buffer(state, action, train_reward, next_state, done)

                if self.buffer.buffer_count() > 1000:

                    states, actions, rewards, next_states, dones = self.buffer.sample_batch(self.BATCH_SIZE)

                    #print("buffer rewards : ", rewards)

                    target_qs = self.target_critic([torch.as_tensor(next_states, device = self.device).type(torch.float32),
                                                    self.target_actor(
                                                        torch.as_tensor(next_states, device = self.device).type(torch.float32))])
                    
                    y_i = self.td_target(rewards, target_qs.detach().cpu().numpy(), dones)
                    

                    
                    self.critic_learn(torch.as_tensor(states, device = self.device).type(torch.float32),
                                      torch.as_tensor(actions, device = self.device).type(torch.float32),
                                      torch.as_tensor(y_i, device = self.device).type(torch.float32))
                    
                    self.actor_learn(torch.as_tensor(states, device = self.device).type(torch.float32))
                    self.update_target_network(self.TAU)

                # 다음 스텝 준비
                pre_noise = noise
                state = next_state
                episode_reward += reward
                time += 1

            print('Episode: ', ep+1, 'Time: ', time, 'Reward: ', episode_reward)

            if episode_reward > self.save_epi_reward[-1]:           # -1로 하면 자기 자신과 비교하게 됨.
                print(" [ Save improved model ] ")
                self.saveasONNX()

            self.save_epi_reward.append(episode_reward)
            print("self.save_epi_reward : ", self.save_epi_reward)
            print("self.save_epi_reward[-1] : ", self.save_epi_reward[-1])
            print("episode_reward : ", episode_reward)


        np.savetxt('./save_weights/pendulum_epi_reward.txt', self.save_epi_reward)
        #print(self.save_epi_reward)




    ## 에피소드와 누적 보상값을 그려주는 함수
    def plot_result(self):
        print("self.save_epi_reward : ", self.save_epi_reward)
        plt.plot(self.save_epi_reward[1:])
        plt.show()

    def saveasONNX(self):
        self.actor.eval()
        self.critic.eval()

        dummy_state = torch.randn(1, 3, device=self.device, requires_grad=True)
        dummy_action = torch.randn(1, 1, device=self.device, requires_grad=True)        
        torch.onnx.export(
            self.actor, 
            dummy_state, 
            "DDPG_actor.onnx", 
            verbose=False,
            input_names= ['input']
            )
        torch.onnx.export(
            self.critic, 
            [dummy_state, dummy_action], 
            "DDPG_critic.onnx", 
            verbose=False
            )         
        


import gym

def main():

    max_episode_num = 10  # 최대 에피소드 설정
    env = gym.make("Pendulum-v0")
    agent = DDPGagent(env)  # DDPG 에이전트 객체

    # 학습 진행
    agent.train(max_episode_num)

    # 학습 결과 도시
    agent.plot_result()


if __name__=="__main__":
    main()

