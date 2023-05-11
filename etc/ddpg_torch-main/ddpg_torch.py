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

        #self.h1 = Dense(64, activation='relu')
        self.h1 = nn.Linear(state_dim, 64)
        self.h1a = nn.ReLU()
        #self.h2 = Dense(32, activation='relu')
        self.h2 = nn.Linear(64,32)
        self.h2a = nn.ReLU()
        #self.h3 = Dense(16, activation='relu')
        self.h3 = nn.Linear(32,16)
        self.h3a = nn.ReLU()
        #self.action = Dense(action_dim, activation='tanh')
        self.action = nn.Linear(16, action_dim)
        self.actiona = nn.Tanh()

    def forward(self, state):
        #print(state)      # tf.Tensor([[ 0.3031706  0.9529363 -0.3052618]], shape=(1, 3), dtype=float32)
        x = self.h1(state)
        x = self.h1a(x)
        x = self.h2(x)
        x = self.h2a(x)
        x = self.h3(x)
        x = self.h3a(x)
        a = self.action(x)
        a = self.actiona(a)

        # 행동을 [-action_bound, action_bound] 범위로 조정
        #a = Lambda(lambda x: x*self.action_bound)(a)
        a = torch.mul(a, self.action_bound)

        return a
    


## 크리틱 신경망
class Critic(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        #self.x1 = Dense(32, activation='relu')
        self.x1 = nn.Linear(state_dim, 32)
        self.x1a = nn.ReLU()
        
        #self.a1 = Dense(32, activation='relu')
        self.a1 = nn.Linear(action_dim, 32)
        self.a1a = nn.ReLU()
        
        #self.h2 = Dense(32, activation='relu')
        self.h2 = nn.Linear(64,32)
        self.h2a = nn.ReLU()
        #self.h3 = Dense(16, activation='relu')
        self.h3 = nn.Linear(32,16)
        self.h3a = nn.ReLU()
        #self.q = Dense(1, activation='linear')
        self.q = nn.Linear(16,1)


    def forward(self, state_action):
        #print("state_action : ", state_action)  # [<tf.Tensor 'Placeholder:0' shape=(None, 3) dtype=float32>, <tf.Tensor 'Placeholder_1:0' shape=(None, 1) dtype=float32>]
        state = state_action[0]
        action = state_action[1]
        x = self.x1(state)
        x = self.x1a(x)
        #print("x : ", x)    # x :  Tensor("critic/dense_8/Relu:0", shape=(None, 32), dtype=float32)
        a = self.a1(action)
        a = self.a1a(a)
        #print("a : ", a)   # a :  Tensor("critic/dense_9/Relu:0", shape=(None, 32), dtype=float32)
        #h = concatenate([x, a], axis=-1)
        h = torch.cat([x,a], dim=-1)
        #print("h : ", h)    # h :  Tensor("critic/concatenate/concat:0", shape=(None, 64), dtype=float32)  
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
        print(f"Using {self.device} device")
        
        # 하이퍼파라미터
        self.GAMMA = 0.95
        self.BATCH_SIZE = 32
        self.BUFFER_SIZE = 20000
        self.ACTOR_LEARNING_RATE = 0.0001
        self.CRITIC_LEARNING_RATE = 0.001
        self.TAU = 0.001
        # 환경
        self.env = env
        # 상태변수 차원
        self.state_dim = env.observation_space.shape[0]
        #print("self.state_dim : ", self.state_dim)   # self.state_dim :  3
        # 행동 차원
        self.action_dim = env.action_space.shape[0]
        #print("self.action_dim : ", self.action_dim)  # self.action_dim :  1
        # 행동의 최대 크기
        self.action_bound = env.action_space.high[0]
        #print("self.action_bound : ", self.action_bound)  # self.action_bound :  2.0

        # 액터, 타깃 액터 신경망 및 크리틱, 타깃 크리틱 신경망 생성
        self.actor = Actor(self.state_dim, self.action_dim, self.action_bound).to(self.device)  # (3, 1, 2)
        self.target_actor = Actor(self.state_dim, self.action_dim, self.action_bound).to(self.device) # (3, 1. 2)

        self.critic = Critic(self.state_dim, self.action_dim).to(self.device)
        self.target_critic = Critic(self.state_dim, self.action_dim).to(self.device)

        #self.actor.build(input_shape=(None, self.state_dim))
        #self.target_actor.build(input_shape=(None, self.state_dim))

        #state_in = Input((self.state_dim,))   # self.state_dim :  3
        #print("state_in : ", state_in)   # state_in :  KerasTensor(type_spec=TensorSpec(shape=(None, 3), dtype=tf.float32, name='input_1'), name='input_1', description="created by layer 'input_1'")
        
        #action_in = Input((self.action_dim,))  # self.action_dim :  1
        #print("action_in : ", action_in)   # action_in :  KerasTensor(type_spec=TensorSpec(shape=(None, 1), dtype=tf.float32, name='input_2'), name='input_2', description="created by layer 'input_2'")
        
        #state_in = torch.tensor(self.state_dim)
        #action_in = torch.tensor(self.action_dim)
        
        #self.critic([state_in, action_in]) # [state_in, action_in] -> state_action
        #print("[state_in, action_in] : ", [state_in, action_in]) # [state_in, action_in] :  [<KerasTensor: shape=(None, 3) dtype=float32 (created by layer 'input_1')>, <KerasTensor: shape=(None, 1) dtype=float32 (created by layer 'input_2')>]
        #self.target_critic([state_in, action_in]) 

        #self.actor.summary()
        #self.critic.summary()

        # 옵티마이저
        #self.actor_opt = Adam(self.ACTOR_LEARNING_RATE)
        #self.critic_opt = Adam(self.CRITIC_LEARNING_RATE)

        self.Actor_optimizer =  torch.optim.Adam(self.actor.parameters(), lr=self.ACTOR_LEARNING_RATE)
        self.Critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.CRITIC_LEARNING_RATE)


        # 리플레이 버퍼 초기화
        self.buffer = ReplayBuffer(self.BUFFER_SIZE)

        # 에피소드에서 얻은 총 보상값을 저장하기 위한 변수
        self.save_epi_reward = []



    ## 신경망의 파라미터값을 타깃 신경망으로 복사
    def update_target_network(self, TAU):
        #theta = self.actor.get_weights()
        theta_paramter = self.actor.state_dict()
        theta_idx = theta_paramter.keys()
        theta = list(theta_paramter.values())

        #print("theta_idx : ",theta_idx)
        #print("theta : ", len(theta))
        #target_theta = self.target_actor.get_weights()
        target_theta = self.target_actor.state_dict()
        target_theta = list(target_theta.values())
        #print("target_theta : ", target_theta)
        #print("len(theta) : ",len(theta))
        #print("len(target_theta) : ",len(target_theta))
        #print("theta[0]",theta[0][0])
        for i in range(len(theta)):
            target_theta[i] = TAU * theta[i] + (1 - TAU) * target_theta[i]
        #print("updated_target_theta : ", target_theta)
        #self.target_actor.set_weights(target_theta)
        target_theta = dict(zip(theta_idx, target_theta))
        self.target_actor.load_state_dict(target_theta)
        #print("updated_target_theta32342342 : ", self.actor.state_dict())




        #phi = self.critic.get_weights()
        phi_parameter = self.critic.state_dict()
        phi_idx = phi_parameter.keys()
        phi = list(phi_parameter.values())
        #target_phi = self.target_critic.get_weights()
        target_phi = self.target_critic.state_dict()
        target_phi = list(target_phi.values())

        for i in range(len(phi)):
            target_phi[i] = TAU * phi[i] + (1 - TAU) * target_phi[i]
        #self.target_critic.set_weights(target_phi)
        target_phi = dict(zip(phi_idx, target_phi))
        self.target_critic.load_state_dict(target_phi)



    ## 크리틱 신경망 학습
    # def critic_learn(self, states, actions, td_targets):
    #     with tf.GradientTape() as tape:
    #         q = self.critic([states, actions], training=True)
    #         loss = tf.reduce_mean(tf.square(q-td_targets))

    #     grads = tape.gradient(loss, self.critic.trainable_variables)
    #     self.critic_opt.apply_gradients(zip(grads, self.critic.trainable_variables))


    ## 크리틱 신경망 학습
    def critic_learn(self, states, actions, td_targets):
        self.critic.train()
        q = self.critic([states, actions])

        loss = torch.mean((q-td_targets)**2)
        self.Critic_optimizer.zero_grad()
        loss.backward()
        self.Critic_optimizer.step()


    # ## 액터 신경망 학습
    # def actor_learn(self, states):
    #     with tf.GradientTape() as tape:
    #         actions = self.actor(states, training=True)
    #         critic_q = self.critic([states, actions])
    #         loss = -tf.reduce_mean(critic_q)

    #     grads = tape.gradient(loss, self.actor.trainable_variables)
    #     self.actor_opt.apply_gradients(zip(grads, self.actor.trainable_variables))



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
                #print("state : ", state)
                # 환경 가시화
                #self.env.render()
                # 행동과 노이즈 계산
                #action = self.actor(tf.convert_to_tensor([state], dtype=tf.float32))
                action = self.actor(torch.as_tensor(state, device=self.device).type(torch.float32))
                #print("action : ", action)  # action :  tf.Tensor([[0.07077996]], shape=(1, 1), dtype=float32)
                #action = action.numpy()[0]
                action = action.detach().cpu().numpy()
                #print("action_numpy : ", action)  # action_numpy :  [-1.7684212]
                noise = self.ou_noise(pre_noise, dim=self.action_dim)
                # 행동 범위 클리핑
                action = np.clip(action + noise, -self.action_bound, self.action_bound)
                # 다음 상태, 보상 관측
                next_state, reward, done, _ = self.env.step(action)
                # 학습용 보상 설정
                train_reward = (reward + 8) / 8
                # 리플레이 버퍼에 저장
                self.buffer.add_buffer(state, action, train_reward, next_state, done)

                # 리플레이 버퍼가 일정 부분 채워지면 학습 진행
                if self.buffer.buffer_count() > 1000:

                    # 리플레이 버퍼에서 샘플 무작위 추출
                    states, actions, rewards, next_states, dones = self.buffer.sample_batch(self.BATCH_SIZE)
                    # 타깃 크리틱에서 행동가치 계산
                    #target_qs = self.target_critic([tf.convert_to_tensor(next_states, dtype=tf.float32),
                    #                                self.target_actor(
                    #                                    tf.convert_to_tensor(next_states, dtype=tf.float32))])
                    
                    target_qs = self.target_critic([torch.as_tensor(next_states, device = self.device).type(torch.float32),
                                                    self.target_actor(
                                                        torch.as_tensor(next_states, device = self.device).type(torch.float32))])
                    
                    #print("target_qs : ", target_qs)
                    

                    # TD 타깃 계산
                    #y_i = self.td_target(rewards, target_qs.numpy(), dones)
                    y_i = self.td_target(rewards, target_qs.detach().cpu().numpy(), dones)
                    
                    # 크리틱 신경망 업데이트
                    #self.critic_learn(tf.convert_to_tensor(states, dtype=tf.float32),
                    #                  tf.convert_to_tensor(actions, dtype=tf.float32),
                    #                  tf.convert_to_tensor(y_i, dtype=tf.float32))
                    
                    self.critic_learn(torch.as_tensor(states, device = self.device).type(torch.float32),
                                      torch.as_tensor(actions, device = self.device).type(torch.float32),
                                      torch.as_tensor(y_i, device = self.device).type(torch.float32))
                    
                    # 액터 신경망 업데이트
                    self.actor_learn(torch.as_tensor(states, device = self.device).type(torch.float32))
                    # 타깃 신경망 업데이트
                    self.update_target_network(self.TAU)

                # 다음 스텝 준비
                pre_noise = noise
                state = next_state
                episode_reward += reward
                time += 1

            # 에피소드마다 결과 보상값 출력
            print('Episode: ', ep+1, 'Time: ', time, 'Reward: ', episode_reward)
            self.save_epi_reward.append(episode_reward)


            # 에피소드마다 신경망 파라미터를 파일에 저장
            #self.actor.save_weights("./save_weights/pendulum_actor.h5")
            #self.critic.save_weights("./save_weights/pendulum_critic.h5")

        # 학습이 끝난 후, 누적 보상값 저장
        np.savetxt('./save_weights/pendulum_epi_reward.txt', self.save_epi_reward)
        #print(self.save_epi_reward)


    ## 에피소드와 누적 보상값을 그려주는 함수
    def plot_result(self):
        plt.plot(self.save_epi_reward)
        plt.show()





# DDPG main (tf2 subclassing API version)
# coded by St.Watermelon
## DDPG 에이전트를 학습하고 결과를 도시하는 파일

# 필요한 패키지 임포트
import gym

def main():

    max_episode_num = 1000  # 최대 에피소드 설정
    env = gym.make("Pendulum-v0")
    agent = DDPGagent(env)  # DDPG 에이전트 객체

    # 학습 진행
    agent.train(max_episode_num)

    # 학습 결과 도시
    agent.plot_result()


if __name__=="__main__":
    main()

