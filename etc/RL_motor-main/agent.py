import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.layers import Dense, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomUniform

import random

def shape_check(array, shape):
    assert array.shape == shape, \
        'shape error | array.shape ' + str(array.shape) + ' shape: ' + str(shape)

class Actor(tf.keras.Model):                                                            # model = tf.keras.Model(inputs=inputs, outputs=outputs)    
    def __init__(self, action_dim, action_bound, state_dim):                            
        super(Actor, self).__init__()
        self.action_dim = action_dim                                                    
        self.action_bound = action_bound                                                
        self.state_dim = state_dim                                                      

        self.fc1 = Dense(200, activation='leaky_relu', input_shape=(self.state_dim,))         
        self.fc2 = Dense(200, activation='leaky_relu')
        self.fc3 = Dense(200, activation='leaky_relu')
        self.fc4 = Dense(200, activation='leaky_relu')
        self.fc_mu = Dense(                                                             # baseline?
            self.action_dim,                                                            # 1
            activation='tanh',                                                          # [-1, 1]
            kernel_initializer=RandomUniform(-1e-3, 1e-3)                               # Random Initialization
        )
        self.fc_std = Dense(                                                            # possible arrange of action using std
            self.action_dim,                                                            # 1
            activation='softplus',                                                      # 매끄러운 relu
            kernel_initializer=RandomUniform(-1e-3, 1e-3)                               
        )

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        mu = self.fc_mu(x)
        std = self.fc_std(x) 
        mu = Lambda(lambda x: (x+1)/2 * self.action_bound)(mu)    #         mu = Lambda(lambda x: (x+1)/2 * self.action_bound)(mu)                                
                              # action_bound = 10, x = mu -> cliping?
        #mu = Lambda(lambda x: x* self.action_bound)(mu)
        return mu, std

class Critic(tf.keras.Model):                                                           # model = tf.keras.Model(inputs=inputs, outputs=outputs)
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.state_dim = state_dim                                                      # 6    

        self.fc1 = Dense(200, activation='leaky_relu', input_shape=(self.state_dim,))         # (6,)
        self.fc2 = Dense(200, activation='leaky_relu')
        self.fc3 = Dense(200, activation='leaky_relu')
        self.fc4 = Dense(200, activation='leaky_relu')
        self.fc_out = Dense(                # If you don't specify anything, no activation is applied (ie. "linear" activation: a(x) = x).
            1, # value 하나만 output 이기 때문에
            kernel_initializer=RandomUniform(-1e-3, 1e-3)
        )

    def call(self, x):     # (self, inputs)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        y = self.fc_out(x)
        return y

class PPO_Agent:
    def __init__(self, parameters, train_name, load_network_path):  # agent_param, args.train_name, args.load_network_path

        self.render = False
        
        # env parameters
        self.state_dim = parameters['state_dim']                # 6 
        self.action_dim = parameters['action_dim']              # 1
        self.action_bound = 10                             # pi/6
        self.std_bound = [0.01, 1.0]    # should be considered by activation function    # 0.01 ~ 1.0

        self.Actor_learning_rate = parameters['learning_rate']['actor']     # 0.0003
        self.Critic_learning_rate = parameters['learning_rate']['critic']   # 0.0003
        self.gamma = parameters['gamma']                                    # 0.99
        self.RATIO_CLIPPING = parameters['clip_ratio'] # clip coefficient   # 0.2
        self.MAX_BUFFER_SIZE = parameters['max_buffer_size']                # 2048
        self.batch_size = parameters['batch_size']                          # 512
        self.EPOCH = parameters['epoch']                                    # 10
        self.GAE_param = parameters['GAE_param']                            # 0.9
        # buffer
        self.buffer = []                                                    # Define the buffer using list type

        if load_network_path == 'NOT_LOADED':                               # default : NOT_LOADED
            self.Actor_model = Actor(self.action_dim, self.action_bound, self.state_dim)    # 1, 10, 1
            self.Critic_model = Critic(self.state_dim)
            self.Actor_model.build(input_shape=(None, self.state_dim))      # ??
            self.Critic_model.build(input_shape=(None, self.state_dim))     # ??
            self.Actor_optimizer = Adam(
                learning_rate= self.Actor_learning_rate,                    # 0.0003
                # clipnorm=1.0
            )    
            self.Critic_optimizer = Adam(
                learning_rate= self.Critic_learning_rate,                   # 0.0003
                # clipnorm=1.0
            )
            self.Actor_model.compile(self.Actor_optimizer)                  # optimizing the actor
            self.Critic_model.compile(self.Critic_optimizer)                # optimizing the critic

            self.model_path = './trained_model'
            print("self.model_path : ", self.model_path)
            
        else:                                                                                       ## Load the pretrained_network

            self.model_path = load_network_path

            self.Actor_model = Actor(self.action_dim, self.action_bound, self.state_dim)    # 1, 10, 1
            self.Critic_model = Critic(self.state_dim)
            self.Actor_model.build(input_shape=(None, self.state_dim))      # ??
            self.Critic_model.build(input_shape=(None, self.state_dim))     # ??
            
            self.Actor_model.load_weights("D:/Simulink/RL_MOTOR/RL_Traction_motor/traction_motor_RL/trained_model/04061935/04061935E1/Actor_weights.h5")             
            self.Critic_model.load_weights("D:/Simulink/RL_MOTOR/RL_Traction_motor/traction_motor_RL/trained_model/04061935/04061935E1/Critic_weights.h5")             

            self.Actor_model.compile(Adam(lr=self.Actor_learning_rate))
            self.Critic_model.compile(Adam(lr=self.Critic_learning_rate))

            self.Actor_model.set_weights(self.Actor_model.get_weights())
            self.Critic_model.set_weights(self.Critic_model.get_weights())

            self.model_path = load_network_path
            


        self.writer = tf.summary.create_file_writer('./summary/' + train_name)
        
    def log_pdf(self, mu, std, action):                                                 # What is it?
        std = tf.clip_by_value(std, self.std_bound[0], self.std_bound[1])               # clip the value, std_bound[0] : 0.01, std_bound[1] : 1
        var = std**2
        log_policy_pdf = -0.5 * (action - mu) ** 2 / var - 0.5 * tf.math.log(var * 2 * np.pi)
        return tf.reduce_sum(log_policy_pdf, 1, keepdims=True)

    def get_action(self, state):
        mu, std = self.Actor_model(                                                     # Get the range of action in continuous state using mean and std 
            tf.convert_to_tensor(state)
        )
        mu = mu.numpy()[0]
        std = std.numpy()[0]
        clipped_std = np.clip(std, self.std_bound[0], self.std_bound[1])                # clip the std
        # print(self.action_dim, mu,  std)
        action = np.random.normal(mu, clipped_std, size = self.action_dim) # 주어진 평균과 분산에 기반하여 랜덤 샘플링 -> 평균을 잘 근사하는 것이 좋을 것 같다.             # action in the range of action with clipped std
        # print(mu, std,clipped_std, action)
        action = np.clip(action, 0.1, self.action_bound)                 # clip the actual action with the action_bound (pi/6)
        return action                                                                   # return clipped action
    
    def sample_append(self, state, action, reward, next_state, done):
        self.buffer.append(                                             # self.buffer = [], unit_length : 6
            [
                state,                                                  # s
                action,                                                 # a
                reward,                                                 # r
                next_state,                                             # s'
                done                                                    # what is it?
            ]
        )

    def GAE_target(self, state_list, reward_list, next_state_list, done_list):
        value = self.Critic_model(
            tf.convert_to_tensor(state_list)
        )
        next_value = self.Critic_model(
            tf.convert_to_tensor(next_state_list)
        )
        reward_list = np.reshape(reward_list, [self.batch_size, 1])     # r
        done_list = np.reshape(done_list, [self.batch_size, 1])         # done
        value = value.numpy()                                           # state list
        next_value = next_value.numpy()                                 # s'

        done = done_list[self.batch_size-1][0]                          # extract the last done
        # next_value = next_value[self.MAX_BUFFER_SIZE-1][0]
        # print(done)
        if done:
            # print("0s")
            next_value[self.MAX_BUFFER_SIZE-1][0] = 0                   # next_state_list

        delta_list = reward_list + self.gamma * next_value - value
        delta_list = np.flip(delta_list)
        
        # print(delta_list.shape, next_value.shape, value.shape)

        GAE = []
        for i, delta in enumerate(delta_list):
            if i == 0:
                GAE.append(delta)
            else:
                GAE.append(delta + self.gamma * self.GAE_param * GAE[i-1])
        GAE = np.flip(GAE)                                              # flip the array
        GAE = np.reshape(GAE, [self.batch_size, 1])
        
        target = GAE + value
        # print(GAE.shape, value.shape)
        return GAE, target

    def GAE_target_test(self, state_list, reward_list, next_state, done):
        predict = self.Critic_model(
            tf.convert_to_tensor(state_list)
        )
        predict = predict.numpy()
        # shape_check(predict, (self.batch_size, 1))
        next_value = 0
        next_GAE = 0

        if not done:
            next_value = self.Critic_model(
                tf.convert_to_tensor(
                    np.reshape(next_state, [1, self.state_dim])
                )
            )
            next_value = next_value.numpy()
            # shape_check(next_value, (1, 1))
        
        reward_list = np.reshape(reward_list, [self.batch_size, 1])
        # shape_check(reward_list, (self.batch_size, 1))
        delta = np.zeros_like(reward_list)
        GAE = np.zeros_like(reward_list)

        for i in reversed(range(0, self.batch_size)):
            delta[i] = reward_list[i] + self.gamma * next_value - predict[i]
            GAE[i] = delta[i] + self.gamma * self.GAE_param * next_GAE
            next_value = predict[i]
            next_GAE = GAE[i]

        target = GAE + predict
        # shape_check(target, (self.batch_size, 1))
        return GAE, target

    def Critic_train(self, target_list, state_list):
        model_params = self.Critic_model.trainable_variables
        with tf.GradientTape() as tape:
            predict = self.Critic_model(
                tf.convert_to_tensor(state_list)
            )
            advantage = target_list - predict
            loss = tf.reduce_mean(tf.square(advantage))
            # print(target_list.shape, predict.shape, advantage.shape)
        grads = tape.gradient(loss, model_params)
        self.Critic_optimizer.apply_gradients(zip(grads, model_params))

    def Actor_train(self, state_list, action_list, old_log_pdf, GAE):
        model_params = self.Actor_model.trainable_variables
        with tf.GradientTape() as tape:
            mu, std = self.Actor_model(
                tf.convert_to_tensor(state_list)
            )
            log_pdf = self.log_pdf(mu, std, action_list)
            ratio = tf.exp(log_pdf - old_log_pdf)
            clipped_ratio = tf.clip_by_value(ratio, 1.0-self.RATIO_CLIPPING, 1.0+self.RATIO_CLIPPING)
            surrogate = -tf.minimum(ratio * GAE, clipped_ratio * GAE)
            loss = tf.reduce_mean(surrogate)
            # print(surrogate.shape, loss)
        grads = tape.gradient(loss, model_params)
        self.Actor_optimizer.apply_gradients(zip(grads, model_params))            

    def train(self):

        if len(self.buffer) < self.MAX_BUFFER_SIZE:
            return
        
        for _ in range(self.EPOCH):
            # print("train!")  
            batch = random.sample(self.buffer, self.batch_size) 
            state_list = [sample[0][0] for sample in batch]
            action_list = [sample[1][0] for sample in batch]
            reward_list = [sample[2][0] for sample in batch]
            next_state_list = [sample[3][0] for sample in batch]
            done_list = [sample[4][0] for sample in batch]

            # GAE, target = self.GAE_target(state_list, reward_list, next_state_list, done_list)
            GAE, target = self.GAE_target_test(state_list, reward_list, next_state_list[self.batch_size-1], done_list[self.batch_size-1])

            mu, std = self.Actor_model(
                tf.convert_to_tensor(state_list)
            )
            old_log_pdf = self.log_pdf(mu, std, action_list)        

            self.Critic_train(target, state_list)
            self.Actor_train(state_list, action_list, old_log_pdf, GAE)
            
        self.buffer.clear()

    # 텐서보드에 학습 정보를 기록
    def draw_tensorboard(self, score, step, episode):
        with self.writer.as_default():
            tf.summary.scalar('Total Reward/Episode', score, step=episode)
            tf.summary.scalar('Steps/Episode', step, step=episode)

    def saveImprovedWeight(self, now, episode):
        save_name = self.model_path + '/' + now + '/' +  now + 'E' + str(episode+1)

        if not os.path.exists(save_name):
            os.makedirs(save_name)
            
        print("save_name : ", save_name)
        print("[AGNT] Saved Improved Model. Model Name: {}".format(save_name))
        # self.Critic_model.fit()
        # self.Actor_model.fit()

        self.Critic_model.save_weights(save_name + '/Critic_weights.h5')
        self.Actor_model.save_weights(save_name + '/Actor_weights.h5')