from DDPG import * 
import json
import others
import numpy as np
from replaybuffer import ReplayBuffer
import matplotlib.pyplot as plt
from save_torch_model import saveasONNX


def train(args, env):

    GPUdevice = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {GPUdevice} device in training")

    with open(args.param_path) as f:
        param = json.load(f)  

    agent_param = param['agent_param']
    train_param = param['train_param']
    buffer_size = agent_param["buffer_size"]
    BATCH_SIZE = agent_param["batch_size"]
    TAU = agent_param["tau"]

    state_dim = agent_param['state_dim']
    action_dim = agent_param['action_dim']
    action_bound = agent_param['action_bound']   

    MAX_EPISODE = train_param["MAX_EPISODE"]         # 500000
    MAX_TIME = train_param["MAX_TIME"]              # 0.25
    Ts = train_param["TIME_DIFFERENCE"]            # 0.0001
    max_steps = int(MAX_TIME / Ts)

    print("[TRAIN START]")

    agent = DDPGagent(agent_param)

    actor = Actor(state_dim, action_dim, action_bound).to(GPUdevice)  # (3, 1, 2)        # Model
    target_actor = Actor(state_dim, action_dim, action_bound).to(GPUdevice) # (3, 1. 2)
    critic = Critic(state_dim, action_dim).to(GPUdevice)
    target_critic = Critic(state_dim, action_dim).to(GPUdevice)
    
    buffer = ReplayBuffer(buffer_size)

    obsInfo = [                     ### 
            'id_LPF',
            'iq_LPF',                                                              # Workspace
            'id_sens',
            'iq_sens',
            'Vd_out',
            'Vq_out',
            'wrm',
            'angle',
            'trqMotor',
            'demand_trq'
        ]
    

    print('[MAIN] Train Start')

    save_epi_reward = [-1e7]
    all_epi_reward = []

    agent.update_target_network(1.0)

    for ep in range(MAX_EPISODE):
        step = 1
        n = 1

        initial_param = others.randomVariables()
        obs, _ = env.reset(obsInfo, initial_param)

        pre_noise = np.zeros(action_dim)
        time, episode_reward, done = 0, 0, False 

        state, _, _ = others.rewardCalc(obs)



        while True:
            action = actor(torch.as_tensor(state, device=GPUdevice).type(torch.float32))
            action = action.detach().cpu().numpy() 


            noise = agent.ou_noise(pre_noise, dim=action_dim)
            # 행동 범위 클리핑
            action = np.clip(action + noise, -action_bound, action_bound)
            # 다음 상태, 보상 관측
            
            abj_parameters = {                          # How to get the abj_parameters?
            'Kp_id': {'Gain': action}              ####################### ACTION #######################
            }

            obs, time = env.step(obsInfo, abj_parameters, n)

            print("time : ", time)

            next_state, reward, done = others.rewardCalc(obs)


            if done[0][0] or step == max_steps:    # max_step = 5
                break

            reward = reward[0][0]           ## [0][0]을 붙이지 않아도 프로그램이 돌아감 왜그럴까?

            train_reward = reward

            print("reward : ", train_reward)

            buffer.add_buffer(state, action, train_reward, next_state, done)

            if buffer.buffer_count() > 2:        # 1000
                print(" Buffer Training Start ")
                    
                # 리플레이 버퍼에서 샘플 무작위 추출
                states, actions, rewards, next_states, dones = buffer.sample_batch(BATCH_SIZE)
                print("buffer rewards : ", rewards)

                target_qs = target_critic([torch.as_tensor(next_states, device = GPUdevice).type(torch.float32),
                                                target_actor(
                                                    torch.as_tensor(next_states, device = GPUdevice).type(torch.float32))])
                

                

                y_i = agent.td_target(rewards, target_qs.detach().cpu().numpy(), dones)
                
                agent.critic_learn(torch.as_tensor(states, device = GPUdevice).type(torch.float32),
                                    torch.as_tensor(actions, device = GPUdevice).type(torch.float32),
                                    torch.as_tensor(y_i, device = GPUdevice).type(torch.float32))
                
                agent.actor_learn(torch.as_tensor(states, device = GPUdevice).type(torch.float32))
                agent.update_target_network(TAU)
            
            pre_noise = noise
            state = next_state
            episode_reward += reward
            time += 1
            step += 1


        print('Episode: ', ep+1, 'Time: ', time, 'Reward: ', episode_reward)

        if episode_reward > save_epi_reward[-1]:
            print(" [ Save improved model ] ")
            saveasONNX(actor, critic, GPUdevice)  
        
            save_epi_reward.append(episode_reward)
            print("save_epi_reward[-1] : ", save_epi_reward[-1])        # -1로 하면 자기 자신과 비교하게 됨.
            
        all_epi_reward.append(episode_reward)
      

        if (ep+1) % 10 == 0:
        # agent.Actor_model.save_weights('DDPG', save_format='tf')
            plt.plot(all_epi_reward)
            plt.savefig('./DDPG.png')



    print('[MAIN] Simulation Finished')
    print("save_epi_reward : ", save_epi_reward)

    plt.plot(all_epi_reward[0:])
    plt.savefig('./DDPG.png')
    plt.show(block=False)
    













    
