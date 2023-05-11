from simManager import SimManager
from reporter import Reporter, TrainReporter
from others import randomVariables, rewardCalc
from agent import PPO_Agent
import argparse
from datetime import datetime
import json

def main(args):
    print("[MAIN] Main Function Called")

    with open(args.param_path) as f:
        param = json.load(f)                    
    train_param = param['train_param'] 
    agent_param = param['agent_param']

    modelName = args.slx_name
    DO_TRAIN = args.train
    MAX_EPISODE = train_param['MAX_EPISODE']        # 500000
    MAX_TIME = train_param['MAX_TIME']              # 0.25
    Ts = train_param['TIME_DIFFERENCE']             # 0.0001
    max_steps = int(MAX_TIME / Ts)                  # 0.25/0.0001 = 2500

    # obsInfo = [                                     # Workspace
    #     'y', 
    #     'dot_y',
    #     'psi',
    #     'r',
    #     'X',
    #     'Y',
    #     'X_ref',
    #     'Y_ref',
    #     'input'
    # ]

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
    

    reporter = Reporter()                           # Plotting
    trainReporter = TrainReporter()                 # Training Session
    trainReporter.reportSession(args, train_param, agent_param)

    agent = PPO_Agent(agent_param, args.train_name, args.load_network_path)

    #ref = referenceGenerator(max_steps, Ts) # (10000, 0.01)     # max_steps = int(MAX_TIME / Ts) # 100/0.01 = 10000

    #print("print ref : ", ref)
    sim = SimManager(modelName)
    sim.connectMatlab()

    print('[MAIN] Reset Simulation')

    if DO_TRAIN == False:                        # pretrained NN           # DO_TRAIN = args.train
        print('[MAIN] Demostrain Start')
        MAX_EPISODE = 1
    else:
        print('[MAIN] Train Start')

    for episode in range(MAX_EPISODE):
        step = 1
        n = 1           # for sampling time period
        initial_param = randomVariables()                   # other.py
    
        obs = sim.reset(obsInfo, initial_param)       # 0.0000s state   # tout, obs = reset the value for per episode
        #print("after reset obs : ", obs)
        trainReporter.initReward()                          # update the reward

        #state, _, _ = rewardCalc(obs, ref[0,:])             # org(state, reward, isDone) return only state
        #print(obs)
        state, _, _ = rewardCalc(obs)    
        while True:                                         # step debug

            action = agent.get_action(state)                # action from the state
            #print("action!")
            #print("currnet_Kp_id : ", action)
            abj_parameters = {                          # How to get the abj_parameters?
                
                'Kp_id': {'Gain': action}              ####################### ACTION #######################
            }

            # get observation
            obs, time = sim.step(obsInfo, abj_parameters, n) ### 0.0001 state obsInfo, 다음 상태를 반환해야함 ## obs 두개를 반환함 -> time_step 문제인거 같음 x -> workspace 조정
            #_, obs = sim.step(obsInfo, abj_parameters)

            #print("time :", time)                   ############################################
            
            #print("next_obs : ", obs)

            #next_state, reward, isDone = rewardCalc(obs, ref[step,:])
            #if time % 0.0001 == 0:
            next_state, reward, isDone = rewardCalc(obs)
        
        


            #print(next_state, reward, isDone)

            if isDone[0][0] or step == max_steps:    # max_step = 5
                break

            if DO_TRAIN:                                                              # agent = PPO_Agnet
                agent.sample_append(state, action, reward, next_state, isDone)       
                agent.train()
                agent.draw_tensorboard(reward[0][0], step, episode)

            else:
                #print("\ntime_type\n", time, type(time))
                reporter.saveRecord(time, obs)
                #reporter.saveRecord(obs)

            step += 1
            n += 1
            
            state = next_state
            #print("reward : ", reward)
            trainReporter.saveReward(reward[0][0])

        if DO_TRAIN:
            trainReporter.reportRewardRecord()

            if trainReporter.reward > trainReporter.max_reward:         # 여기를 만족해야 파일 생성
                trainReporter.max_reward = trainReporter.reward
                
                agent.saveImprovedWeight(now, episode)
      
    if DO_TRAIN:
        #print("reward_list : ", reward)
        #print("[MAIN] Train Finished. Final Reward: {}".format(reward))
        #print("trainReporter.rewardRecord : ",trainReporter.rewardRecord)
        trainReporter.plotRecord()
    else:
        print("[MAIN] Demostration Finished. Reward: {}".format(round(trainReporter.reward),3))
        reporter.plotRecord(obsInfo)
    print('[MAIN] Simulation Finished')
    sim.disconnectMatlab()

if __name__ == "__main__":
    now = datetime.now()
    now = now.strftime('%m%d%H%M')

    parser = argparse.ArgumentParser(description = "Available Options")
    parser.add_argument('--train_name', type=str,
                        default=now, dest="train_name", action="store",
                        help='trained model would be saved in typed directory')
    parser.add_argument('--slx_name', type=str,
                        default='RL_Traction_motor_2022b', dest='slx_name', action="store",
                        help='simulink model that you want to use as envirnoment')
    parser.add_argument('--train',
                        default=False, dest='train', action='store_true',
                        help='type this arg to train')
    parser.add_argument('--load_network', type=str,
                        default='NOT_LOADED', dest='load_network_path', action='store',
                        help='to load trained network, add path')
    parser.add_argument('--parameters', type=str,
                        default='./train_param.json', dest='param_path', action="store",
                        help='type path of train parameters file (json)')

    args = parser.parse_args()

    main(args)
