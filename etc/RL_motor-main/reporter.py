import numpy as np
import matplotlib.pyplot as plt

class Reporter:
    def __init__(self) -> None:
        self.record = np.array([[]])            # data to array
        self.time = np.array([])                        

    def saveRecord(self, time, record):
        # record는 행벡터
        self.time = np.append(self.time, np.array([time]))
        #print("\nself.time : \n", self.time)                        
        self.record = np.append(
            self.record, np.array([record]), axis=1
        )

    def plotRecord(self, obsInfo):
        print('[RPTR] Plotting Simulation Observations')
        #self.time = np.array(self.time)
        #self.time = self.time[0][1:]
        self.time = self.time.reshape(len(self.time),1)
        #print("\nself.time_plotRecord : \n", self.time)
        self.record = self.record.reshape(
            self.time.shape[0], len(obsInfo)
        )
        obs_size = self.record.shape[1]
        
        for i in range(obs_size):
            plt.subplot(obs_size, 1, i+1)
            plt.plot(self.time[:,0].T, self.record[:,i])
            plt.title('{}'.format(obsInfo[i]))
            plt.xlabel('Time [s]')

        plt.show()
        print('[RPTR] Terminating')

class TrainReporter:
    def __init__(self) -> None:
        self.reward = None
        self.max_reward = -1e9      # 0.000000001, 1e-9        ### 

    def initReward(self):
        try:
            self.rewardRecord.append(self.reward)
        except:
            self.rewardRecord = []
        self.reward = 0

    def saveReward(self, reward):
        self.reward += reward

    def reportRewardRecord(self):
        print('[RPTR] EPISODE: {}| TOTAL REWARD: {}'.format(
            len(self.rewardRecord)+1, round(self.reward, 3)
        ))

    def plotRecord(self):
        print('[RPTR] Plotting Train Result')

        #print("self.rewardRecord : ", self.rewardRecord)
        
        plt.plot(self.rewardRecord)
        plt.title('Reward Report')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        
        plt.show()
        print('[RPTR] Terminating')

    def reportSession(self, args, train_param, agent_param):
        print("")
        print("[RPTR] Session Description")
        print("\tSimulink Model: {}".format(args.slx_name))
        print("\tLoaded Network: {}".format(args.load_network_path))
        print("\tTrain Session: {}".format(args.train))
        
        if args.train:
            print("\t\tMax Episode: {}".format(train_param['MAX_EPISODE']))
            print("\t\tMax Time: {}".format(train_param['MAX_TIME']))
            print("\t\tTime Difference: {}".format(train_param['TIME_DIFFERENCE']))

            print("\t\tAlgorithm: {}".format(agent_param['Algorithm']))
            print("\t\tLearning Rate (actor): {}".format(agent_param['learning_rate']['actor']))
            print("\t\tLearning Rate (critic): {}".format(agent_param['learning_rate']['critic']))
            print("\t\tGamma: {}".format(agent_param['gamma']))
            print("\t\tClip Ratio: {}".format(agent_param['clip_ratio']))
            print("\t\tMax Buffer Size: {}".format(agent_param['max_buffer_size']))
            print("\t\tBatch Size: {}".format(agent_param['batch_size']))
            print("\t\tEpoch: {}".format(agent_param['epoch']))
            print("\t\tGAE Param: {}".format(agent_param['GAE_param']))
        print("")      