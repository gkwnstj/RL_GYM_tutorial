import gym
from DDPG import * 
from reporter import *

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
