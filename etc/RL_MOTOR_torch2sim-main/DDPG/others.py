import numpy as np

def rewardFunction():
    pass



def randomVariables():
    params = {
        'Kp_id': {
            'Gain': randomRange(0, 10)
        }
    }
    return params

def randomRange(max, min): ## matrix 생성
    return (np.random.rand(1,1)*2-1) * (max-min)/2 + (max+min)/2 # 

def rewardCalc(obs):

    err_id = obs[0] - obs[2]  # i_LPF - i_sens
    err_iq = obs[1] - obs[3]  # i_LPF - i_sens
    wrm = obs[6]
    angle = obs[7]
    err_trq = obs[9]-obs[8]
    
    norm_err_id = err_id/358.2837
    norm_err_iq = err_iq/358.2837
    norm_err_trq = err_trq/200

    state = [norm_err_id, norm_err_iq, wrm, angle, norm_err_trq]   # should we normalize the state? 
    rwd = -abs(norm_err_id) - abs(norm_err_iq)    # with normarlization and Sum
    icts = np.sqrt(obs[2]**2 + obs[3]**2)
    vcts = np.sqrt(obs[4]**2 + obs[5]**2)       # Later we have to calculate the Vdqout_summed_FFD

    #print(rwd)

    if icts > 358.2837:
        rwd = rwd - 10                     # Constraints가 넘으면 강하게 reward를 주는 것이 맞을까?
        #print("Current_Constraints_reward_Occurs")

    if vcts > 325/np.sqrt(3):
        rwd = rwd - 10


    if abs(rwd) > 10000000:      # abs라서 멈췄다 일단은 최악의 gain에서도 150은 넘지 않음. 학습시키는 데에 집중해봐자.
        isDone = True
        print("if rwd>150 : ", rwd)
    else:
        isDone = False

    #print("rwd : ", rwd)

    state = np.array(state).reshape(1, len(state))
    #reward = np.array((5.0-dis) / 5.0 * 0.1).reshape(1, 1)     # define the reward
    reward = np.array(rwd).reshape(1,1)
    isDone = np.array(isDone).reshape(1,1)

    return state, reward, isDone

