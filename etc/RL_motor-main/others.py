import numpy as np

def rewardFunction():
    pass


# def randomVariables():
#     params = {
#         'vehicleModel': {
#             'X_o': randomRange(-2,2),
#             'Y_o': randomRange(-0.1,0.1),
#             'ydot_o': randomRange(-0.1,0.1),
#             'psi_o': randomRange(-0.01,0.01),
#             'r_o': randomRange(-0.01,0.01)
#         },
#         'Vx': {
#             # 'Value': randomRange(25, 35)
#             'Value': 30
#         },
#         'u': {
#             'Value': randomRange(-np.pi/12, np.pi/12)
#         }
#     }
#     return params

def randomVariables():
    params = {
        'Kp_id': {
            'Gain': randomRange(0, 10)
        }
    }
    return params

def randomRange(max, min): ## matrix 생성
    return (np.random.rand(1,1)*2-1) * (max-min)/2 + (max+min)/2 # 

#def rewardCalc(obs, ref):
def rewardCalc(obs):
    # print('RewardCalc Called')
    # print(obs)
    # print(obs[0])
    # print(ref)
    # print(ref[0])
    #print("obs : ", obs, type(obs))
    # i_LPF =np.asarray(obs[0])
    # i_sens = np.asarray(obs[1])
    #print("obs : ", obs, type(obs))
    err_id = obs[0] - obs[2]  # i_LPF - i_sens
    err_iq = obs[1] - obs[3]  # i_LPF - i_sens
    wrm = obs[6]
    angle = obs[7]
    err_trq = obs[9]-obs[8]
    
    norm_err_id = err_id/358.2837
    norm_err_iq = err_iq/358.2837
    norm_err_trq = err_trq/200

    #print("err_i", obs[0], obs[1], err_i)

    #print("err_i : ", err_i, type(err_i))

    #print("obs : ", obs[0], obs[1])
    #print("err_i : ", err_i)

    #state = [err_y, err_ydot, err_psi, err_r, err_X, err_y]
    state = [norm_err_id, norm_err_iq, wrm, angle, norm_err_trq]   # should we normalize the state? 
    #dis = np.sqrt(err_X**2 + err_Y**2)
    #rwd = -abs(err_id)/358.2837 - abs(err_iq)/358.2837    # with normarlization and Sum
    rwd = -abs(norm_err_id) - abs(norm_err_iq)    # with normarlization and Sum
    icts = np.sqrt(obs[2]**2 + obs[3]**2)
    vcts = np.sqrt(obs[4]**2 + obs[5]**2)       # Later we have to calculate the Vdqout_summed_FFD

    #print(rwd)

    if icts > 358.2837:
        rwd = rwd - 10                     # Constraints가 넘으면 강하게 reward를 주는 것이 맞을까?
        #print("Current_Constraints_reward_Occurs")

    if vcts > 325/np.sqrt(3):
        rwd = rwd - 10
        #print("Voltage_Constraints_reward_Occurs")
    #c_cons = 366.3094
    #v_cons = 325

    #print("dis :", dis)

    #print(rwd)

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

# def referenceGenerator(max_step, Ts):       # (10000, 0.01)
#     ref = np.ones([max_step , 4])           # [10000,4]

#     ref[:,0] = ref[:,0] * 0                 # 0st column = 0 
#     ref[:,1] = ref[:,1] * 0                 # 1st column = 0

#     dot_psi = np.pi/6/12                    # 0.04363323129985824

#     q = int(max_step/4)                     # 10000/4 = 2500

#     q_time = np.arange(q) * Ts              # array([0.000e+00, 1.000e-02, 2.000e-02, ..., 2.497e+01, 2.498e+01, 2.499e+01])
#     psi = q_time * dot_psi                  

#     ref[0:q,2] = 0
#     ref[q:q*2,2] = psi
#     ref[q*2:q*3,2] = psi[-1]-psi
#     ref[q*3:,2] = 0;

#     ref[0:q,3] = 0
#     ref[q:q*2,3] = dot_psi
#     ref[q*2:q*3,3] = -dot_psi
#     ref[q*3:,3] = 0;

#     return ref

