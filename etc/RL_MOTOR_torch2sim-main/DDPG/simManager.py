import matlab.engine
import numpy as np
import time

class SimManager:
    def __init__(self, modelName) -> None:
        self.modelName = modelName

    def __get_obs(self, obsInfo):
        #print("Observation!")
        tout = self.eng.workspace['tout']
        #print("present_tout : ", tout, type(tout))
        if isinstance(tout, float):
        # #     #print("type tout and tout(float) : ",type(tout), tout)
            tout1 = tout
        else:       # matlab double 일때 float 일때
            tout1 = float(tout[-1][0])
            #tout = [tout][-1]
            #print("type tout and tout(list) : ",type(tout1), tout1)
        #print("tout1 :",tout1)
        #tout = [self.eng.workspace['tout_sample']]

        #if tout1 % 0.0001 == 0:
        #    tout1 = []
        #    tout1.append(tout) 
        obs = []        # obs 초기화
        for name in obsInfo:
            obs.append(self.eng.workspace[name])
        #print("obs(__get_obs) : ", obs, type(obs))
  
                #obs.append(np.asarray(self.eng.workspace[name]))
            
            #print("obs_list : ", obs)
        
        return  obs, tout1 #id_LPF
    
    def __setParameter(self, block, name, value):
        self.eng.set_param(
            '{}/PMSM controller/Inner loop control/{}'.format(self.modelName, block),
            name, str(value),
            nargout=0
        )

    def connectMatlab(self):
        print('[ MNG] Connecting Matlab')
        self.eng = matlab.engine.start_matlab()
        print('[ MNG] Connected Successfully')

        print('[ MNG] Loading Simulink Model')
        self.eng.eval(
            "model = '{}';".format(self.modelName), nargout=0
        )
        self.eng.eval(
            "load_system(model)", nargout=0
        )
        print('[ MNG] Loaded Successfully')

    def reset(self, obsInfo, initial_parameters):     # Kp_id, initial_parameters -> randomVariables
        # print('[ MNG] Reset Env')
        print("reset!!!!!!!!")
        self.eng.set_param(self.modelName, 'SimulationCommand', 'stop', nargout=0)

        for block in initial_parameters:        # initial_parameters -> randomVariables -> 'Gain': randomRange(0, 10)
            for name in initial_parameters[block]:
                value = initial_parameters[block][name]
                self.__setParameter(block, name, value)

        self.eng.set_param(self.modelName,                  # update the state value
                           'SimulationCommand', 'start', 'SimulationCommand', 'pause', 
                           nargout=0)               ### 0.0001 state
        #tout = self.eng.workspace['tout']


        
        return self.__get_obs(obsInfo)#, tout

    def step(self, obsInfo, abj_parameters, n):
        #print("step!!!!!!!!!!!")
        #print("abj_parameters : ", abj_parameters)
        for block in abj_parameters:
            for name in abj_parameters[block]:
                value = abj_parameters[block][name]
                self.__setParameter(block, name, value)   # define the action
        
        time = 0.99 # initialize 의미없음
        m = 0.0001*n        # period for each step 
        #while True:
        for i in range(20):       # 0.0001 / 0.000005 = 20
            #print("m : ", m)
            self.eng.set_param(self.modelName,
                           'SimulationCommand', 'continue', 'SimulationCommand', 'pause', 
                           nargout=0)

            #tout = self.eng.workspace['tout']
            #print("present_tout : {:.5f}".format(tout), type(tout))
            #print("present_tout : ",tout, type(tout))
            #if isinstance(tout, float):
                #print("type tout and tout(float) : ",type(tout), tout)
                #print("float#####", tout)
            #    tout = tout
                #print("first_step")

            #else:       # matlab double 일때 float 일때
                #print("org_data : ", tout)
            #    time = float(tout[-1][0])       # 시작 0.0001
                #print("float(tout[-1][0]) : ", time)
                #tout1 = [tout][-1]
                #print("tout1 : ", tout1, type(tout1))
                #print("type tout and tout(list) : ",type(tout1), tout1)
                #print("first_step")
            
            #if round(round(time,10) % 0.0001, 3) == 0:           # 0.0001이 되면 (fist step)에서 반환 -> 1step에서 obs, reward 계산 가능
            #    break
            
            #if time >= m :  # 0.0001
            #    break


            #print("tout1 : ", tout)
          

        
        #self.eng.set_param(self.modelName, 'SimulationCommand', 'continue', nargout=0)
        #self.eng.set_param(self.modelName, 'SimulationCommand', 'pause', nargout=0)
        
        return self.__get_obs(obsInfo)#, time

    def disconnectMatlab(self):
        print('[ MNG] Disconnecting Env')

        self.eng.set_param(self.modelName, 'SimulationCommand', 'stop', nargout=0)
        self.eng.quit()

        print('[ MNG] Disconnected Successfully')
