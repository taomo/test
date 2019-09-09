# -*- coding: utf-8 -*-
'''
@author: taomo 
2018.6.14 morining

'''


import gym
from gym import spaces
import numpy as np
from gym.utils import seeding
import time
import math
from scipy.integrate import odeint, solve_bvp, solve_ivp


# import plotly
# import plotly.graph_objs as go
# from plotly.graph_objs import Layout,Scatter

class VibrationEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }

    #ms = 391, mt = 50.7, ks = 60000, kt = 362000, cs = 400, ct = 400  1e-6
    def __init__(self, steps = 1e-3, ms = 60.216, mt = 120.12, ks = 1764000, kt = 2940000, cs = 0.1, ct = 0.1):
        self.steps = steps
        self.ms = ms
        self.mt = mt
        self.ks = ks
        self.kt = kt
        self.cs = cs
        self.ct = ct
        self.global_target = 40.0
        self.min_action = -20
        self.max_action = 20

        self.max_xs = 1e-1
        self.min_xs = -1e-1
        self.max_dot_xs = 1
        self.min_dot_xs = -1

        self.max_xt = 1e-1
        self.min_xt = -1e-1
        self.max_dot_xt = 1
        self.min_dot_xt = -1



        self.low_state = np.array([self.min_xs, self.min_xt, self.min_dot_xs, self.min_dot_xt])
        self.high_state = np.array([self.max_xs, self.max_xt, self.max_dot_xs, self.max_dot_xt])

        self.action_space = spaces.Box(low=self.min_action, high=self.max_action, shape=(1,))
        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state)

        self.action_space.n = 1

        self.seed()
        self.reset()

        self.viewer = None
        self.state = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    '''
    def odeVibration(self, state, t, args):
        x,y,p,q = state
        f,u = args
        fz = 50
        f = 1e1 * math.sin(2 * math.pi * fz * t)
        # f = 1e4 * math.sin(2 * t)
        # u = 1e6 * math.sin(2 * t)
        dx = p
        dy = q
        dp = (-self.ks * (x - y ) - self.cs * (p - q ) - f) / self.ms
        dq = (self.ks * (x - y ) + self.cs * (p - q ) - self.kt * y - self.ct * q - u) / self.mt
        # print('dq',dq)
        return np.array([dx,dy,dp,dq])
    
    def step(self, action):
       
        u = min(max(action, self.min_action), self.max_action) * 1
        
        f = 0
        h = self.steps        
        self.b = self.a + h
        t = np.arange(self.a, self.b, h * 1e-2)

        P = odeint(self.odeVibration, self.state, t, args = ([f,u],))       
      
        # temp = P[len(P)-1:len(P), :]
        # temp = temp.reshape(1,4)
        # temp = temp.flatten()
        # self.state = temp

        self.state = P[len(P)-1:len(P), :].reshape(1,4).flatten()

        delta = self.odeVibration(self.state, self.b - h * 1e-2, [f,u])

        self.a = self.b
        self.counts += 1        

        # print('delta',delta[3])      

        V = 20 * math.log(math.fabs(delta[3]), 10) + 120 
        done = bool(V <= 90)  #2e-3
        # print(delta[3])
        reward = 0
        if done:
            reward = 10000            
        reward -= V

        # print(math.fabs(delta[3]),[V],done,u,reward,self.state,self.counts)
        # print(math.fabs(delta[3]),done,u.numpy(),reward,self.state,self.counts)
        # print(type(u),u.numpy())
        
        # if done:
        #     time.sleep(1)
        #     pass 
        
        return self.state, reward, done, {"input": u, "delta": delta}
    '''
    def odeVibration(self, t, y, args):
        x,y,p,q = y
        # print(args)
        f,u = args
        fz = 50
        # f = 1e1 * math.sin(2 * math.pi * fz * t) + 1e1 * math.sin(2 * math.pi * 1e3 * t)
        f = 1e1 * math.sin(2 * math.pi * fz * t) 
        # f = 1e4 * math.sin(2 * t)
        # u = 1e1 * math.sin(2 * t)
        dx = p
        dy = q
        dp = (-self.ks * (x - y ) - self.cs * (p - q ) - f) / self.ms
        dq = (self.ks * (x - y ) + self.cs * (p - q ) - self.kt * y - self.ct * q - u) / self.mt
        # print('dq',dq)
        return np.array([dx,dy,dp,dq])

    
    def step(self, action):
       
        u = min(max(action, self.min_action), self.max_action) * 1
        
        f = 0
        h = self.steps        
        self.b = self.a + h
        # t = np.arange(self.a, self.b, h * 1e-2)

        # P = odeint(self.odeVibration, self.state, t, args = ([f,u],))  
        t_span = (self.a, self.b)
        t_eval = np.arange(self.a, self.b, h * 1e-2)

        # solve_ivp(fun=lambda t, y: fun(t, y, *args), ...)
        args = [f,u]
        odeVibration = lambda t, y: self.odeVibration(t, y, args)
        sol = solve_ivp(odeVibration, t_span=t_span, y0 = self.state, method = 'RK45', t_eval = [self.b])
        # sol = solve_ivp(odeVibration, t_span=t_span, y0 = self.state, method = 'RK45', t_eval = t_eval)


        # print(sol.y)
        # print(sol.y.reshape(1,4).flatten())

        self.state = sol.y.reshape(1,4).flatten()
        # print('state',self.state)
        delta = self.odeVibration(self.b, self.state, args)

        self.a = self.b
        self.counts += 1        
        # print('delta',delta) 
        # print('delta',delta[3])      
        epsilon = 1e-10
        V = 20 * math.log(math.fabs(delta[3] + epsilon), 10) + 120 
        done = bool(V <= 130)  #2e-3
        # print(delta[3])
        reward = 0
        if done:
            reward = 1e4
        reward -= V


        costs = delta[3]**2  + 0.001*(u**2)    
       
        reward = -costs

        if done:
            print(done,'u',u,'V',V,reward,self.state,self.counts)
            # time.sleep(1)
        pass 

        done = False




        # print(math.fabs(delta[3]),[V],done,u,reward,self.state,self.counts)

        # print(math.fabs(delta[3]),done,u.numpy(),reward,self.state,self.counts)
        # print(type(u),u.numpy())
        
        # if done:
        #     time.sleep(1)
        #     pass 
        
        # return self.state, reward, done, {"input": u, "delta": delta}
        return self.state, reward, done, {"input": u, "delta": delta}



    def reset(self):
        #self.state = np.array([self.np_random.uniform(low=-0.6, high=-0.4), 0, 0, 0])   
        # self.state = np.array(self.np_random.uniform(low=-2e-4, high=2e-4, size=4))    
        self.state = np.array(self.np_random.uniform(low=-2e-4, high=2e-4, size=4))  
        # self.state = np.array([2e-4, 2e-4, 2e-4, 2e-4])     
        self.counts = 0
        self.a = 0
        return np.array(self.state)
    
    



    def render(self, mode='human'):
        
        screen_width = 600
        screen_height = 400

        ms_width = 200
        ms_height = 50

        mt_width = 300
        mt_height = 50

        world_width_s = (self.max_xs - self.min_xs)
        scale_s = screen_height/world_width_s

        world_width_t = (self.max_xt - self.min_xt)
        scale_t = screen_height/world_width_t



        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            #l,r,t,b = -s_width/2, s_width/2, s_height/2, -s_height/2
            #l,r,t,b = 200, 400, 100+s_height/2, 100-s_height/2
            l,r,t,b = (screen_width - ms_width)/2, (screen_width + ms_width)/2, 300 + ms_height/2, 300 - ms_height/2
            m_s = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            m_s.set_color(.8,.6,.4)
            self.m_s_trans = rendering.Transform()
            m_s.add_attr(self.m_s_trans)
            self.viewer.add_geom(m_s)

            l,r,t,b = (screen_width - mt_width)/2, (screen_width + mt_width)/2, 200+mt_height/2, 200-mt_height/2
            m_t = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            m_t.set_color(.8,.6,.4)
            self.m_t_trans = rendering.Transform()
            m_t.add_attr(self.m_t_trans)
            self.viewer.add_geom(m_t)

            track = rendering.Line((0,100), (screen_width,100))
            track.set_color(0,0,0)
            self.viewer.add_geom(track)


        if self.state is None: return None
        
        state = self.state
        m_s_x = state[0]*scale_s  
        m_s_x = min(max(m_s_x , -100+ms_height/2+mt_height/2), 100-ms_height/2)

        self.m_s_trans.set_translation(0, m_s_x)

        m_t_x = state[1]*scale_t  
        m_t_x = min(max(m_t_x , -100+mt_height/2), 100-ms_height/2-mt_height/2)
        
        self.m_t_trans.set_translation(0, m_t_x)


        return self.viewer.render(return_rgb_array = mode=='rgb_array')
  

    def close(self):
        if self.viewer: self.viewer.close()        



if __name__ == '__main__':
 
    import matplotlib.pyplot as plt
    import pandas as pd

    count = 0
    episodes =[]
    eval_rewards =[]
    eval_done = []
    eval_states = []
    eval_input = []
    eval_delta = []

    env_id="VibrationEnv-v0"
    env = gym.make(env_id)   #创造环境
    observation = env.reset()       #初始化环境，observation为环境状态

    
    # input()

    for t in range(int(1 / env.steps)):
        action = env.action_space.sample()  #随机采样动作
        observation, reward, done, info = env.step(0)  #与环境交互，获得下一步的时刻
        # if done:             
            # break
            # pass
        env.render()         #绘制场景
        
        # count+=1
        # time.sleep(0.001)      #每次等待0.2s
        # print(info['input'],env.state, env.counts)
        print(env.counts)           


        episodes.append(env.counts)
        eval_states.append(observation)
        eval_rewards.append(reward)
        eval_done.append(done)        
        
        eval_input.append(info['input'])
        eval_delta.append(info['delta'])
    
        
    

    episodes = np.array(episodes)
    eval_rewards = np.array(eval_rewards)
    eval_states = np.array(eval_states)
    eval_done = np.array(eval_done)
    eval_input = np.array(eval_input)
    eval_delta = np.array(eval_delta)
    
    # dataframe = pd.DataFrame({'time':episodes,'VibrationAcc':eval_delta[:,3]})
    # dataframe.to_csv("taomo10_1k.csv",index=False,sep=',')

    # import pandas as pd
    # data = pd.read_csv('taomo10.csv')
    # print(data)


    fig = plt.figure("VibrationEnv-states")
    plt.plot(episodes, eval_states[:,:2])
    plt.title("%s"%env_id)
    plt.xlabel("Episode")
    plt.ylabel("eval_states")
    plt.legend(["x","y","p","q"])
    plt.show()

    fig = plt.figure("VibrationEnv-u")
    plt.plot(episodes, eval_input)
    plt.title("%s"%env_id)
    plt.xlabel("Episode")
    plt.ylabel("eval_input")
    plt.legend(["u"])
    plt.show()


    fig = plt.figure("VibrationEnv-delta")
    plt.plot(episodes, eval_delta[:,2:])
    plt.title("%s"%env_id)
    plt.xlabel("Episode")
    plt.ylabel("eval_delta")
    plt.legend(["dp","dq"])
    plt.show()

    fig = plt.figure("VibrationEnv-delta")
    plt.plot(episodes, eval_states[:,2:])
    plt.title("%s"%env_id)
    plt.xlabel("Episode")
    plt.ylabel("eval_delta")
    plt.legend(["dp","dq"])
    plt.show()

    env.close()
    

    
    
    
