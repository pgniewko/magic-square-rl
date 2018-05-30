#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
... 
"""

# core modules
import random
import math

# 3rd party modules
import gym
import numpy as np
from gym import spaces
from gym.utils import seeding


class MagicSquareEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, DIM_=3, seed_=None):
        # SET SEED FIXED
        seed_=123
  
        self. M = DIM_*(DIM_*DIM_+1)/2
        # General variables defining the environment
        self.DIM = DIM_
        self.swaps = []

        extra_step = 0
        for i in range(self.DIM):
            for j in range(self.DIM):
                first_ = i*self.DIM+j
                for m in range(self.DIM):
                    for n in range(self.DIM):
                        second_ = m*self.DIM+n
                        if first_ > second_:
                            self.swaps.append( (first_,second_) )
                            extra_step += 1
                    
        self.action_space = spaces.Discrete( extra_step )
                
        self.observation_space = np.ones(self.DIM*self.DIM)
        self.state = None
        self.is_square_solved=False
        self.curr_step = -1 
   
        # Simulation related variables.
        self.seed(seed_)
        self.reset()
 

    def reset(self):
        self.state = np.arange( 1, self.DIM*self.DIM+1, 1, dtype=int)
        np.random.shuffle( self.state )
        return self.state


    def step(self, action):
        assert self.action_space.contains(action)
        
        if self.is_square_solved:
            self._print_ms()
            raise RuntimeError("Episode is done")

        self.curr_step += 1
        new_state = self._take_action(action)
        reward = self._get_reward()
        info_ = {}
        self.is_square_solved = reward == 0

        if self.is_square_solved:
            self._print_ms()
            raise RuntimeError("Episode is done")


        return new_state, reward, self.is_square_solved, info_


    def _take_action(self, action):
        f_, s_ = self.swaps[ action ]
        f_val = self.state[f_]
        self.state[f_] = self.state[s_]
        self.state[s_] = f_val
            
        return self.state


    def _get_reward(self):
#        if 0 in self.state:
#            return 0.0
#        u_, c_ = np.unique(self.state,return_counts=True)
#        if c_.max() > 1:
#            return 0.0

        ms_ = self.state.reshape( (self.DIM,self.DIM) )
        row_sums =  np.sum(ms_,axis=1)
        column_sums = np.sum(ms_,axis=0)
        diagonal_sums = np.array( [np.trace(ms_), np.trace(np.flip(ms_,1)) ] )
      
        sums_ = np.append(np.append( row_sums, column_sums), diagonal_sums  )
        reward = self._calc_reward_score(sums_)
        return reward


    def _get_space_size(self):
        return self.DIM*self.DIM


    def _calc_reward_score(self, sums_):
        arr_ = sums_ - self.M
        reward = np.sum( arr_**2 )
        reward *= -1 

        return reward


    def _print_ms(self):
        print self.state
        return


    def seed(self, seed=None):
#        np.random.seed(123)
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    
    def render(self, mode='human', close=False):
        return

    
    def close(self):
        return


class MagicSquare3x3(MagicSquareEnv):
    """
    """
    def __init__(self,seed_=None):
        self.__version__ = "0.1"
        print("MagicSqaure3x3 - Version {}".format(self.__version__))
        super(MagicSquare3x3, self).__init__(DIM_=3,seed_=seed_)


class MagicSquare5x5(MagicSquareEnv):
    """
    """
    def __init__(self,seed_=None):
        self.__version__ = "0.1"
        print("MagicSqaure5x5 - Version {}".format(self.__version__))
        super(MagicSquare5x5, self).__init__(DIM_=5,seed_=seed_)


class MagicSquare10x10(MagicSquareEnv):
    """
    """
    def __init__(self,seed_=None):
        self.__version__ = "0.1"
        print("MagicSqaure10x10 - Version {}".format(self.__version__))
        super(MagicSquare10x10, self).__init__(DIM_=10,seed_=seed_)


