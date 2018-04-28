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

    def __init__(self, DIM_=3, POW_=1, seed=None):
        self.BASE_=2
   
        # General variables defining the environment
        self.DIM = DIM_
        self.POW = POW_
        self.action_space = spaces.Discrete(2*self.DIM*self.DIM)
        self.observation_space = np.ones(self.DIM*self.DIM)
        self.state = None
        self.is_square_solved=False
        self.curr_step = -1 
   
        
        self.reward_range = (0, self.BASE_**(2*self.DIM+2))
        
        # Simulation related variables.
        self.seed(seed)
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
        self.is_square_solved = reward == self.reward_range[1]

        if self.is_square_solved:
            self._print_ms()
            raise RuntimeError("Episode is done")


        return new_state, reward, self.is_square_solved, info_


    def _take_action(self, action):
        if action < self.DIM*self.DIM:
            self.state[action] += 1
        else:
            if self.state[action - self.DIM*self.DIM] > 1:
                self.state[action - self.DIM*self.DIM] -= 1
                
            
        return self.state


    def _get_reward(self):
        if 0 in self.state:
            return 0.0
        u_, c_ = np.unique(self.state,return_counts=True)
        if c_.max() > 1:
            return 0.0

        pow_numbers = np.power(self.state,self.POW)
        ms_ = pow_numbers.reshape( (self.DIM,self.DIM) )
        row_sums =  np.sum(ms_,axis=1)
        column_sums = np.sum(ms_,axis=0)
        diagonal_sums = np.array( [np.trace(ms_), np.trace(np.flip(ms_,1)) ] )
      
        sums_ = np.append(np.append( row_sums, column_sums), diagonal_sums  )
        reward = self._calc_reward_score(sums_)
        
        return reward


    def _get_space_size(self):
        return self.DIM*self.DIM


    def _calc_reward_score(self, sums_):
        uniqs_, counts_ = np.unique(sums_,return_counts=True)
        reward = self.BASE_**( counts_.max() )
        return reward


    def _print_ms(self):
        return


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    
    def render(self, mode='human', close=False):
        return

    
    def close(self):
        return


class MagicSquare3x3P1(MagicSquareEnv):
    """
    """
    def __init__(self):
        self.__version__ = "0.1"
        print("MagicSqaure3x3P1 - Version {}".format(self.__version__))
        super(MagicSquare3x3P1, self).__init__(DIM_=3,POW_=1)


class MagicSquare6x6P1(MagicSquareEnv):
    """
    """
    def __init__(self):
        self.__version__ = "0.1"
        print("MagicSqaure6x6P1 - Version {}".format(self.__version__))
        super(MagicSquare6x6P1, self).__init__(DIM_=6,POW_=1)


class MagicSquare6x6P3(MagicSquareEnv):
    """
    """
    def __init__(self):
        self.__version__ = "0.1"
        print("MagicSqaure6x6P3 - Version {}".format(self.__version__))
        super(MagicSquare6x6P3, self).__init__(DIM_=6,POW_=3)


