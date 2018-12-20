#!/usr/bin/env python

"""
... 
"""

import random
import math
import gym
import numpy as np
from gym import spaces
import ms as ms_lib


class MagicSquare3x3(gym.Env):
    """
    ...
    """
    def __init__(self, seed=None):
#        self.__version__ = "0.3"
#        print("MagicSqaure3x3 - Version {}".format(self.__version__))
        self.DIM = 3
        self.M = self.DIM * (self.DIM * self.DIM+ 1 ) / 2
        self.swaps = ms_lib.all_moves

        self.action_space = spaces.Discrete( len(self.swaps) )
        self.observation_space = np.ones(self.DIM * self.DIM)
        self.state = None
        self.is_square_solved=False
        self.curr_step = 0 
        
        ## EXTRAS
        self.scramble = 1
        self.success_counter = 0
        self.experience_factor = 1000
   
        # Simulation related variables.
        self.seed(seed)
        self.reset()
        

    def reset(self):
        self.curr_step = 0
        self.is_square_solved=False
        self.state = ms_lib.random_ms( self.scramble )
       
        return self.state


    def step(self, action):
        assert self.action_space.contains(action)
        
        self.curr_step += 1
        new_state = self._take_action(action)
        reward = self._get_reward()

        info_ = {}
        info_['steps'] = self.curr_step

        if self.is_square_solved:
            self._print_ms()
            self.success_counter += 1
            if self.success_counter % self.experience_factor == 0:
                self.scramble += 1
                print "Scrambling level increased to ", self.scramble

        return new_state, reward, self.is_square_solved, info_


    def _take_action(self, action):
        f_, s_ = self.swaps[ action ]
        f_val = self.state[f_]
        self.state[f_] = self.state[s_]
        self.state[s_] = f_val
        return self.state.copy()


    def _get_reward(self):
        ms_ = self.state.reshape( (self.DIM,self.DIM) )
        row_sums =  np.sum(ms_,axis=1)
        column_sums = np.sum(ms_,axis=0)
        diagonal_sums = np.array( [np.trace(ms_), np.trace(np.flip(ms_,1)) ] )
      
        sums_ = np.append(np.append( row_sums, column_sums), diagonal_sums  )
        arr_ = sums_ - self.M
        residues = np.sum( arr_**2 )
        
        self.is_square_solved = residues == 0
        if self.is_square_solved:
            reward = 1
        else:
            reward = 0
        
        return reward 


    def _print_ms(self):
        from tabulate import tabulate
       
        ms_ = self.state.reshape( (self.DIM,self.DIM) )
        row_sums =  np.sum(ms_,axis=1)
        column_sums = np.sum(ms_,axis=0)
        diagonal_sums = np.array( [np.trace(ms_), np.trace(np.flip(ms_,1)) ] )
        sums_ = np.append(np.append( row_sums, column_sums), diagonal_sums  )
        
        ms_ext = []
        for i in range(self.DIM+2):
            row = []
            for j in range(self.DIM+1):
                row.append( '' )

            ms_ext.append(row)
        
        for i in range(self.DIM):
            for j in range(self.DIM):
                ms_ext[i+1][j] = ms_[i][j]
        
        for i in range(self.DIM):
            ms_ext[i+1][self.DIM] = row_sums[i]
        
        for i in range(self.DIM):
            ms_ext[self.DIM+1][i] = column_sums[i]

        ms_ext[0][self.DIM] = diagonal_sums[1]
        ms_ext[self.DIM+1][self.DIM] = diagonal_sums[0]
        table = tabulate(ms_ext, tablefmt="fancy_grid")
        
        print(table)
        
        return


    def seed(self, seed_):
        random.seed(seed_)
        return [seed_]
    
    
    def render(self, mode='human', close=False):
        return

    
    def close(self):
        return

