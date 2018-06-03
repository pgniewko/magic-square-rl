#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
... 
"""

import random
import math
import gym
import numpy as np
from gym import spaces


class MagicSquareEnv(gym.Env):
    """
    ...
    """
    def __init__(self, DIM=3, seed=None):
        self.M = DIM*(DIM*DIM+1)/2
        self.rx_ = [0,0,0]
        self.DIM = DIM
        self.swaps = []

        for i in range(self.DIM):
            for j in range(self.DIM-1):
                first_  = i*self.DIM+j
                second_ = i*self.DIM+(j+1)
                self.swaps.append( (first_,second_) )
        
        for i in range(self.DIM-1):
            for j in range(self.DIM):
                first_  = i*self.DIM + j
                second_ = (i+1)*self.DIM + j
                self.swaps.append( (first_,second_) )
        

        self.action_space = spaces.Discrete( len(self.swaps) )
        self.observation_space = np.ones(self.DIM*self.DIM)
        self.state = None
        self.is_square_solved=False
        self.curr_step = 0 
   
        # Simulation related variables.
        self.seed(seed)
        self.reset()


    def reset(self):
        self.curr_step = 0
        self.rx_ = [0,0,0]
        self.is_square_solved=False
        self.state = np.arange( 1, self.DIM*self.DIM+1, 1, dtype=int)
        np.random.shuffle( self.state )
        return self.state


    def step(self, action):
        assert self.action_space.contains(action)
        
        self.curr_step += 1
        new_state = self._take_action(action)
        reward = self._get_reward()

        info_ = {}
        info_['steps'] = self.curr_step
        info_['nrows'] = self.rx_[0]
        info_['ncols'] = self.rx_[1]
        info_['ndiag'] = self.rx_[2]

        if self.is_square_solved:
            self._print_ms()

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
        
        r1 = np.sum( (row_sums-self.M) == 0)
        r2 = np.sum( (column_sums-self.M) == 0)
        r3 = np.sum( (diagonal_sums-self.M) == 0)

        previous = np.sum(self.rx_)
        current  = np.sum([r1,r2,r3])
        self.rx_ = [r1,r2,r3]

        if previous == current:
            reward = 0
        elif previous > current:
            reward = -1
        else:
            reward = 1

        self.is_square_solved = residues == 0
        if self.is_square_solved:
            reward = 1

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
            ms_ext[self.DIM+1][i] = row_sums[i]

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


class MagicSquare3x3(MagicSquareEnv):
    """
    """
    def __init__(self, seed_=None):
        self.__version__ = "0.1"
        print("MagicSqaure3x3 - Version {}".format(self.__version__))
        super(MagicSquare3x3, self).__init__(DIM=3, seed=seed_)


class MagicSquare5x5(MagicSquareEnv):
    """
    """
    def __init__(self, seed_=None):
        self.__version__ = "0.1"
        print("MagicSqaure5x5 - Version {}".format(self.__version__))
        super(MagicSquare5x5, self).__init__(DIM=5, seed=seed_)


class MagicSquare10x10(MagicSquareEnv):
    """
    """
    def __init__(self, seed_=None):
        self.__version__ = "0.1"
        print("MagicSqaure10x10 - Version {}".format(self.__version__))
        super(MagicSquare10x10, self).__init__(DIM=10, seed=seed_)


