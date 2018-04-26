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


class MagicSquae(gym.Env):

    def __init__(self, DIM_=3, POW_=1):
        self.__version__ = "0.1.0"
        print("MagicSqaure - Version {}".format(self.__version__))

        # General variables defining the environment
        self.DIM = DIM_
        self.POW = POW_
        self.action_space = spaces.Discrete(2*self.DIM*self.DIM)
        self.state = None
        self.is_square_solved=False
        self.curr_step = -1 


        # Simulation related variables.
        self.seed()
        self.reset()
 
       # Just need to initialize the relevant attributes
        self._configure()


    def step(self, action):
        """
        """
        if self.is_square_solved:
            raise RuntimeError("Episode is done")

        self.curr_step += 1
        self._take_action(action)
        reward = self._get_reward()
        info_ = {}
        return self.state, reward, self.is_square_solved, info_

    def _take_action(self, action):
        return 

    def _get_reward(self):
        return 0.0

    def reset(self):
        """
        Reset the state of the environment and returns an initial observation.

        Returns
        -------
        observation (object): the initial observation of the space.
        """
        return self.state

    def render(self, mode='human', close=False):
        return
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _configure(self):
        return 

class MagicSquare3x3P1(MagicSquare):
    """
    """
    def __init__(self):
        super(MagicSquare3x3P1, self).__init__(DIM_=3,POW_=1)


class MagicSquare3x3P1(MagicSquare):
    """
    """
    def __init__(self):
        super(MagicSquare3x3P1, self).__init__(DIM_=6,POW_=1)


class MagicSquare6x6P3(MagicSquare):
    """
    """
    def __init__(self):
        super(MagicSquare6x6P3, self).__init__(DIM_=6,POW_=3)


