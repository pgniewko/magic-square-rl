# OpenGym MagicSqaure-v0
# -------------------
#
# author: Pawel Gniewek, 2018
#

import random
import numpy as np
from keras import backend as K
import tensorflow as tf
from keras.models import load_model

import gym
import gym_magic

if __name__ == "__main__":

    PROBLEM = 'MagicSquare3x3-v0'
    model_dir = './model'
    model_file = model_dir + "/" + PROBLEM + "-dqn.h5"
    model = load_model(model_file)

    env = gym.make(PROBLEM)
    state_cnt = env.observation_space.shape[0]
    action_cnt = env.action_space.n
    dim = int(state_cnt**0.5)

    epsilon = 0.05
    # Play only one game
    for episode in range(1):
        c = 0
        s = env.reset()
        game_over = False

        while not game_over:
            if random.random() < epsilon:
                a = random.randint(0, action_cnt-1)
            else:
                a = np.argmax( model.predict( s.reshape(1, state_cnt) ).flatten() )
            
            s_, r, done, info = env.step(a)
            s = s_
            
            print done, info
            if done:
                game_over = True


