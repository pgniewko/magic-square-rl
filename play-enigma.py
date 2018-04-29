# OpenGym MagicSqaure-v0
# -------------------
#
# author: Pawel Gniewek, 2018
#
#
#--- enable this to run on GPU
# import os
# os.environ['THEANO_FLAGS'] = "device=gpu,floatX=float32"


import numpy as np

from keras import backend as K
import tensorflow as tf
from keras.models import load_model

import gym
import gym_magic

if __name__ == "__main__":

    PROBLEM = 'MagicSquare3x3P1-v0'
    model_dir = './model'
    model_file = model_dir + "/" + PROBLEM + "-dqn.h5"
    model = load_model(model_file)

    env = gym.make(PROBLEM)
    state_cnt = env.observation_space.shape[0]
    dim = int(state_cnt**0.5)

    # Play only one game
    for episode in range(1):
        c = 0
        s = env.reset()
        game_over = False

        while not game_over:
            a=np.argmax( model.predict( s.reshape(1, state_cnt) ).flatten() )

            s_, r, done, info = env.step(a)

            print c, r, a, s_.reshape((dim,dim)) 
            
            s = s_
            
            c +=1

