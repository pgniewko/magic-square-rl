# OpenGym MagicSqaure-v0
# -------------------
#
# author: Pawel Gniewek, 2018

from keras import backend as K
import tensorflow as tf
from keras.models import load_model

if __name__ == "__main__":

    PROBLEM = 'MagicSquare3x3P1-v0'
    model_dir = './model'
    model_file = model_dir + "/" + PROBLEM + "-dqn.h5"
    model = load_model(model_file)

