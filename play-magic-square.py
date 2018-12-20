# OpenGym MagicSqaure-v0
# -------------------
#
# author: Pawel Gniewek, 2018
#

import time
import random
import numpy as np
import turtle
from keras import backend as K
import tensorflow as tf
from keras.models import load_model


import gym
import gym_magic

myPen = turtle.Turtle()
myPen.tracer(0)
myPen.speed(10)
myPen.hideturtle()
topLeft_x = -200
topLeft_y = 250

def text(message,x,y,size, correct):
    if correct:
        myPen.color("#000000")
    else:
        myPen.color('red')

    FONT = ('Arial', size, 'normal')
    myPen.penup()
    myPen.goto(x, y)
    myPen.write(message, align="left", font=FONT)

#A procedure to draw the grid on screen using Python Turtle
def draw_grid(ms_):
    ms = ms_.reshape( (3,3) )
    row_sums =  np.sum(ms,axis=1)
    column_sums = np.sum(ms,axis=0)
    diagonal_sums = np.array( [np.trace(ms), np.trace(np.flip(ms,1)) ] )
    sums_ = np.append(np.append( row_sums, column_sums), diagonal_sums  )
    
    grid = np.zeros( (5,4) )

    for i in range(3):
        for j in range(3):
            grid[i+1][j] = ms[i][j]

    grid[1][3] = row_sums[0]
    grid[2][3] = row_sums[1]
    grid[3][3] = row_sums[2]
    grid[4][0] = column_sums[0]
    grid[4][1] = column_sums[1]
    grid[4][2] = column_sums[2]
    grid[4][3] = diagonal_sums[0] 
    grid[0][3] = diagonal_sums[1]

    intDim = 100
    myPen.color("#000000")
    
    for row in range(0, 6):
        if row == 1 or row == 4 or row == 5:
            myPen.penup()
            myPen.goto(topLeft_x, topLeft_y - row * intDim)
            myPen.pendown()
            myPen.goto(topLeft_x + 4 * intDim, topLeft_y - row*intDim)
            myPen.penup()
            myPen.goto(topLeft_x, topLeft_y - row * intDim + 1)
            myPen.pendown()
            myPen.goto(topLeft_x + 4 * intDim, topLeft_y - row * intDim+1)
        
        elif row == 0:
            myPen.penup()
            myPen.goto(topLeft_x + 3 * intDim, topLeft_y - row * intDim)
            myPen.pendown()
            myPen.goto(topLeft_x + 4 * intDim, topLeft_y - row*intDim)
            myPen.penup()
            myPen.goto(topLeft_x + 3 * intDim, topLeft_y - row * intDim + 1)
            myPen.pendown()
            myPen.goto(topLeft_x + 4 * intDim, topLeft_y - row * intDim+1)
        else:
            myPen.penup()
            myPen.goto(topLeft_x, topLeft_y - row * intDim)
            myPen.pendown()
            myPen.goto(topLeft_x + 4 * intDim, topLeft_y - row*intDim)


    for col in range(0,5):
        if col == 3 or col == 4:
            myPen.penup()
            myPen.goto(topLeft_x + col * intDim, topLeft_y)
            myPen.pendown()
            myPen.goto(topLeft_x + col * intDim, topLeft_y - 5 * intDim)
            myPen.penup()
            myPen.goto(topLeft_x + col * intDim + 1, topLeft_y)
            myPen.pendown()
            myPen.goto(topLeft_x + col * intDim + 1, topLeft_y - 5 * intDim)
        else:
            myPen.penup()
            myPen.goto(topLeft_x + col  * intDim, topLeft_y - 1 * intDim)
            myPen.pendown()
            myPen.goto(topLeft_x + col  * intDim, topLeft_y - 5 * intDim)
            


    for row in range (0,5):
        for col in range (0,4):
            if grid[row][col] != 0:
                if row == 4 or col == 3:
                    text( int(grid[row][col]), topLeft_x + col * intDim + 25, topLeft_y - row * intDim - intDim + 25, 40, int(grid[row][col])==15)
                else:
                    text( int(grid[row][col]), topLeft_x + col * intDim + 25, topLeft_y - row * intDim - intDim + 25, 40, True)



if __name__ == "__main__":

    PROBLEM = 'MagicSquare3x3-v0'
    DIFFICULTY_LEVEL = 1
    model_dir = './model'
    model_file = model_dir + "/" + PROBLEM + "-dqn.dl-%d.h5" %(DIFFICULTY_LEVEL)
    print model_file
    model = load_model(model_file)


    env = gym.make(PROBLEM)
    env = env.unwrapped
    env.configure(dl=DIFFICULTY_LEVEL)
    state_cnt = env.observation_space.shape[0]
    action_cnt = env.action_space.n

    epsilon = 0.1
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
            
            myPen.clear()
            draw_grid(s_)
            myPen.getscreen().update()
            
            if done:
                game_over = True

            print done, info 
            time.sleep(1)
            
    time.sleep(5)
