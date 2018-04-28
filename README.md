This repository contains a PIP package which is an OpenAI environment for
simulating an enironment in which bananas get sold.


## Installation

Install the [OpenAI gym](https://gym.openai.com/docs/).

Then install this package via

```
pip install -e .
```

## Usage

```
import gym
import gym_magic

env = gym.make('MagicSquare3x3P1-v0')
```

See https://github.com/matthiasplappert/keras-rl/tree/master/examples for some
examples.


## The Environment
Describe the Env here

## TODO
1. Constant Seed
2. setup.py correction and installation instruction
3. Finishup this file; add resources
4. Be able to choose what Brain model we want to use
5. Printout Magic Square
6. Keras on GPU, i.e. if there is GPU run on it
7. Printout training method
8. Implement game-play
9. Implement new actions: swaps
