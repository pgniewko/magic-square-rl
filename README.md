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


### Train
```bash
python train-enigma-DQN.py
```

### Test
1) Generate figures
```bash
python play-enigma.py
```

## The Environment
Describe the Env here

## TODO
1. Add CNN-2D for the brain 
2. Constant Seed
3. setup.py correction and installation instruction
4. Finishup this file; add resources
5. Printout Magic Square
6. Printout training method
