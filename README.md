This repository contains a PIP package which is an OpenAI environment for
simulating an environment in which a magic-square is solved.


## Installation

Install the [OpenAI gym](https://gym.openai.com/docs/).

Then install this package via

```
pip install -e .
```


## REQUIRED LIBRARIES ##
* numpy
* keras
* tensorflow
* gym: `pip install gym` 
* tabulate: `pip install tabulate`


## Usage

```python
import gym
import gym_magic

env = gym.make('MagicSquare3x3-v0')
```


### Train
```bash
python train-magic-square-DQN.py
```

### Test
1) Play the game
```bash
python play-magic-square.py
```

## The Environment
...

## CREDITS
Part of the code for Deep-Q-Learning is based on this [blog-post](https://github.com/jaara/AI-blog/blob/master/CartPole-basic.py).  

## REFERENCES
1. "Human-level control through deep reinforcement learning", Nature 518 (7540), 529-533
2. "Playing Atari with Deep Reinforcement Learning", preprint, arXiv:1312.5602
3. Deep Reinforcement Learning [Blog](http://karpathy.github.io/2016/05/31/rl/)

## TODO
1. Tinker Brain's architecture   
2. Implement graphics window for `play-magic-sqaure.py`   
3. Explore (and clip) the reward function   
