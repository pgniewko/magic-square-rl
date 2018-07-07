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
3. For further improvement ideas check out these:
    * [Beat Atari with Deep Reinforcement Learning! (Part 1: DQN)](https://becominghuman.ai/lets-build-an-atari-ai-part-1-dqn-df57e8ff3b26)
    * [Beat Atari with Deep Reinforcement Learning! (Part 2: DQN improvements)](https://becominghuman.ai/beat-atari-with-deep-reinforcement-learning-part-2-dqn-improvements-d3563f665a2c)
    * [How to build your own AlphaZero AI using Python and Keras](https://medium.com/applied-data-science/how-to-build-your-own-alphazero-ai-using-python-and-keras-7f664945c188)
    * [A Deep Dive into Reinforcement Learning](https://www.toptal.com/machine-learning/deep-dive-into-reinforcement-learning)
    * [OpenAI Baselines: DQN](https://blog.openai.com/openai-baselines-dqn/)
    * [Frame Skipping and Pre-Processing for Deep Q-Networks on Atari 2600 Games](https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/)



