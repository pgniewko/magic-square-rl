This repository contains a PIP package which is an OpenAI environment for
simulating an enironment in which bananas get sold.


## Installation

Install the [OpenAI gym](https://gym.openai.com/docs/).

Then install this package via

```
pip install -e .
```


## REQUIRED LIBRARIES ##
* tabulate: `pip install tabulate`


## Usage

```python
import gym
import gym_magic

env = gym.make('MagicSquare3x3-v0')
```


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
...

## REFERENCES
1. "Human-level control through deep reinforcement learning", Nature 518 (7540), 529-533
2. "Playing Atari with Deep Reinforcement Learning", arXiv preprint arXiv:1312.5602

## TODO
1. Add CNN-2D for the brain
2. Remember last 4 moves
3. setup.py correction and installation instruction
4. https://github.com/matthiasplappert/keras-rl/tree/master/examples
5. actions shold be - switch rows, switch columns
