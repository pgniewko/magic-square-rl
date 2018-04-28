import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='MagicSquare3x3P1-v0',
    entry_point='gym_magic.envs:MagicSquare3x3P1',
    timestep_limit=10000,
)


register(
    id='MagicSquare6x6P1-v0',
    entry_point='gym_magic.envs:MagicSquare6x6P1',
    timestep_limit=10000,
)


register(
    id='MagicSquare6x6P3-v0',
    entry_point='gym_magic.envs:MagicSquare6x6P3',
    timestep_limit=10000,
)
