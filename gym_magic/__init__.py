import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='MagicSquare3x3-v0',
    entry_point='gym_magic.envs:MagicSquare3x3',
    timestep_limit=1000,
)
