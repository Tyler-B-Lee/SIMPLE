from gym.envs.registration import register

register(
    id='Mancala-v0',
    entry_point='mancala.envs:MancalaEnv',
)

