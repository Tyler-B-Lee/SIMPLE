from gym.envs.registration import register

register(
    id='Anti-Mancala-v0',
    entry_point='antimancala.envs:AntiMancalaEnv',
)

