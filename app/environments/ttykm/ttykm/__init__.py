from gym.envs.registration import register

register(
    id='ttykm-v0',
    entry_point='ttykm.envs:ttykmEnv',
)

