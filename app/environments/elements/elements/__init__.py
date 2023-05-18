from gym.envs.registration import register

register(
    id='Elements-v0',
    entry_point='elements.envs:ElementsEnv',
)

