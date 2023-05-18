from gym.envs.registration import register

register(
    id='Shobu-v0',
    entry_point='shobu.envs:ShobuEnv',
)

