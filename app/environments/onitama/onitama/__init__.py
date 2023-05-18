from gym.envs.registration import register

register(
    id='Onitama-v0',
    entry_point='onitama.envs:OnitamaEnv',
)

