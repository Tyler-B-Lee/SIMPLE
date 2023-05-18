from gym.envs.registration import register

register(
    id='Quarto-v0',
    entry_point='quarto.envs:QuartoEnv',
)

