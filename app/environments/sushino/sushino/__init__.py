from gym.envs.registration import register

register(
    id='SushiNO-v0',
    entry_point='sushino.envs:SushiNOEnv',
)

