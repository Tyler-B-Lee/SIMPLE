from gym.envs.registration import register

register(
    id='ProxyEnv-v0',
    entry_point='proxyrunner.envs.proxyrunner:ProxyEnv',
)
