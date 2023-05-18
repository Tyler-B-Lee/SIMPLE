from gym.envs.registration import register

register(
    id='MinecraftCG-v0',
    entry_point='minecraftcg.envs:MinecraftCGEnv',
)

