from gym.envs.registration import register

register(
    id='MinecraftCG-v3',
    entry_point='minecraftcg.envs:MinecraftCGEnv',
)

