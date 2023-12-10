from gym.envs.registration import register

register(
    id='TicTacToeSolo-v0',
    entry_point='tictactoesolo.envs.tictactoesolo:TicTacToeSoloEnv',
)
