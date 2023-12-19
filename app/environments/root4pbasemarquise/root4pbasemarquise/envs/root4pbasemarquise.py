import gym
import numpy as np

import config

from stable_baselines import logger
from .rootGameClasses.rootMechanics import *

# Game Obs length: 1487 (down from 4474 i think)
# docker-compose exec app tensorboard --logdir ./logs
# notes:
# - os (optimizer stepsize) I think is the 'learning rate' parameter
#        - Should be decreased linearly over training to 0 or very small
#        - Some papers have it as small as 1e-6 at the end

# start
# docker-compose exec app mpirun -np 2 python3 train.py -e root4pbasemarquise -ne 50 -t 0.05 -ef 20480 -tpa 2048 -ent 0.03 -oe 10 -ob 128
# docker-compose exec app mpirun -np 2 python3 train.py -e root4pbasemarquise -ne 50 -t 0.15 -ef 20480 -tpa 2048 -ent 0.025 -ob 128 -g 0.995

MARQUISE_ID = 0
EYRIE_ID = 1
ALLIANCE_ID = 2
VAGABOND_ID = 3

MAIN_PLAYER_ID = EYRIE_ID

# marquise - 1538 obs / 548 actions
# eyrie - 1522 / 492

class rootEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, verbose = False, manual = False):
        super(rootEnv, self).__init__()
        self.name = 'root4pbasemarquise'
        self.n_players = 1
        self.manual = manual

        self.action_space = gym.spaces.Discrete(492)
        self.observation_space = gym.spaces.Box(-1, 1, (
            1522
            + self.action_space.n
            , )
        )  
        self.verbose = verbose
        self.env = RootGame(CHOSEN_MAP,STANDARD_DECK_COMP)

    @property
    def current_player(self) -> Player:
        return self.players[self.current_player_num]
    
    @property
    def opposing_player(self) -> Player:
        i = (self.current_player_num + 1) % 2
        return self.players[i]
        
    @property
    def observation(self):
        ret = np.append(self.env.get_eyrie_observation(),self.legal_actions)
        return ret

    @property
    def legal_actions(self):
        ret = np.zeros(self.action_space.n)
        ret.put(self.env.legal_actions(), 1)
        return ret

    def step(self, action):
        reward, self.done = self.env.step(action)
        self.current_player_num = (self.env.to_play() - MAIN_PLAYER_ID)

        # play the game until it comes back to the main player's turn or the game ends
        while self.current_player_num != 0 and (not self.done):
            action_chosen = random.choice(self.env.legal_actions())
            logger.debug(f"Random action {action_chosen} chosen")

            r, self.done = self.env.step(action_chosen)
            reward = [reward[i] + r[i] for i in range(4)]
            self.current_player_num = (self.env.to_play() - MAIN_PLAYER_ID)

        return self.observation, reward[MAIN_PLAYER_ID:MAIN_PLAYER_ID + 1], self.done, {}

    def reset(self):
        self.done = False
        self.env.reset()
        self.current_player_num = (self.env.to_play() - MAIN_PLAYER_ID)

        # play the game until it comes back to the main player's turn or the game ends
        while self.current_player_num != 0 and (not self.done):
            action_chosen = random.choice(self.env.legal_actions())
            logger.debug(f"Random action {action_chosen} chosen")

            r, self.done = self.env.step(action_chosen)
            self.current_player_num = (self.env.to_play() - MAIN_PLAYER_ID)

        return self.observation

    def render(self, mode='human', close=False):
        if close:
            return
        
        # self.env.render()

        if self.verbose:
            logger.debug(f'\nObservation: \n{[i if o == 1 else (i,o) for i,o in enumerate(self.observation) if o != 0]}')
        
        if not self.done:
            logger.debug(f'\nLegal actions: {[i for i,o in enumerate(self.legal_actions) if o != 0]}')


    def rules_move(self):
        raise Exception('Rules based agent is not yet implemented for Geschenkt!')

# if __name__ == "__main__":
#     env = rootEnv()
#     env.reset()
#     done = False
#     total_rewards = np.zeros(1)
#     while not done:
#         legal_actions = env.legal_actions