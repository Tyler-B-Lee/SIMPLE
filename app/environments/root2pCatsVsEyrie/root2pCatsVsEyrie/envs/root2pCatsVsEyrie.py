import gym
import numpy as np

import config

from stable_baselines import logger
from .rootGameClasses.classes import *
from .rootGameClasses.root2PCE import *

# docker-compose exec app python3 train.py -e root2pCatsVsEyrie -ef 25600 -ne 20 -tpa 2560 -ob 2560

class rootEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, verbose = False, manual = False):
        super(rootEnv, self).__init__()
        self.name = 'root2pCatsVsEyrie'
        self.n_players = 2
        self.manual = manual

        self.action_space = gym.spaces.Discrete(4530)
        self.observation_space = gym.spaces.Box(-1, 1, (
            2560 + 6 # fixed crafting power
            + self.action_space.n  # legal_actions
            , )
        )  
        self.verbose = verbose
        self.env = root2pCatsVsEyrie(MAP_AUTUMN,STANDARD_DECK_COMP)

    @property
    def current_player(self) -> Player:
        return self.players[self.current_player_num]
    
    @property
    def opposing_player(self) -> Player:
        i = (self.current_player_num + 1) % 2
        return self.players[i]
        
    @property
    def observation(self):
        ret = np.append(self.env.get_observation(),self.legal_actions)
        return ret

    @property
    def legal_actions(self):
        ret = np.zeros(self.action_space.n)
        ret.put(self.env.legal_actions(), 1)
        return ret

    def step(self, action):
        reward = [0,0]
        chooser = self.env.to_play()
        observation, pts, self.done = self.env.step(action)
        self.current_player_num = self.env.to_play()

        reward[chooser] = pts
        if self.done:
            i = 0 if (chooser == 1) else 1
            reward[i] = -pts

        return np.append(observation,self.legal_actions), reward, self.done, {}

    def reset(self):
        self.current_player_num = 0
        self.done = False
        return self.env.reset()

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
