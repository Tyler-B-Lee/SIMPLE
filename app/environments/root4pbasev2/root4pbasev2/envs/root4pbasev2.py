import gym
import numpy as np

import config

from stable_baselines import logger
from .rootGameClasses.classes import *
from .rootGameClasses.rootMechanics import *

# Game Obs length: 1487 (down from 4474 i think)
# docker-compose exec app tensorboard --logdir ./logs
# notes:
# - os (optimizer stepsize) I think is the 'learning rate' parameter
#        - Should be decreased linearly over training to 0 or very small
#        - Some papers have it as small as 1e-6 at the end

# start (base), avg games around 150-160 actions per player, similar to random play
# docker-compose exec app mpirun -np 2 python3 train.py -e root4pbase -ne 30 -t 0.1 -ent 0.05 -os 0.0003
# gen 2, games 130 actions/player, ~285-300 games played (40,000 steps here)
#       Commander 50% pick 1st, Tinker 63% lol, Ranger 29%, Thief 8%
# docker-compose exec app mpirun -np 2 python3 train.py -e root4pbase -ne 24 -t 0.2 -ent 0.05 -os 0.0003



class rootEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, verbose = False, manual = False):
        super(rootEnv, self).__init__()
        self.name = 'root4pbasev2'
        self.n_players = 4
        self.manual = manual

        self.action_space = gym.spaces.Discrete(5679)
        self.observation_space = gym.spaces.Box(-1, 4, (
            1536
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
        ret = np.append(self.env.get_observation(),self.legal_actions)
        return ret

    @property
    def legal_actions(self):
        ret = np.zeros(self.action_space.n)
        ret.put(self.env.legal_actions(), 1)
        return ret

    def step(self, action):
        observation, reward, self.done = self.env.step(action)
        self.current_player_num = self.env.to_play()

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
