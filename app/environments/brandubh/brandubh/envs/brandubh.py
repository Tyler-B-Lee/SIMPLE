
import gym
import numpy as np

import config

from stable_baselines import logger

from .classes import *

class Player():
    def __init__(self, id: str):
        self.id = id

class BrandubhEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, verbose = False, manual = False):
        super(BrandubhEnv, self).__init__()
        self.name = 'brandubh'
        self.n_players = 2
        self.manual = manual
        self.board = Board(Brandubh())
        self.n = self.board.size
        
        self.action_space = gym.spaces.Discrete(self.n ** 4) # 2,401
        self.observation_space = gym.spaces.Box(-1, 1, ( # 99 + 2,401 = 2,500
            1 # 1 or -1, is the current position meant to be played as white or black?
            + self.n ** 2 # locations of normal pieces
            + self.n ** 2 # location of the king
            + self.action_space.n ,)
        )
        self.verbose = verbose

        
    @property
    def observation(self):
        foo = np.zeros((self.n, self.n))
        king_loc = np.zeros((self.n, self.n))
        for p in self.board.pieces:
            if p[0] >= 0:
                if p[2] != 2:
                    foo[p[0]][p[1]] = p[2]
                else:
                    king_loc[p[0]][p[1]] = 1

        ret = np.zeros(1)
        ret[0] = self.board.getPlayerToMove()
        ret = np.append(ret, foo.flatten())
        ret = np.append(ret, king_loc.flatten())
        ret = np.append(ret, self.legal_actions)
        return ret

    @property
    def legal_actions(self):
        legal_actions = [0]*(self.n**4)
        legal_moves = self.board.get_legal_moves(self.board.getPlayerToMove())
        if len(legal_moves) == 0:
            legal_actions[-1] = 1
            return np.array(legal_actions)
        for x1, y1, x2, y2 in legal_moves:
            legal_actions[x1+y1*self.n+x2*self.n**2+y2*self.n**3] = 1
        return np.array(legal_actions)

    @property
    def current_player(self):
        n = 0 if (self.board.getPlayerToMove() > 0) else 1
        return self.players[n]

    def step(self, action):
        
        reward = [0,0]
        self.done = False
        p = self.board.getPlayerToMove()
        move_to_play = int2base(action, self.n, 4)
        self.board.execute_move(move_to_play, p)
        self.current_player_num = (self.current_player_num + 1) % 2

        done = self.board.done * p
        if done != 0:
            p_index = 0 if (p == 1) else 1
            reward[p_index] = done
            reward[(p_index + 1) % 2] = -done
            self.done = True

        return self.observation, reward, self.done, {}

    def reset(self):
        self.players = []
        player_id = 1
        for p in range(2):
            self.players.append(Player(str(player_id)))
            player_id += 1
        
        self.board = Board(Brandubh())
        self.current_player_num = 0
        self.done = False
        logger.debug(f'\n\n---- NEW GAME ----')
        return self.observation

    def render(self, mode='human', close=False):
        
        if close:
            return

        logger.debug(f'\n\n--------- TURN {self.board.time + 1}-----------')
        foo = "White" if (self.current_player.id == "1") else "Black"
        logger.debug(f"It is Player {self.current_player.id}'s turn ({foo})")

        render_chars = {
            "-1": "b",
            "0": " ",
            "1": "W",
            "2": "K",
            "10": "#",
            "12": "E",
            "20": "_",
            "22": "x"
        }
        logger.debug("\n---------------------")
        image = self.board.getImage()

        logger.debug("  ", " ".join(str(i) for i in range(len(image))))
        for i in range(len(image)-1,-1,-1):
            out = "{:2} ".format(i)

            row = image[i]
            for col in row:
                out += render_chars[str(col)] + " "
            logger.debug(out)

        logger.debug("\n---------------------")

        if self.verbose:
            logger.debug(f'\nObservation: \n{[i if o == 1 else (i,o) for i,o in enumerate(self.observation) if o != 0]}')
        
        if not self.done:
            logger.debug(f'\nLegal actions: {[i for i,o in enumerate(self.legal_actions) if o != 0]}')

        if self.done:
            logger.debug(f'\n\nGAME OVER')


    def rules_move(self):
        raise Exception('Rules based agent is not yet implemented for this game!')
