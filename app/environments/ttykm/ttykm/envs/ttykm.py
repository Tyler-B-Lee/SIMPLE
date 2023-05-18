
import gym
import numpy as np

import config

from stable_baselines import logger

from .classes import *

class Player():
    def __init__(self, id: str):
        self.id = id

class ttykmEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, verbose = False, manual = False):
        super(ttykmEnv, self).__init__()
        self.name = 'ttykm'
        self.n_players = 2
        self.manual = manual
        self.board = Board()
        self.n = self.board.size
        
        self.action_space = gym.spaces.Discrete( # 259
            3 * (16 * 4) # move a piece from any space to ortho-adjacent space on one of the 3 boards
            + 2 * 32 # for each square, time travel forward or back (if possible)
            + 3 # choose where to move the focus token to at end of turn
        )
        self.observation_space = gym.spaces.Box(-1, 1, ( # 70 + 259 = 329
            3 * 16 # locations of the player pieces (1 for friendly, -1 for opponent)
            + 2 * 3 # presence of a player in an era
            + 2 * 4 # number of pieces in each players' reserve
            + 2 * 3 # location of focus token for each player
            + 2 # how many moves are left to take this turn?
            + self.action_space.n ,)
        )
        self.verbose = verbose

        
    @property
    def observation(self):
        ret = np.zeros((3,4,4))
        presence = np.zeros((2,3))
        player_number = self.board.getPlayerToMove()
        for era in range(3):
            for p in self.board.pieces[era]:
                if p[0] >= 0:
                    # set the appropriate square to the relative indicator
                    # if black's turn, they see their pieces as 1's, white as -1
                    ret[era][p[0]][p[1]] = p[2] * player_number
                    if p[2] == player_number:
                        presence[0][era] = 1
                    else:
                        presence[1][era] = 1
        ret = np.append(ret,presence)
        
        supplies = np.zeros((2,4))
        focus = np.zeros((2,3))
        actions_left = np.zeros(2)
        
        if player_number == 1:
            if self.board.copy_supplies[0] > 0:
                supplies[0][self.board.copy_supplies[0] - 1] = 1
            if self.board.copy_supplies[1] > 0:
                supplies[1][self.board.copy_supplies[1] - 1] = 1
            focus[0][self.board.current_focuses[0]] = 1
            focus[1][self.board.current_focuses[1]] = 1
        else:
            if self.board.copy_supplies[1] > 0:
                supplies[0][self.board.copy_supplies[1] - 1] = 1
            if self.board.copy_supplies[0] > 0:
                supplies[1][self.board.copy_supplies[0] - 1] = 1
            focus[0][self.board.current_focuses[1]] = 1
            focus[1][self.board.current_focuses[0]] = 1
        if self.board.remaining_actions > 0:
            actions_left[self.board.remaining_actions - 1] = 1
        
        ret = np.append(ret,supplies)
        ret = np.append(ret,focus)
        ret = np.append(ret,actions_left)
        ret = np.append(ret,self.legal_actions)
        return ret

    @property
    def legal_actions(self):
        legal_actions = np.zeros(self.action_space.n)
        
        ra = self.board.remaining_actions
        if ra > 0:
            # actions remain to use
            legal_moves = self.board.get_legal_moves(self.board.getPlayerToMove())
            if len(legal_moves) > 0:
                for era, x1, y1, x2, y2, special in legal_moves:
                    if special == 0:
                        # this move is an orthogonal move
                        if x1 != x2:
                            direction = 0 if (x2 < x1) else 2
                        else:
                            direction = 1 if (y2 > y1) else 3
                        i = era * 64 + (x1 * 4 + y1) * 4 + direction
                    elif special == 1:
                        # time travel forward
                        i = 192 + 16 * era + x1 * 4 + y1
                    elif special == -1:
                        # time travel backward
                        i = 224 + 16 * (era - 1) + x1 * 4 + y1
                    legal_actions[i] = 1
                return legal_actions
        # we get here if all actions are used up
        # or the player has no legal moves
        # must move focus token to different era
        for i in range(-3,0):
            if i != (-3 + self.board.current_focuses[self.board.getPlayerIndex()]):
                legal_actions[i] = 1
        return legal_actions

    @property
    def current_player(self):
        n = 0 if (self.board.getPlayerToMove() > 0) else 1
        return self.players[n]

    def translate_move(self, action):
        if action < 192:
            # orthogonal move
            era,foo = divmod(action,64)
            start_square,direction = divmod(foo,4)
            x1,y1 = divmod(start_square,4)
            
            ortho_coords = [(x1-1,y1),(x1,y1+1),(x1+1,y1),(x1,y1-1)]
            x2,y2 = ortho_coords[direction]
            return [era,x1,y1,x2,y2,0]
        elif action < 224:
            # time travel forward
            era,foo = divmod(action-192,16)
            x1,y1 = divmod(foo,4)
            return [era,x1,y1,x1,y1,1]
        elif action < 256:
            # time travel backward
            era,foo = divmod(action-224,16)
            x1,y1 = divmod(foo,4)
            return [era + 1,x1,y1,x1,y1,-1]

    def step(self, action):
        
        reward = [0,0]
        self.done = False
        p = self.board.getPlayerToMove()
        if action < (self.action_space.n - 3):
            # perform one of the two actions
            move_to_play = self.translate_move(action)
            era,cpn = self.board.execute_move(move_to_play)
            # update the chosen piece number and era
            # to account for time travel
            self.board.chosen_pieceno = cpn
            self.board.chosen_piece_era = era
            self.board.remaining_actions -= 1
            # check for game over after either making
            # the second move or if the chosen piece dies
            if self.board.remaining_actions == 0 or cpn == -1:
                self.board.done = self.board._getWinLose()
        elif self.board.remaining_actions != 0:
            # this means we were forced to make 
            self.board.done = self.board._getWinLose()
        
        done = self.board.done * p
        if done != 0:
            p_index = 0 if (p == 1) else 1
            reward[p_index] = done
            reward[(p_index + 1) % 2] = -done
            self.done = True
        elif action >= (self.action_space.n - 3):
            # we are not done with the current game and
            # the current player must shift their focus
            if p == 1:
                self.board.current_focuses[0] = action - self.action_space.n + 3
            else:
                self.board.current_focuses[1] = action - self.action_space.n + 3
            # setup the next player's turn
            self.board.remaining_actions = 2
            self.board.chosen_piece = None
            self.board.chosen_piece_era = None
            self.board.time += 1
            self.current_player_num = (self.current_player_num + 1) % 2

        return self.observation, reward, self.done, {}

    def reset(self):
        self.players = []
        player_id = 1
        for p in range(2):
            self.players.append(Player(str(player_id)))
            player_id += 1
        
        # white goes first, starting focus in past
        self.board = Board()
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
            "-1": "B",
            "0": " ",
            "1": "W",
            "10": "#",
            "12": "E",
            "20": "_",
            "22": "x"
        }
        logger.debug(f"\nWhite: {self.board.copy_supplies[0]} Copies in Supply")
        logger.debug(f"Black: {self.board.copy_supplies[1]} Copies in Supply")

        logger.debug(f"Actions left: {self.board.remaining_actions}")
        apiece = None if (self.board.chosen_piece_era == None) else self.board.pieces[self.board.chosen_piece_era][self.board.chosen_pieceno]
        foo = self.board.chosen_piece_era or self.board.current_focuses[self.current_player_num]
        logger.debug(f"Chosen piece: {apiece} in era {foo}")

        logger.debug("-------------------------------")
        image = self.board.getImage()
        logger.debug(" " * (5+11*self.board.current_focuses[0]) + "<W>")

        foo = " ".join(str(i) for i in range(4)) + "  |"
        logger.debug("  ",foo,foo,foo)
        for x in range(4):
            out = "{:2} ".format(x)
            for era in range(3):
                row = image[era][x]
                for y in range(4):
                    out += render_chars[str(row[y])] + " "
                out += " | "
            logger.debug(out)

        logger.debug(" " * (5+11*self.board.current_focuses[1]) + "<B>")
        logger.debug("-------------------------------")

        if self.verbose:
            logger.debug(f'\nObservation: \n{[i if o == 1 else (i,o) for i,o in enumerate(self.observation) if o != 0]}')
        
        if not self.done:
            logger.debug(f'\nLegal actions: {[i for i,o in enumerate(self.legal_actions) if o != 0]}')

        if self.done:
            logger.debug(f'\n\nGAME OVER')


    def rules_move(self):
        raise Exception('Rules based agent is not yet implemented for this game!')
