
import gym
import numpy as np

import config

from stable_baselines import logger

from .classes import *

class Player():
    def __init__(self, id: str):
        self.id = id

class ttykmch1Env(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, verbose = False, manual = False):
        super(ttykmch1Env, self).__init__()
        self.name = 'ttykmch1'
        self.n_players = 2
        self.manual = manual
        self.board = Board()
        self.n = self.board.size
        
        self.action_space = gym.spaces.Discrete( # 739
            3 * (16 * 4) # move a piece from any space to ortho-adjacent space on one of the 3 boards
            + 2 * 32 # for each square, time travel forward or back (if possible)

            + 3 * (16 * 5) # for each square, plant a seed on square or orthogonal
            + 3 * (16 * 5) # for each square, remove a seed on square or orthogonal

            + 3 # choose where to move the focus token to at end of turn
        )
        self.observation_space = gym.spaces.Box(-1, 1, ( # 414 + 739 = 1,153
            3 * 16 # locations of the player pieces (1 for friendly, -1 for opponent)
            + 2 * 3 # presence of a player in an era

            + 7 * (3 * 16) # for each square, what object (if any) is on it?
            + 6 # number of seeds currently in the supply

            + 2 * 4 # number of pieces in each players' reserve
            + 2 * 3 # location of focus token for each player
            + 2 # how many moves are left to take this turn?
            + 2 # which player (white or black?) / which direction do the boards go in
            + self.action_space.n ,)
        )
        self.verbose = verbose

        
    @property
    def observation(self):
        ret = np.zeros((3,4,4))
        presence = np.zeros((2,3))
        player_number = self.board.getPlayerToMove()
        for era in range(3):
            for (x,y),value in self.board.pieces[era].items():
                # set the appropriate square to the relative indicator
                # if black's turn, they see their pieces as 1's, white as -1
                ret[era][x][y] = value * player_number
                if value == player_number:
                    presence[0][era] = 1
                else:
                    presence[1][era] = 1
        if player_number == -1:
            tree_dict = {4:5,5:4,6:7,7:6}
            # black will see the boards reversed and flipped as well
            ret = ret[::-1]
            for i in range(3):
                ret[i] = np.rot90(ret[i],2)
            # for i in range(2):
            #     presence[i] = presence[i][::-1]
        ret = np.append(ret,presence)
        
        # what objects are on the board
        objects = np.zeros((3,4,4,7))
        for era in range(3):
            for (x,y),value in self.board.board_objects[era].items():
                if player_number == -1 and value in tree_dict:
                    value = tree_dict[value]
                # set the appropriate square to the relative indicator
                objects[era][x][y][value-1] = 1
        if player_number == -1:
            # black will see the boards reversed and flipped as well
            objects = objects[::-1]
            for i in range(3):
                objects[i] = np.rot90(objects[i],2)
        ret = np.append(ret,objects)

        # number of seeds in supply (specifying 0 too)
        seeds = np.zeros(6)
        seeds[self.board.seed_supply] = 1
        ret = np.append(ret,seeds)

        supplies = np.zeros((2,4))
        focus = np.zeros((2,3))
        foo = np.zeros(4)

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
            foo[self.board.remaining_actions - 1] = 1
        # which player's turn is it (should help with time travel direction?)
        foo[self.current_player_num + 2] = 1
        
        ret = np.append(ret,supplies)
        ret = np.append(ret,focus)
        ret = np.append(ret,foo)
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
                    if self.current_player_num == 1:
                        # need to flip the move for black
                        era = 2 - era
                        x1 = 3 - x1
                        x2 = 3 - x2
                        y1 = 3 - y1
                        y2 = 3 - y2
                        if special in (1,-1):
                            special *= -1
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
                    elif special == 2:
                        # plant a seed
                        if x1 == x2 and y1 == y2:
                            # same square
                            direction = 4
                        elif x1 != x2:
                            direction = 0 if (x2 < x1) else 2
                        else:
                            direction = 1 if (y2 > y1) else 3
                        i = 256 + 80 * era + (x1 * 4 + y1) * 5 + direction
                    elif special == -2:
                        # remove a seed
                        if x1 == x2 and y1 == y2:
                            # same square
                            direction = 4
                        elif x1 != x2:
                            direction = 0 if (x2 < x1) else 2
                        else:
                            direction = 1 if (y2 > y1) else 3
                        i = 496 + 80 * era + (x1 * 4 + y1) * 5 + direction
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

    def translate_move(self, action, current_player):
        move = []
        if action < 192:
            # orthogonal move
            era,foo = divmod(action,64)
            start_square,direction = divmod(foo,4)
            x1,y1 = divmod(start_square,4)
            ortho_coords = [(x1-1,y1),(x1,y1+1),(x1+1,y1),(x1,y1-1)]
            x2,y2 = ortho_coords[direction]
            move = [era,x1,y1,x2,y2,0]
        elif action < 224:
            # time travel forward
            era,foo = divmod(action-192,16)
            x1,y1 = divmod(foo,4)
            move = [era,x1,y1,x1,y1,1]
        elif action < 256:
            # time travel backward
            era,foo = divmod(action-224,16)
            x1,y1 = divmod(foo,4)
            move = [era + 1,x1,y1,x1,y1,-1]
        elif action < 496:
            # plant a seed
            era,foo = divmod(action-256,80)
            start_square,direction = divmod(foo,5)
            x1,y1 = divmod(start_square,4)
            ortho_coords = [(x1-1,y1),(x1,y1+1),(x1+1,y1),(x1,y1-1),(x1,y1)]
            x2,y2 = ortho_coords[direction]
            move = [era,x1,y1,x2,y2,2]
        elif action < 736:
            # remove a seed
            era,foo = divmod(action-496,80)
            start_square,direction = divmod(foo,5)
            x1,y1 = divmod(start_square,4)
            ortho_coords = [(x1-1,y1),(x1,y1+1),(x1+1,y1),(x1,y1-1),(x1,y1)]
            x2,y2 = ortho_coords[direction]
            move = [era,x1,y1,x2,y2,-2]
        if current_player == -1:
            # need to flip the move for the black player
            move[0] = 2 - move[0]
            for i in range(1,5):
                move[i] = 3 - move[i]
            if move[5] in (1,-1):
                move[5] *= -1
        return move

    def step(self, action):
        
        reward = [0,0]
        self.done = False
        p = self.board.getPlayerToMove()
        if action < (self.action_space.n - 3):
            # perform one of the two actions
            move_to_play = self.translate_move(action,p)
            self.board.execute_move(move_to_play)
            self.board.remaining_actions -= 1
            # check for game over after either making
            # the second move or if the chosen piece dies
            if self.board.remaining_actions == 0 or self.board.chosen_piece_era == None:
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
            self.board.chosen_piece_x = None
            self.board.chosen_piece_y = None
            self.board.chosen_piece_era = None
            self.board.time += 1
            if self.board.time > 90:
                self.done = True
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
            "-1": "b",
            "0": " ",
            "1": "w",
            "10": "s",
            "20": "R",
            "30": "T",
            "40": "^",
            "50": "v",
            "60": "<",
            "70": ">",
            "11": "W", # white copy on space with seed
            "9": "B", # black copy on space with seed
        }
        # render_chars = {
        #     "-1": "⬛",
        #     "0": "  ",
        #     "1": "⬜",
        #     "10": "🌰",
        #     "20": "🌵",
        #     "30": "🌴",
        #     "40": "⬆️",
        #     "50": "⬇️",
        #     "60": "⬅️",
        #     "70": "➡️",
        #     "11": "⚝", # white copy on space with seed
        #     "9": "★", # black copy on space with seed
        # }
        logger.debug(f"\nWhite: {self.board.copy_supplies[0]} Copies in Supply")
        logger.debug(f"Black: {self.board.copy_supplies[1]} Copies in Supply")
        logger.debug(f"{self.board.seed_supply} Seeds in Supply")

        logger.debug(f"\nActions left: {self.board.remaining_actions}")
        logger.debug(f"Chosen piece: ({self.board.chosen_piece_x},{self.board.chosen_piece_y}) in era {self.board.chosen_piece_era}")

        logger.debug("-" * 50)
        image = self.board.getImage()
        logger.debug(" " * (5+11*self.board.current_focuses[0]) + "<W>")

        foo = " ".join(str(i) for i in range(4)) + "  |"
        # foo = "  ".join(str(i) for i in range(4)) + "  |"
        logger.debug("  ",foo,foo,foo)
        for x in range(4):
            out = "{:2} ".format(x)
            for era in range(3):
                row = image[era][x]
                for y in range(4):
                    out += render_chars[str(row[y])] + " "
                    # out += render_chars[str(row[y])]
                out += " | "
                # out += "  |  "
            logger.debug(out)

        logger.debug(" " * (5+11*self.board.current_focuses[1]) + "<B>")
        logger.debug("-" * 50)

        if self.verbose:
            logger.debug(f'\nObservation: \n{[i if o == 1 else (i,o) for i,o in enumerate(self.observation) if o != 0]}')
        
        if not self.done:
            logger.debug(f'\nLegal actions: {[i for i,o in enumerate(self.legal_actions) if o != 0]}')

        if self.done:
            logger.debug(f'\n\nGAME OVER')
            logger.debug('\n    ------ Stats ------')
            logger.debug(f'- Number of Seeds Planted: {self.board.num_seeds_planted}')
            logger.debug(f'- Number of Copies Squished: {self.board.num_squishes}')
            logger.debug(f'- Number of Paradoxes: {self.board.num_paradoxes}')
            logger.debug(f'- Number of Plant-Based Kills: {self.board.plant_kills}')


    def rules_move(self):
        raise Exception('Rules based agent is not yet implemented for this game!')
