
import gym
import numpy as np

import config

from stable_baselines import logger

class Player():
    def __init__(self, id):
        self.id = id

class AntiMancalaEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, verbose = False, manual = False):
        super(AntiMancalaEnv, self).__init__()
        self.name = 'antimancala'
        self.n_players = 2

        self.manual = manual

        self.board_size = 6
        self.num_pits = self.board_size * 2

        self.max_score = 44

        # for a turn, a player must pick one of their pits
        self.action_space = gym.spaces.Discrete(self.board_size)
        # 15 one-hot flags per pit on the board to denote the number of stones in that pit
        # one flag to denote the score of each player (out of max_score)
        # one flag for each pit for the player denoting their valid actions
        self.observation_space = gym.spaces.Box(0, 1, (self.num_pits * 26 + self.n_players + self.board_size,))
        self.verbose = verbose

        
    @property
    def observation(self):
        pit_num = 0
        if self.current_player.id == '1':
            start_i = 0
            opponent_i = 7
        else:
            start_i = 7
            opponent_i = 0
        obs = np.zeros([self.num_pits, 26]) # SET MAX NUMBER OF SEEDS IN ONE PIT
        for i in range(start_i, self.board_size + start_i):
            num_seeds = self.board[i]
            obs[pit_num][num_seeds] = 1
            pit_num += 1
        for i in range(opponent_i, self.board_size + opponent_i):
            num_seeds = self.board[i]
            obs[pit_num][num_seeds] = 1
            pit_num += 1

        obs = obs.flatten()

        for store_i in [start_i + 6, opponent_i + 6]:
            obs = np.append(obs, self.board[store_i] / self.max_score)

        out = np.append(obs,self.legal_actions)
        return out

    @property
    def legal_actions(self):
        legal_actions = []
        start_i = 0 if (self.current_player.id == '1') else 7
        for action_num in range(start_i, start_i + 6):
            if self.board[action_num] > 0: #non-empty pit
                legal_actions.append(1)
            else:
                legal_actions.append(0)
        return np.array(legal_actions)
        

    def check_game_over(self):

        # check game over
        p1_side = self.board[0:6]
        p2_side = self.board[7:13]

        if (sum(p1_side) == 0):
            return -1, True, 1, sum(p2_side)
        elif (sum(p2_side) == 0):
            return -1, True, 2, sum(p1_side)

        return 0, False, 0, 0

    @property
    def current_player(self):
        return self.players[self.current_player_num]


    def step(self, action):
        
        reward = [0,0]
        done = False
        self.bonus_turn = False
        
        # check move legality
        board = self.board
        pit_i = action if (self.current_player.id == '1') else (action + 7)
        
        if (board[pit_i] == 0):  # empty pit
            done = True
            reward = [1, 1]
            reward[self.current_player_num] = -1
        else:
            self.move_seeds(board, pit_i)

            self.turns_taken += 1

            storage = self.board_size if (self.current_player.id == '1') else (self.num_pits + 1)
            if self.board[storage] > 24:
                reward = [1,1]
                i = 0 if (storage == self.board_size) else 1
                reward[i] = -1
                done = True
            else:
                r, done, empty, total = self.check_game_over()
                reward = [r,r]
                if done:
                    if empty == 1:
                        self.board[-1] += total
                    else:
                        self.board[self.board_size] += total
                    p1_score = self.board[self.board_size]
                    p2_score = self.board[-1]

                    if p1_score > p2_score:
                        reward[1] = 1
                    elif p1_score < p2_score:
                        reward[0] = 1
                    else:
                        reward = [0,0]

        self.done = done

        if (not done) and (not self.bonus_turn):
            self.current_player_num = (self.current_player_num + 1) % 2

        return self.observation, reward, done, {}

    def move_seeds(self, board, pit_i):
        player_id = self.current_player.id
        seeds_in_hand = board[pit_i]
        board[pit_i] = 0
        while seeds_in_hand > 0:
            pit_i += 1
            if (pit_i == self.board_size and player_id == '2') or \
                    (pit_i == (self.num_pits + 1) and player_id == '1'):
                pit_i += 1
            pit_i %= (self.num_pits + 2)

            board[pit_i] += 1
            seeds_in_hand -= 1
        if pit_i in (self.board_size, self.num_pits + 1):
            self.bonus_turn = True
            logger.debug(f'\n-- BONUS TURN for Player {player_id}! --')
            # print(f'\n-- BONUS TURN for Player {player_id}! --')
        elif (board[pit_i] == 1):
            pit_j = -1
            if (player_id == '1' and pit_i in range(0,self.board_size)) or \
                (player_id == '2' and pit_i in range(self.board_size + 1,self.num_pits + 1)):
                pit_j = self.num_pits - pit_i
            if (pit_j > -1) and (board[pit_j] > 0):
                reward = 1 + board[pit_j]
                board[pit_i] = 0
                board[pit_j] = 0
                store_i = self.board_size if (player_id == '1') else (self.num_pits + 1)
                board[store_i] += reward
                logger.debug(f'\n-- Player {player_id} stole {reward - 1} stone(s)! --')
                # print(f'\n-- Player {player_id} stole {reward - 1} stone(s)! --')
        

    def reset(self):
        # the board is normally made of 18 spaces for beads:
        # 2 * 6 for the two players plus 2 for each player's mancala/store
        # the first 6 are the current player's pits, while the 7th is their store
        # the next 7 are the opponent's pits and store
        self.board = (([4] * self.board_size) + [0]) * 2
        self.players = [Player('1'), Player('2')]
        self.current_player_num = 0
        self.turns_taken = 0
        self.done = False
        self.bonus_turn = False
        logger.debug(f'\n\n---- NEW GAME ----')
        # print('\n\n---- NEW GAME ----')
        return self.observation


    def render(self, mode='human', close=False, verbose = True):
        logger.debug('')
        # print('')
        if close:
            return
        if self.done:
            logger.debug(f'GAME OVER')
            # print('GAME OVER')
        else:
            logger.debug(f"--- Turn {self.turns_taken} ---")
            logger.debug(f"It is Player {self.current_player.id}'s turn to move")
            # print(f"It is Player {self.current_player.id}'s turn to move")
        
        p2_side = [str(x) for x in reversed(self.board[self.board_size + 1:-1])]
        logger.debug('\t    ' + ' '.join(p2_side))
        logger.debug(f'P2 Score: {self.board[-1]}\t\t {self.board[self.board_size]} :P1 Score')
        # print('\t\t' + ' '.join(p2_side))
        # print(f'P2 Score: {self.board[-1]}\t\t P1 Score: {self.board[self.board_size]}')
        p1_side = [str(x) for x in self.board[0:self.board_size]]
        logger.debug('\t    ' + ' '.join(p1_side))
        # print('\t\t' + ' '.join(p1_side))

        if self.verbose:
            logger.debug(f'\nObservation: \n{self.observation}')
            # print(f'\nObservation: \n{self.observation}')
        
        if not self.done:
            logger.debug(f'\nLegal actions: {[i for i,o in enumerate(self.legal_actions) if o != 0]}')
            # print(f'\nLegal actions: {[i for i,o in enumerate(self.legal_actions) if o != 0]}')


    def rules_move(self):
        raise Exception('Rules based agent is not yet implemented for this game!')
