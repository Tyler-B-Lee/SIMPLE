import gym
import numpy as np
import random
import os
import time

import config
import utils.agents

from stable_baselines import logger
from stable_baselines.ppo1 import PPO1

RUNNER_ID = 1
MAIN_PLAYER_ID = 0
TURN_LIMIT = 60
POINT_SCALAR = 0.003
BEST_MODEL_CHANCE = 0.8

class Player():
    def __init__(self, id, token):
        self.id = id
        self.token = token
        

class Token():
    def __init__(self, symbol, number):
        self.number = number
        self.symbol = symbol
    

class ProxyEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, verbose = False, manual = False):
        super(ProxyEnv, self).__init__()
        self.name = 'proxychaser'
        self.manual = manual
        
        self.grid_length = 5
        self.n_players = 1

        self.num_squares = self.grid_length * self.grid_length

        self.action_space = gym.spaces.Discrete(self.num_squares)
        self.observation_space = gym.spaces.Box(-1, 1, (4*self.grid_length + self.grid_length*self.grid_length + 2 + self.action_space.n, ))
        self.verbose = verbose
        self.opp_model_pool = []

        opp_dir = os.path.join(config.MODELDIR, "proxyrunner")

        opp_files = os.listdir(opp_dir)
        for model_file_name in opp_files:
            full_path = os.path.join(opp_dir,model_file_name)
            if os.path.exists(full_path):
                logger.info(f"Loading {model_file_name} to play / train against...")
                cont = True
                while cont:
                    try:
                        ppo_model = PPO1.load(full_path)
                        cont = False
                    except Exception as e:
                        time.sleep(5)
                        print(e)
            else:
                raise Exception(f'\n{full_path} not found!')

            this_model = utils.agents.Agent(model_file_name[:-4], ppo_model)
            if model_file_name == "best_model.zip":
                self.best_opponent_model = this_model
            else:
                self.opp_model_pool.append(this_model)
        

    def get_manhattan_distance(self):
        "Returns the Manhattan distance between the runner and the chaser."
        return abs(self.runner_pos[0] - self.chaser_pos[0]) + abs(self.runner_pos[1] - self.chaser_pos[1])
    
    def runner_in_sight(self):
        "Returns True only if the runner is within one move of the chaser."
        horiz_diff = abs(self.runner_pos[0] - self.chaser_pos[0])
        vert_diff = abs(self.runner_pos[1] - self.chaser_pos[1])
        manh_dist = horiz_diff + vert_diff
        return (manh_dist <= 2) and (max(horiz_diff,vert_diff) <= 1)
        
    def get_legal_action_numbers(self):
        legal_actions = []
        if self.current_player_num == RUNNER_ID: # runner
            startx = self.runner_pos[0]
            starty = self.runner_pos[1]
        else:
            startx = self.chaser_pos[0]
            starty = self.chaser_pos[1]

        for i in range(-1,2):
            for j in range(-1,2):
                nx = startx + i
                ny = starty + j
                if (nx >= 0) and (ny >= 0) and (nx < self.grid_length) and (ny < self.grid_length):
                    legal_actions.append(nx * self.grid_length + ny)

        return legal_actions
    
    @property
    def legal_actions(self):
        legal_actions = np.zeros(self.action_space.n)      
        legal_actions.put(self.get_legal_action_numbers(), 1)

        return legal_actions

    @property
    def observation(self): # for the RUNNER
        return self.get_chaser_obs()

    @property
    def current_player(self):
        return self.players[self.current_player_num]
    
    def get_runner_obs(self): # length: 37 + 25
        ret = np.zeros(self.num_squares + 1)

        ret[self.runner_pos[0] * self.grid_length + self.runner_pos[1]] = 1
        ret[self.chaser_pos[0] * self.grid_length + self.chaser_pos[1]] = -1
        ret[-1] = self.turns_taken / TURN_LIMIT

        foo = np.zeros((2,self.grid_length))
        foo[0][self.runner_last_seen_pos[0]] = 1
        foo[1][self.runner_last_seen_pos[1]] = 1
        foo = np.append(foo, self.moves_since_runner_last_seen / TURN_LIMIT)

        ret = np.append(ret,foo)

        return np.append(ret,self.legal_actions)
    
    def get_chaser_obs(self): # length: 47 + 25
        ret = np.zeros((4,self.grid_length))
        ret[0][self.chaser_pos[0]] = 1
        ret[1][self.chaser_pos[1]] = 1
        ret[2][self.runner_last_seen_pos[0]] = 1
        ret[3][self.runner_last_seen_pos[1]] = 1
        ret = np.append(ret, self.turns_taken / TURN_LIMIT)
        ret = np.append(ret, self.moves_since_runner_last_seen / TURN_LIMIT)

        # "Vision" Grid
        # -1 if space is not in view
        # 0 if space is in view but runner is not there
        # 1 if space is in view and runner is there
        foo = np.full((self.grid_length,self.grid_length), -1)
        for i in range(-1,2):
            for j in range(-1,2):
                nx = self.chaser_pos[0] + i
                ny = self.chaser_pos[1] + j
                if (nx >= 0) and (ny >= 0) and (nx < self.grid_length) and (ny < self.grid_length):
                    foo[nx][ny] = 0
        if self.runner_in_sight():
            foo[self.runner_pos[0]][self.runner_pos[1]] = 1
        ret = np.append(ret,foo)

        return np.append(ret,self.legal_actions)

    def run_opponent_turn(self):
        self.render()

        opp = self.opponent_model
        logger.debug(f'\n{opp.name} model choices')
        # action_chosen = random.choice(self.get_legal_action_numbers())
        opponent_obs = self.get_runner_obs()

        action_probs = opp.model.action_probability(opponent_obs)
        value = opp.model.policy_pi.value(np.array([opponent_obs]))[0]
        logger.debug(f"Value {value:.3f}")

        opp.print_top_actions(action_probs)

        action_probs = utils.agents.mask_actions(self.legal_actions, action_probs)
        logger.debug('Masked ->')
        opp.print_top_actions(action_probs)

        action_chosen = utils.agents.sample_action(action_probs)
        logger.debug(f'Sampled action {action_chosen} chosen')
        
        reward, done = self.game_step(action_chosen)

        return reward,done

    def step(self, action):

        reward, done = self.game_step(action)
        # game_step CHANGES THE CURRENT PLAYER NUMBER!

        self.done = done

        # play the game until it comes back to the main player's turn or the game ends
        while self.current_player_num != MAIN_PLAYER_ID and (not self.done):
            r, done = self.run_opponent_turn()

            reward = [reward[i] + r[i] for i in range(2)]
            self.done = done

        return self.observation, reward[:1], done, {}
    
    def game_step(self, action):
        
        logger.debug(f"Action played: {action}")

        x,y = divmod(action, self.grid_length)
        if self.current_player_num == RUNNER_ID: # runner to move
            self.runner_pos = (x,y)
        else:
            self.chaser_pos = (x,y)

        self.turns_taken += 1

        if self.runner_pos == self.chaser_pos:
            logger.debug("> The Runner has been Caught by the Chaser!")
            reward = [1,-1]
            done = True
        elif self.turns_taken == TURN_LIMIT:
            logger.debug("Turn limit reached! The Runner has survived!")
            reward = [0,0]
            done = True
        else:
            # continue game
            if self.runner_in_sight(): # chaser can see the runner
                logger.debug("> Runner spotted!")
                self.moves_since_runner_last_seen = 0
                self.runner_last_seen_pos = self.runner_pos
            else:
                self.moves_since_runner_last_seen += 1

            manh_dist = self.get_manhattan_distance()
            reward = [(-manh_dist * POINT_SCALAR), (manh_dist * POINT_SCALAR)]
            
            if max(self.global_reward) > 0:
                reward = [reward[i] + self.global_reward[i] for i in range(2)]
                self.global_reward = [0,0]
            
            done = False

        if not done:
            self.current_player_num = (self.current_player_num + 1) % 2
            logger.debug(f"Not Done, next player: {self.current_player_num}")
                
        return reward, done

    def reset(self):
        # runner starts in top left, chaser in bottom right
        self.runner_pos = (0,0)
        self.chaser_pos = (self.grid_length - 1, self.grid_length - 1)
        self.players = [Player('Chaser', Token('O', -1)), Player('Runner', Token('X', 1))]
        self.current_player_num = 1
        self.turns_taken = 0

        self.runner_last_seen_pos = (0,0)
        self.moves_since_runner_last_seen = 0

        if random.random() < BEST_MODEL_CHANCE:
            self.opponent_model = self.best_opponent_model
        else:
            self.opponent_model = random.choice(self.opp_model_pool)
        logger.debug(f"> Chosen opponent: {self.opponent_model.name}")

        self.done = False
        logger.debug(f'\n\n---- NEW GAME ----')
        self.global_reward = [0,0]

        # play the game until it comes back to the main player's turn or the game ends
        while self.current_player_num != MAIN_PLAYER_ID and (not self.done):
            r, done = self.run_opponent_turn()

            self.global_reward = [self.global_reward[i] + r[i] for i in range(2)]
            self.done = done

        return self.observation


    def render(self, mode='human', close=False, verbose = True):
        logger.debug('')
        if close:
            return
        if self.done:
            logger.debug(f'GAME OVER')
        else:
            logger.debug(f"It is the {self.current_player.id}'s turn to move (Action {self.turns_taken + 1} / 60)")
            logger.debug(f"\tRunner last seen {self.moves_since_runner_last_seen} moves ago at {self.runner_last_seen_pos}")
        
        board = [['?'] * self.grid_length for _ in range(self.grid_length)]
        for i in range(-1,2):
            for j in range(-1,2):
                nx = self.chaser_pos[0] + i
                ny = self.chaser_pos[1] + j
                if (nx >= 0) and (ny >= 0) and (nx < self.grid_length) and (ny < self.grid_length):
                    board[nx][ny] = '.'

        board[self.runner_last_seen_pos[0]][self.runner_last_seen_pos[1]] = '!'
        board[self.chaser_pos[0]][self.chaser_pos[1]] = 'O'

        if self.runner_in_sight():
            board[self.runner_pos[0]][self.runner_pos[1]] = 'X'
        else:
            board[self.runner_pos[0]][self.runner_pos[1]] = 'x'

        for row in board:
            logger.debug(' '.join(row))

        if self.verbose:
            logger.debug(f'\nObservation: \n{self.observation}')
        
        if not self.done:
            logger.debug(f'\nLegal actions: {[i for i,o in enumerate(self.legal_actions) if o != 0]}')