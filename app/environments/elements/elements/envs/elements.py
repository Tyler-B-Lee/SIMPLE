import gym
import numpy as np
import random

import config

from stable_baselines import logger

class Card():
    def __init__(self, value: int) -> None:
        self.value = value

class Deck():
    def __init__(self) -> None:
        self.cards = []
        for i in range(1,7):
            num_cards = 6 if (i==6) else 2
            for j in range(num_cards):
                c = Card(i)
                self.cards.append(c)
        self.shuffle()
    
    def shuffle(self):
        "Shuffles the deck."
        random.shuffle(self.cards)
    
    def draw(self) -> Card:
        "Draws and returns the next card in the deck (last index)."
        return self.cards.pop()

class Player():
    def __init__(self, id: str):
        self.id = id
        self.hand = []
        self.open_list = []
        self.cards_revealed = []
        self.victory_points = 0
        self.discarded_sixes = 0
    
    def pick(self, num: int) -> Card:
        """Removes and returns the first card found in the player's
        hand with a value of 'num'."""
        for i in range(len(self.hand)):
            card_i = self.hand[i]
            if card_i.value == num:
                card_chosen = self.hand.pop(i)
                self.cards_revealed.append(card_chosen)
                return card_chosen
        raise Exception(f"Tried to pick a card with value {num}, but none were found in the hand: {self.hand}")
    
    def get_card_total(self) -> int:
        """Returns the sum of the values of all of the cards in the
        player's hand and in their open list of cards."""
        x = [card.value for card in self.hand]
        y = [card.value for card in self.open_list]
        return sum(x + y)
    

class ElementsEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, verbose = False, manual = False):
        super(ElementsEnv, self).__init__()
        self.name = 'elements'
        self.n_players = 2

        self.manual = manual
        self.action_space = gym.spaces.Discrete( # Size: 10
            # for a turn, a player has several discrete actions possible:
            # play one of the 6 card types from their hand
            6
            # take the top card from the middle to their open cards
            + 1
            # Discard a 6 from their hand
            + 1
            # Knock
            + 1
            # Yield
            + 1
        )
        self.observation_space = gym.spaces.Box(0, 1, ( # Total: 188 + 10 = 198
            # how many of each card is in the player's hand (up to 2 for cards 1-5, up to 6 for 6)
            5 * 2 + 6
            # number of cards in opponent's hand
            + 12
            # which cards are in each player's open card lists
            + 2 * (5 * 2 + 6)
            # which cards each player has placed in this round
            + 2 * (5 * 2 + 6)
            # the order and type of cards in the center
            + 12 * 6
            # how many sixes have been discarded this round by each player
            + 6 * 2
            # current Victory Points per player
            + 2 * 6
            + self.action_space.n
            ,)
        )
        self.verbose = verbose
        
    @property
    def observation(self):
        p = self.current_player
        opp = self.opposing_player

        # current player's hand
        obs = np.zeros([5,2])
        num = [0] * 6
        for card in p.hand:
            num[card.value - 1] += 1
        for i in range(5):
            val = num[i]
            if val > 0:
                obs[i][val - 1] = 1
        foo = np.zeros(6)
        if num[5] > 0:
            foo[ num[5] - 1] = 1
        obs = np.append(obs.flatten(), foo)
        
        # number of cards in opponent's hand
        foo = np.zeros(12)
        val = len(opp.hand)
        if val > 0:
            foo[val - 1] = 1
        obs = np.append(obs, foo)

        # which cards are in the two open lists
        for lst in (p.open_list, opp.open_list):
            num = [0] * 6
            foo = np.zeros([5,2])
            for card in lst:
                num[card.value - 1] += 1
            for i in range(5):
                val = num[i]
                if val > 0:
                    foo[i][val - 1] = 1
            bar = np.zeros(6)
            if num[5] > 0:
                bar[ num[5] - 1 ] = 1
            obs = np.append(obs, foo.flatten())
            obs = np.append(obs, bar)
        
        # which cards have been revealed by each player
        for lst in (p.cards_revealed, opp.cards_revealed):
            num = [0] * 6
            foo = np.zeros([5,2])
            for card in lst:
                num[card.value - 1] += 1
            for i in range(5):
                val = num[i]
                if val > 0:
                    foo[i][val - 1] = 1
            bar = np.zeros(6)
            if num[5] > 0:
                bar[ num[5] - 1 ] = 1
            obs = np.append(obs, foo.flatten())
            obs = np.append(obs, bar)
        
        # card info/order in the center
        foo = np.zeros([12,6])
        # foo can now hold the information for up to 12 cards
        # it should stop after inputting the information for however many
        # cards are in the center, starting from the 'top' card
        i = 0
        for card in reversed(self.center):
            foo[i][card.value - 1] = 1
            i += 1
        obs = np.append(obs, foo.flatten())

        # number of discarded sixes and victory points per player
        foo = np.zeros([4,6])
        if p.discarded_sixes > 0:
            foo[0][p.discarded_sixes - 1] = 1
        if opp.discarded_sixes > 0:
            foo[1][opp.discarded_sixes - 1] = 1
        if p.victory_points > 0:
            foo[2][min(p.victory_points - 1, 5)] = 1
        if opp.victory_points > 0:
            foo[3][min(opp.victory_points - 1, 5)] = 1
        obs = np.append(obs, foo.flatten())

        obs = np.append(obs,self.legal_actions)
        return obs

    @property
    def legal_actions(self):
        legal_actions = np.zeros(10)
        p = self.current_player
        foo = set(card.value for card in p.hand)
        # playing a card from hand
        for i in range(1,7):
            if i in foo:
                legal_actions[i - 1] = 1
        # taking a card from center to
        if len(self.center) > 0:
            legal_actions[6] = 1
        # discarding a 6 from hand
        if (6 in foo):
            legal_actions[7] = 1
        # knocking
        lim = sum([card.value for card in self.center])
        if p.get_card_total() <= lim:
            legal_actions[8] = 1
        # yielding the round
        legal_actions[9] = 1
        return legal_actions

    @property
    def current_player(self):
        return self.players[self.current_player_num]
    @property
    def opposing_player(self):
        return self.players[(self.current_player_num + 1) % 2]
    
    def game_over_check(self):
        """Returns a tuple:
        - The reward list (1 for the winner, or both 0's if game not over yet)
        - True if game is over, else False"""
        reward = [0,0]
        if self.current_player.victory_points >= 6:
            reward = [-1,-1]
            reward[self.current_player_num] = 1
            return reward, True
        elif self.opposing_player.victory_points >= 6:
            reward = [-1,-1]
            reward[(self.current_player_num + 1) % 2] = 1
            return reward, True
        else:
            return reward, False

    def step(self, action: int):
        reward = [0,0]
        done = False
        p = self.current_player
        if action in range(6):
            # player wants to add a card with value (action+1)
            card_chosen = p.pick(action+1)
            self.center.append(card_chosen)
            self.turns_taken += 1
            self.current_player_num = (self.current_player_num + 1) % 2
            logger.debug(f"\n--- Player {p.id} adds a {action + 1} to the center.")
        elif action == 6:
            # player wants to take top card in center
            card_chosen = self.center.pop()
            p.open_list.append(card_chosen)
            self.turns_taken += 1
            self.current_player_num = (self.current_player_num + 1) % 2
            logger.debug(f"\n--- Player {p.id} takes the top {card_chosen.value}.")
        elif action == 7:
            # Discard 6 from current player's hand
            card_chosen = p.pick(6)
            p.discarded_sixes += 1
            self.turns_taken += 1
            self.current_player_num = (self.current_player_num + 1) % 2
            logger.debug(f"\n--- Player {p.id} discards a 6 from their hand.")
        elif action == 8:
            # Current player knocks (assumed valid)
            current_player_sum = p.get_card_total()
            opposing_sum = self.opposing_player.get_card_total()
            limit = sum([card.value for card in self.center])
            if opposing_sum > limit:
                p.victory_points += 2
                next_start_num = self.current_player_num
                logger.debug(f"\n        Player {p.id} has knocked and won the round! (Their {current_player_sum} vs opponent's {opposing_sum} for limit of {limit}).")
                logger.debug(f"  - They gain 2 points.")
            elif opposing_sum >= current_player_sum:
                self.opposing_player.victory_points += 2
                next_start_num = (self.current_player_num + 1) % 2
                logger.debug(f"\n        Player {p.id} has knocked but lost the round! (Their {current_player_sum} vs opponent's {opposing_sum} for limit of {limit}).")
                logger.debug(f"  - Their opponent gains 2 points.")
            else:
                p.victory_points += 2
                next_start_num = self.current_player_num
                logger.debug(f"\n        Player {p.id} has knocked and won the round! (Their {current_player_sum} vs opponent's {opposing_sum} for limit of {limit}).")
                logger.debug(f"  - They gain 2 points.")
            self.turns_taken = 1
            self.round += 1
            # check for game over
            reward, done = self.game_over_check()
            # if not done, set winner to start next round
            if not done:
                self.current_player_num = next_start_num
                self.setup_new_round()
        elif action == 9:
            # Current player Yields the round
            self.opposing_player.victory_points += 1
            self.turns_taken = 1
            self.round += 1
            logger.debug(f"\n        Player {p.id} yields the round! Their opponent gains 1 point.")
            # check for game over
            reward, done = self.game_over_check()
            # if not done, set winner to start next round
            if not done:
                self.current_player_num = (self.current_player_num + 1) % 2
                self.setup_new_round()

        return self.observation, reward, done, {}

    def setup_new_round(self):
        # Make a new deck, shuffled by default on creation
        logger.debug("\nStarting the next round!")
        self.center = []
        self.deck = Deck()
        for p in self.players:
            p.discarded_sixes = 0
            p.hand = []
            p.open_list = []
            p.cards_revealed = []
            for i in range(6):
                card = self.deck.draw()
                p.hand.append(card)

    def reset(self):
        # Make a new deck, shuffled by default on creation
        self.center = []
        self.deck = Deck()
        self.players = [Player('1'), Player('2')]
        for p in self.players:
            for i in range(6):
                card = self.deck.draw()
                p.hand.append(card)

        self.current_player_num = 0
        self.turns_taken = 1
        self.round = 1
        self.done = False
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
            logger.debug(f"\n     - ROUND {self.round} - ")
            logger.debug(f"--- Turn {self.turns_taken} ---")
            logger.debug(f"It is Player {self.current_player.id}'s turn.")
            # print(f"It is Player {self.current_player.id}'s turn to move")
        
        p = self.current_player
        opp = self.opposing_player
        logger.debug(f"Opponent - Player {opp.id} / {opp.victory_points} Victory Points")
        foo = " ".join([str(card.value) for card in opp.cards_revealed])
        logger.debug(f"  Cards Revealed: {foo}")
        logger.debug(f"    # of Sixes Discarded this Round: {opp.discarded_sixes} / Cards In Hand: {len(opp.hand)}")
        foo = " ".join([str(card.value) for card in opp.open_list])
        logger.debug(f"    Taken Cards: {foo} ({sum([card.value for card in opp.open_list])})")
        
        foo = " ".join([str(card.value) for card in self.center])
        logger.debug(f"\n\nCenter (Bottom to Top): {foo} ({sum([card.value for card in self.center])})")

        logger.debug(f"\n\nCurrent Player - Player {p.id} / {p.victory_points} Victory Points")
        foo = " ".join([str(card.value) for card in p.cards_revealed])
        logger.debug(f"  Cards Revealed: {foo}")
        foo = " ".join([str(card.value) for card in p.hand])
        logger.debug(f"    # of Sixes Discarded this Round: {p.discarded_sixes} / Current Hand: {foo} ({sum([card.value for card in p.hand])})")
        foo = " ".join([str(card.value) for card in p.open_list])
        logger.debug(f"    Taken Cards: {foo} ({sum([card.value for card in p.open_list])})")
        logger.debug(f"      Total Card Value: {p.get_card_total()}")

        if self.verbose:
            logger.debug(f'\n\nObservation: \n{self.observation}')
            # print(f'\nObservation: \n{self.observation}')
        
        if not self.done:
            logger.debug(f'\nLegal actions: {[i for i,o in enumerate(self.legal_actions) if o != 0]}')
            # print(f'\nLegal actions: {[i for i,o in enumerate(self.legal_actions) if o != 0]}')


    def rules_move(self):
        raise Exception('Rules based agent is not yet implemented for this game!')
