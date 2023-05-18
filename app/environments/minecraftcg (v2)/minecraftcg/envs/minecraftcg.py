
import gym
import numpy as np

import config

from stable_baselines import logger

from .classes import *

class MinecraftCGEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, verbose = False, manual = False):
        super(MinecraftCGEnv, self).__init__()
        self.name = 'minecraftcg'
        self.manual = manual
        
        self.n_players = 3
        self.resource_card_types = 20
        self.total_craft_cards = self.craft_card_types = 25
        
        self.max_score = 20 # depends on number of players: 2=24,3=20,4=16
        
        self.resource_contents = [
          {'card': ResourceCard, 'info': {'name': 'wild1', 'material': 'wild', 'amount': 1}, 'count': 7}  #0 
            ,  {'card': ResourceCard, 'info': {'name': 'wild2', 'material': 'wild', 'amount': 2}, 'count': 3} #1 
            ,  {'card': ResourceCard, 'info': {'name': 'wild3', 'material': 'wild', 'amount': 3}, 'count': 1}  #2   
            , {'card': ResourceCard, 'info': {'name': 'diamond1', 'material': 'diamond', 'amount': 1}, 'count': 3}
            , {'card': ResourceCard, 'info': {'name': 'diamond2', 'material': 'diamond', 'amount': 2}, 'count': 2}
            , {'card': ResourceCard, 'info': {'name': 'diamond3', 'material': 'diamond', 'amount': 3}, 'count': 1} #5
            , {'card': ResourceCard, 'info': {'name': 'gold1', 'material': 'gold', 'amount': 1}, 'count': 4}
            , {'card': ResourceCard, 'info': {'name': 'gold2', 'material': 'gold', 'amount': 2}, 'count': 2}
            , {'card': ResourceCard, 'info': {'name': 'gold3', 'material': 'gold', 'amount': 3}, 'count': 1} #8
            , {'card': ResourceCard, 'info': {'name': 'iron1', 'material': 'iron', 'amount': 1}, 'count': 4}
            , {'card': ResourceCard, 'info': {'name': 'iron2', 'material': 'iron', 'amount': 2}, 'count': 2}
            , {'card': ResourceCard, 'info': {'name': 'iron3', 'material': 'iron', 'amount': 3}, 'count': 2}
            , {'card': ResourceCard, 'info': {'name': 'stone1', 'material': 'stone', 'amount': 1}, 'count': 4} #12
            , {'card': ResourceCard, 'info': {'name': 'stone2', 'material': 'stone', 'amount': 2}, 'count': 3}
            , {'card': ResourceCard, 'info': {'name': 'stone3', 'material': 'stone', 'amount': 3}, 'count': 2}
            , {'card': ResourceCard, 'info': {'name': 'wood1', 'material': 'wood', 'amount': 1}, 'count': 8}
            , {'card': ResourceCard, 'info': {'name': 'wood2', 'material': 'wood', 'amount': 2}, 'count': 10} #16
            , {'card': ResourceCard, 'info': {'name': 'wood3', 'material': 'wood', 'amount': 3}, 'count': 6}
            , {'card': Creeper, 'info': {'name': 'creeper'}, 'count': 5}
            , {'card': TNT, 'info': {'name': 'tnt'}, 'count': 5} #19
        ]
        self.name_action_mapping = {
            'wild1': 0, 'wild2': 1, 'wild3': 2, 'diamond1': 3, 'diamond2': 4, 'diamond3': 5,
            'gold1': 6, 'gold2': 7, 'gold3': 8, 'iron1': 9, 'iron2': 10, 'iron3': 11,
            'stone1': 12, 'stone2': 13, 'stone3': 14, 'wood1': 15, 'wood2': 16, 'wood3': 17
        }
        points = [
            [1,2,2,3,4],
            [1,2,3,3,4],
            [1,3,3,4,5],
            [1,3,3,4,5],
            [2,3,4,5,6]
        ]

        self.craft_contents = []
        for i, material in enumerate(['wood','stone','iron','gold','diamond']):
            for j, tool in enumerate(['hoe','shovel','sword','axe','pickaxe']):
                materials_dict = {'wood': 1, 'stone': 0, 'iron': 0, 'gold': 0, 'diamond': 0}
                if tool in ('shovel','pickaxe'):
                    materials_dict['wood'] += 1
                if tool in ('shovel','hoe'):
                    num_mats = 1
                elif tool == 'sword':
                    num_mats = 2
                else:
                    num_mats = 3
                
                if material == 'wood':
                    materials_dict['wood'] += num_mats
                else:
                    materials_dict[material] = num_mats

                card = {'card': CraftCard, 'info': {'name': f'{material}-{tool}', 'material': material, 
                'tool': tool, 'points': points[i][j], 'recipe': materials_dict}, 'count': 1}
                self.craft_contents.append(card)
        # default is 75 total cards
        self.total_resource_cards = sum([x['count'] for x in self.resource_contents])

        # ideas for actions, TRC=total resource cards (75), TCC=total craft cards (25):
        self.action_space = gym.spaces.Discrete(
        # Moves that use up 1 Action on your turn:
        # - Mine from one of the Resource piles (Piles 1-5)
        5
        # - Reserve one of the visible recipes (Piles 1-4)
        + 4
        # - Craft one of the visible recipes (Piles 1-4)
        # - Craft your reserved recipe (1)
        # - Axe: Counts as 2 wood when crafting (2, use or not?)
        # Must add duplicate options for crafting specifically using up some or no axes
        #    LOL realized that up to 3 axes can be used at once (wooden pick needs 5 wood)
        #    Must have options for crafting with 0, 1, 2, or 3 axes
        + ((4 + 1) * 4)
        
        #     - Crafting itself can have choices, may have multiple combinations possible to craft
        #     - When choosing to craft, sent to a special set of turns to pick which cards to use up
        #       - Choosing a card to use to craft it, 1 per resource card type (18 without creeper/tnt)
        #       - Like "adding ingredient to crafting bench" until enough is added to make
        + self.resource_card_types - 2 # 18 default
        # - When TNT is mined, you get to pick 2 of the 4 other top cards, and the other 2 are discarded
        # - Adds one option per pair of piles possible
        + 10
        # - When a creeper is revealed, everyone must discard 1 resource card
        #     - 1 discard possible per resource card type
        #       - Is this action the same as the 18 crafting options above? Guessing no
        + self.resource_card_types - 2
        #     - BUT, if player has a sword, they may use it to avoid this (use sword is a choice instead of one of 18)
        + 1
        # Plus a dummy action when a player has nothing to discard
        + 1
        # Using Tool Powers, no Action cost, can be done "any time"?
        # - Sword discussed above
        # - Shovel: 1 choice per enemy player (2 enemies default)
        + self.n_players - 1
        # - Pickaxe: Use for bonus Action on turn
        + 1
        # - Hoe: Clear top card of all resource piles
        + 1
        # - Choice to end turn now, regardless of what tools/actions a player could do here
        + 1
        )

        self.observation_space = gym.spaces.Box(0, 1, (
        # which cards are the top ones for the Draw Piles?
        # Resource cards, number of positions:
        # 5 Resource piles + n player positions + sure in discard pile or unsure about location (1 or 0 respectively)
        (5 + self.n_players + 1) * self.total_resource_cards
        # Crafting Cards:
        # - 4 Crafting Piles times # of different Craft/Tool Cards + player reserves
        #    - Plus the crafted tool positions for each player (+2 per player, used and unused tools)
        #    - Plus the one recipe left out of the game? (removed for now)
        + (4 + 2 * self.n_players + self.n_players) * self.total_craft_cards
        # sizes of each of the 5 resource piles / 4 craft piles
        + 5 + 4
        # current player scores
        + self.n_players
        # current game state (player turn, crafting, tnt, or creeper)
        + 4
        # number of actions each player has at the moment
        + self.n_players
        # for crafting - amount of each resource owed out of the max possible
        + 5
        # for creeper attack - number of cards the current player still owes
        + 1
        # possible actions for this turn
        + self.action_space.n
        ,))
        self.verbose = verbose

        
    @property
    def observation(self):
        # first, locations of Resource Cards
        obs = np.zeros(([(5 + self.n_players + 1), self.total_resource_cards]))
        obs2 = np.zeros(([(4 + 2 * self.n_players + self.n_players),self.total_craft_cards]))

        # top card on the five resource piles
        piles = self.resource_piles
        for i in range(5):
            pile_i = piles[i]
            if pile_i.size() > 0:
                top_card_id = pile_i.peek().id
                obs[i][top_card_id] = 1
        
        # top card of the crafting piles
        piles = self.craft_piles
        for i in range(4):
            pile_i = piles[i]
            if pile_i.size() > 0: # because these could run out
                top_card_id = pile_i.peek().id
                obs2[i][top_card_id] = 1
        
        # inventory of each player, going from current player around in turn order
        player_num = self.current_player_num
        for i in range(self.n_players):
            p = self.players[player_num]

            # Resource Cards
            player_resource_list = p.resources.cards
            for card in player_resource_list:
                obs[5 + i][card.id] = 1
            
            # Tool Cards
            player_crafting_list = p.tools.cards
            for card in player_crafting_list:
                if card.used:
                    obs2[4 + 2*i][card.id] = 1
                else:
                    obs2[4 + 2*i + 1][card.id] = 1

            # Reserve
            if p.reserve != None:
                obs2[-self.n_players + i][p.reserve.id] = 1

            player_num = (player_num + 1) % self.n_players
        
        # 1 = Resource card is certainly in discard pile, 0 otherwise
        for card in self.certain_discarded:
            obs[-1][card.id] = 1
        
        # Which craft card is out of the game from the start
        #obs2[-1][self.leftover_recipe.id] = 1

        ret = np.append(obs.flatten(), obs2.flatten())
        
        pile_sizes = np.zeros((9,))
        for i in range(5):
            pile_sizes[i] = (self.resource_piles[i].size() / 15)
        for i in range(5,9):
            pile_sizes[i] = (self.craft_piles[i - 5].size() / 6)
        ret = np.append(ret, pile_sizes)

        # what the current game state is
        foo = np.zeros((4,))
        if self.game_state < 2:
            foo[0] = 1
        else:
            foo[self.game_state - 1] = 1
        ret = np.append(ret, foo)

        # number of actions each player has
        foo = np.zeros((self.n_players,))

        player_num = self.current_player_num
        for i in range(self.n_players):
            p = self.players[player_num]
            foo[i] = min(p.actions / 2, 1)
            # print(len(ret) + i)
            # print(score_obs[i])
            player_num = (player_num + 1) % self.n_players

        ret = np.append(ret, foo)

        # materials owed if crafting
        foo = np.zeros((5,))
        if self.game_state == 2:
            foo[0] = max(self.materials_owed['wood'] / 5, 0)
            for i, material in enumerate(['stone','iron','gold','diamond']):
                foo[i + 1] = max(self.materials_owed[material] / 3, 0)
        ret = np.append(ret, foo)

        # number of cards still owed if creeper attack
        foo = 0
        if self.game_state == 4:
            foo = self.num_creepers / 5
        ret = np.append(ret, [foo])

        # Adding the score of each player
        score_obs = np.zeros((self.n_players, ))

        player_num = self.current_player_num
        for i in range(self.n_players):
            p = self.players[player_num]
            score_obs[i] = min(p.score / self.max_score, 1)
            # print(len(ret) + i)
            # print(score_obs[i])
            player_num = (player_num + 1) % self.n_players

        ret = np.append(ret, score_obs)

        # print('Legal actions')
        # for i in range(len(self.legal_actions)):
        #     if self.legal_actions[i] == 1:
        #         print(len(ret) + i)
        ret = np.append(ret, self.legal_actions)

        return ret

    @property
    def legal_actions(self):
        p = self.current_player
        legal_actions = np.zeros(self.action_space.n)
        if self.game_state == 0:
            i = 0 # resource piles
            for pile in self.resource_piles:
                if pile.size() > 0:
                    legal_actions[i] = 1
                i += 1
            
            if p.reserve == None: # reserving a recipe
                i = 5
                for pile in self.craft_piles:
                    if pile.size() > 0:
                        legal_actions[i] = 1
                    i += 1
            
            # choosing to start crafting
            num_axes = min(3, p.num_usable('axe'))
            foo = set()
            for n in range(num_axes + 1):
                i = 9
                for pile in self.craft_piles:
                    if pile.size() > 0:
                        rcp = pile.peek().recipe.copy()
                        rcp['wood'] -= (n * 2)
                        if (rcp['wood'] > -2):
                            if all([x < 1 for x in rcp.values()]):
                                legal_actions[i + 5*n] = 1
                            elif p.resources.size() > 0:
                                mat = pile.peek().material
                                c = p.resources.cards[0]
                                if p.crafting_helper(mat, c, foo, rcp):
                                    legal_actions[i + 5*n] = 1
                    i += 1
                # i should be 13 by now
                if p.reserve != None:
                    rcp = p.reserve.recipe.copy()
                    rcp['wood'] -= (n * 2)
                    if (rcp['wood'] > -2):
                        if all([x < 1 for x in rcp.values()]):
                            legal_actions[i + 5*n] = 1
                        elif p.resources.size() > 0:
                            mat = p.reserve.material
                            c = p.resources.cards[0]
                            if p.crafting_helper(mat, c, foo, rcp):
                                legal_actions[i + 5*n] = 1
            
            # Using extra tools
            if p.num_usable('shovel') > 0:
                for j in range(1,self.n_players):
                    opponent = self.players[ (self.current_player_num + j) % self.n_players ]
                    if opponent.actions > 0:
                        legal_actions[76 + j] = 1
            if p.num_usable('pickaxe') > 0:
                legal_actions[79] = 1
            if p.num_usable('hoe') > 0:
                legal_actions[80] = 1

        elif self.game_state == 1: # 0 actions left
            if p.num_usable('shovel') > 0:
                for j in range(1,self.n_players):
                    opponent = self.players[ (self.current_player_num + j) % self.n_players ]
                    if opponent.actions > 0:
                        legal_actions[76 + j] = 1
            if p.num_usable('pickaxe') > 0:
                legal_actions[79] = 1
            if p.num_usable('hoe') > 0:
                legal_actions[80] = 1
            legal_actions[81] = 1
        
        elif self.game_state == 2: # in the middle of crafting
            tool_mat = self.recipe_to_craft.material
            if self.materials_owed[tool_mat] > 0:
                # print(f'    Still need {tool_mat}')
                mat_needed = tool_mat
            else:
                # print('     Still need wood')
                mat_needed = 'wood'
            checked_names = set()
            foo = set()
            for card in p.resources.cards:
                if (card.name not in checked_names) and ((card.material == mat_needed) or (card.material == 'wild')):
                    possible = p.crafting_helper(tool_mat, card, foo, self.materials_owed)
                    checked_names.add(card.name)
                    # print(f'   --- Possible = {possible}')
                    if possible:
                        action = self.name_action_mapping[card.name] + 29
                        legal_actions[action] = 1

        elif self.game_state == 3: # choosing tnt cards to keep
            valid_piles = []
            for i in range(5):
                pile = self.resource_piles[i]
                if (pile.size() > 0) and (pile.peek().type == 'resource'):
                    valid_piles.append(i)
            while len(valid_piles) > 1:
                x = valid_piles.pop(0)
                for y in valid_piles:
                    if x == 0:
                        legal_actions[46 + y] = 1
                    elif x == 1:
                        legal_actions[49 + y] = 1
                    elif x == 2:
                        legal_actions[51 + y] = 1
                    elif x == 3:
                        legal_actions[52 + y] = 1

        elif self.game_state == 4: # what to do during a creeper attack
            if p.resources.size() == 0:
                legal_actions[76] = 1
            else:
                for card in p.resources.cards:
                    action = self.name_action_mapping[card.name] + 57
                    legal_actions[action] = 1
                if p.num_usable('sword') > 0:
                    legal_actions[75] = 1
        
        return legal_actions    

    @property
    def current_player(self):
        return self.players[self.current_player_num]

    def redeal_resources(self):
        """Checks if any resource pile is empty and redeals 15 cards to any that are. Before doing so,
        it will shuffle the discard pile and clear the certain_discarded list."""
        for pile in self.resource_piles:
            if (pile.size() == 0) and (self.discard.size() > 0):
                self.certain_discarded.clear()
                self.discard.shuffle()
                cards = self.discard.draw(15)
                pile.add(cards)
    
    def update_game_state(self, p: Player):
        """If there is a creeper currently revealed, sets the state to 4. Otherwise, if the
        given player 'p' has no actions, sets the state to 0. Otherwise, the state is set to 1."""
        if self.top_resource_count()[1] > 0:
            self.game_state = 4
            self.creeper_player_start = self.current_player_num
            self.num_creepers = self.top_resource_count()[1]
            
        elif p.actions == 0:
            if (p.num_usable('hoe') > 0) or (p.num_usable('shovel') > 0) or (p.num_usable('pickaxe') > 0):
                self.game_state = 1
            else:
                # pass the turn to the next player
                self.current_player.actions = 2
                self.game_state = 0
                self.current_player_num = (self.current_player_num + 1) % self.n_players
                self.turns_taken += 1
                # check if the next player has had their 2 actions taken away by shovels already
                if self.current_player.actions == 0:
                    self.game_state = 1
        else:
            self.game_state = 0

    def player_uses_tool(self, p: Player, action: int):
        """Makes 'p' use up a tool for the corresponding action number and performs
        the associated action. Updates the game state for pickaxes and hoes accordingly."""
        if action == 77 or action == 78:
            target_i = (self.current_player_num + action - 76) % self.n_players
            self.players[target_i].actions -= 1
            p.use_tool('shovel')
        elif action == 79:
            p.actions += 1
            p.use_tool('pickaxe')
        elif action == 80:
            discarded_cards = []
            for pile in self.resource_piles:
                if pile.size() > 0:
                    discarded_cards.append(pile.draw_top())
            self.discard.add(discarded_cards)
            self.certain_discarded += discarded_cards
            p.use_tool('hoe')
            self.redeal_resources()
        else:
            raise Exception(f'Invalid action ({action}) for player_uses_tool!')
        self.update_game_state(p)
    
    def add_card_to_craft(self, p: Player, action: int):
        """Use up the given card from the player's hand in order for them to craft the current
        recipe specified in the game env's self variables. Subtracts the card's value from the
        remaining materials needed and discards it afterwards.

        NOTE: This assumes that the 'action' input is a valid input that could be used to craft
        the recipe, checked already and given as a valid move.
        Any wild cards given here are used for the material of the tool itself first, then for wood."""
        # find what card is being used up
        card_name = self.resource_contents[action - 29]['info']['name']
        card_to_use = p.resources.pick(card_name)
        assert card_to_use != None, 'Card to use in crafting is not in hand!'
        
        card_mat = card_to_use.material
        # if a wildcard is chosen to be used, it is assumed that it is first
        # being used for the specific material of the tool, not for wood, and
        # that the player has enough remaining cards to finish crafting
        if card_mat == 'wild':
            if self.materials_owed[self.recipe_to_craft.material] > 0:
                card_mat = self.recipe_to_craft.material
            else:
                card_mat = 'wood'

        self.materials_owed[card_mat] -= card_to_use.amount
        self.discard.add([card_to_use])
        self.certain_discarded.append(card_to_use)
    
    def craft_tool(self, p, reward):
        """A player has submitted enough resources and we can craft a card for them.
        Also checks if this causes the current player to win the game, in which case it
        sets 'done' to True and returns the reward for the players. (Returns the current
        reward regardless)."""
        done = False
        if self.pile_i == 4:
            p.reserve = None
        else:
            self.recipe_to_craft = self.craft_piles[self.pile_i].draw_top()
        p.tools.add([self.recipe_to_craft])

        p.actions -= 1
        self.update_game_state(p)

        if p.score > 19: # this could be the winning move!
            done = True
            reward = self.score_game()
        return reward, done
    
    def top_resource_count(self):
        """Returns a list of 3 integers. The meaning of each index is:

        0 = number of piles that have a resource card on top
        
        1 = number of piles with a creeper on top
        
        2 = number of piles with tnt on top."""
        ans = [0, 0, 0]
        for i in range(5):
            pile = self.resource_piles[i]
            if pile.size() > 0:
                top_type = pile.peek().type
                if top_type == 'creeper':
                    ans[1] += 1
                elif top_type == 'tnt':
                    ans[2] += 1
                else:
                    ans[0] += 1
        return ans

    def mine_tnt(self, p: Player, action = 0, piles = None):
        """Called when a player chooses to mine a TNT card. If they then have a choice on which
        cards to take, they are simply put in state 3. When they then pick their action, this
        function converts the action and performs it on the current piles.
        
        If a player will not have a choice because there are other tnt cards, their only action
        is played out instantly, assuming there are less than 3 cards to choose from."""
        if (self.game_state == 0) and (self.top_resource_count()[0] > 2):
            self.game_state = 3
            return
        elif action > 0:
            piles = [0,4]
            if action == 56:
                piles[0] = 3
            elif action in (54,55):
                piles[0] = 2
            elif action > 50:
                piles[0] = 1
            if action == 47:
                piles[1] = 1
            elif action in (48,51):
                piles[1] = 2
            elif action in (49,52,54):
                piles[1] = 3
        else:
            piles = []
            for i in range(5):
                pile = self.resource_piles[i]
                if (pile.size() > 0) and (pile.peek().type == 'resource'):
                    piles.append(i)
        for i in range(5):
            pile = self.resource_piles[i]
            if pile.size() > 0:
                card_drawn = self.resource_piles[i].draw_top()
                if i in piles:
                    p.resources.add([card_drawn])
                else:
                    self.discard.add([card_drawn])
                    self.certain_discarded.append(card_drawn)
        p.actions -= 1
        self.redeal_resources()
        self.update_game_state(p)

    def creeper_helper(self):
        """When resolving a creeper attack, this function moves the current player num to
        the next player. If we have returned to the first player (whose turn it really still is),
        it discards the creepers and updates the game state. Otherwise, it resets the creeper cards owed."""
        self.current_player_num = (self.current_player_num + 1) % self.n_players
        # if we have made it back to the start, we are done
        if self.current_player_num == self.creeper_player_start:
            # get rid of the creepers on top
            for pile in self.resource_piles:
                if (pile.size() > 0) and (pile.peek().type == 'creeper'):
                    c = pile.draw_top()
                    self.discard.add([c])
                    self.certain_discarded.append(c)
            self.redeal_resources()
            self.update_game_state(self.current_player)
        else:
            self.num_creepers = self.top_resource_count()[1]

    def creeper_attack(self, p: Player, action: int):
        """Handles the action chosen by the current player for the current creeper attack. Afterwards,
        moves on to the next player if enough cards were discarded. When a full cycle is completed,
        the creepers are discarded and the game state is updated accordingly."""
        if action == 76:
            self.creeper_helper()
            return

        elif action == 75:
            p.use_tool('sword')
        else:
            card_name = self.resource_contents[action - 57]['info']['name']
            card_to_use = p.resources.pick(card_name)
            assert card_to_use != None, 'Card to discard for creeper is not in hand!'

            self.discard.add([card_to_use])
            self.certain_discarded.append(card_to_use)
        
        self.num_creepers -= 1
        # has this player discarded enough cards or do they have no cards left?
        if (self.num_creepers == 0) or (p.resources.size() == 0):
            # move to next player if so
            self.creeper_helper()
        

    def score_game(self):
        reward = [0.0] * self.n_players
        scores = [p.score for p in self.players]
        best_score = max(scores)
        worst_score = min(scores)
        winners = []
        losers = []
        for i, s in enumerate(scores):
            if s == best_score:
                winners.append(i)
            if s == worst_score:
                losers.append(i)

        for w in winners:
            reward[w] += 1.0 / len(winners)
        for l in losers:
            reward[l] -= 1.0 / len(losers)

        return reward

    # Gamestates:
    # 0 - Normal Turn, player has actions left to spend
    # 1 - Current player has no actions, but the option to use a tool or not
    # 2 - Crafting an item, must pick valid cards to use up from player Resources
    # 3 - TNT, Picking out the 2 top cards to keep
    # 4 - Discarding cards from a creeper attack
    def step(self, action: int):
        reward = [0] * self.n_players
        done = False
        p = self.current_player
        # check move legality
        if self.legal_actions[action] == 0:
            reward = [1.0/(self.n_players-1)] * self.n_players
            reward[self.current_player_num] = -1
            done = True
        
        # Normal turn, player has actions left to spend
        elif self.game_state == 0:
            # draw from Resource pile
            if action in range(0,5):
                top_card = self.resource_piles[action].peek()
                if top_card.type == 'tnt':
                    self.mine_tnt(p)
                else:
                    card_drawn = self.resource_piles[action].draw_top()
                    p.resources.add([card_drawn])
                    p.actions -= 1
                    self.redeal_resources()
                    self.update_game_state(p)

            # reserve one of the recipes
            elif action in range(5,9):
                card_drawn = self.craft_piles[action - 5].draw_top()
                p.reserve = card_drawn
                p.actions -= 1
                self.update_game_state(p)

            # begin crafting an item
            elif action in range(9,29):
                self.game_state = 2
                # save the item to craft
                num_axes_to_use, self.pile_i = divmod(action - 9, 5)
                for i in range(num_axes_to_use):
                    p.use_tool('axe')
                if self.pile_i == 4:
                    self.recipe_to_craft = p.reserve
                else:
                    self.recipe_to_craft = self.craft_piles[self.pile_i].peek()
                
                self.materials_owed = self.recipe_to_craft.recipe.copy()
                self.materials_owed['wood'] -= (2 * num_axes_to_use)
                # the axes could have already afforded a wooden tool
                if all([x < 1 for x in self.materials_owed.values()]):
                    reward, done = self.craft_tool(p, reward) # craft_tool handles actions/gamestates

            # use a crafted tool's ability
            elif action in range(77,81):
                self.player_uses_tool(p,action)
            else:
                raise Exception(f'Invalid action ({action}) for state 0!')
        
        elif self.game_state == 1:
            # use a crafted tool's ability
            if action in range(77,81):
                self.player_uses_tool(p,action)
            # use no tools
            elif action == 81:
                # pass the turn to the next player
                self.current_player.actions = 2
                self.game_state = 0
                self.current_player_num = (self.current_player_num + 1) % self.n_players
                self.turns_taken += 1
                # check if the next player has had their 2 actions taken away by shovels already
                if self.current_player.actions == 0:
                    self.game_state = 1
            else:
                raise Exception(f'Invalid action ({action}) for state 1!')
        
        elif self.game_state == 2: # Picking an item to submit to craft
            self.add_card_to_craft(p, action)
            # check if enough materials were added to craft
            if all([x < 1 for x in self.materials_owed.values()]):
                reward, done = self.craft_tool(p, reward) # craft_tool handles actions/gamestates
        
        elif self.game_state == 3: # Picking other cards to take from mining TNT
            self.mine_tnt(p, action)

        elif self.game_state == 4: # Creeper attack!
            self.creeper_attack(p,action)

        self.done = done

        return self.observation, reward, done, {}


    def reset(self):
        self.turns_taken = 0
        self.discard = Deck([])
        self.resource_piles = []
        self.craft_piles = []
        self.certain_discarded = []
        resource_cards_deck = Deck(self.resource_contents)
        for i in range(5):
            cards = resource_cards_deck.draw(15)
            p = Pile()
            p.add(cards)
            self.resource_piles.append(p)
        craft_cards_deck = Deck(self.craft_contents)
        for i in range(4):
            cards = craft_cards_deck.draw(6)
            p = Pile()
            p.add(cards)
            self.craft_piles.append(p)
        self.leftover_recipe = craft_cards_deck.draw(1)[0]

        self.players = []
        player_id = 1
        for p in range(self.n_players):
            self.players.append(Player(str(player_id)))
            player_id += 1

        self.current_player_num = 0
        self.update_game_state(self.current_player)
        self.done = False
        logger.debug(f'\n\n---- NEW GAME ----')
        logger.debug(f'Leftover Craft Card: {self.leftover_recipe.symbol}')
        return self.observation

    def render(self, mode='human', close=False):
        
        if close:
            return

        if not self.done:
            logger.debug(f'\n\n\t\t\t-------TURN {self.turns_taken + 1}-----------')
            logger.debug(f"\t\t\tPlayer {self.current_player.id} currently has control:")
            if self.game_state == 0:
                logger.debug(f'\t\t ⚡ - They have {self.current_player.actions} action(s) to spend.')
            if self.game_state == 1:
                logger.debug(f'\t\t 🕳️ - They have no actions to spend, but may use tool powers if they can.')
            if self.game_state == 2:
                logger.debug(f'\t\t 🛠️ - They are crafting a {self.recipe_to_craft.material} {self.recipe_to_craft.tool}.')
                logger.debug(f'    - Resources owed: {self.materials_owed}')
            if self.game_state == 3:
                logger.debug(f'\t\t 🧨 - They must choose which cards to take using the TNT.')
            if self.game_state == 4:
                logger.debug('\t\t 🟩💥  ----- CREEPER ATTACK -----  💥🟩')
                logger.debug(f'- They must discard {self.num_creepers} more card(s) if they can or use a sword.')
        else:
            logger.debug(f'\n\n\t\t\t-------FINAL POSITION-----------')

        foo = [self.current_player] if (self.game_state > 1) else self.players
        for p in foo:
            logger.debug(f'\n-- Player {p.id}\'s position ({p.actions} actions)')
            logger.debug('--- 💎 Resource Cards ---')
            if p.resources.size() > 0:
                out = '\n'.join([card.symbol for card in sorted(p.resources.cards, key=lambda x: x.id)])
                logger.debug(out)
            else:
                logger.debug('❌ Empty')
            logger.debug('\n--- ⚒️ Tool Cards --- ')
            if p.tools.size() > 0:
                out = '\n'.join([card.symbol for card in sorted(p.tools.cards, key=lambda x: x.id)])
                logger.debug(out)
            else:
                logger.debug('❌ Empty')
            if p.reserve != None:
                out = p.reserve.symbol
            else:
                out = '❌ Empty'
            logger.debug(f'\n🔒 Reserve: {out}')
        
        logger.debug('---Crafting Piles---')
        for i, pile in enumerate(self.craft_piles):
            if pile.size() > 0:
                logger.debug(f'Pile {i}: {pile.peek().symbol}')
            else:
                logger.debug(f'Pile {i}: Empty')
            logger.debug(f'Size: {pile.size()}')

        logger.debug('---Resource Piles---')
        for i, pile in enumerate(self.resource_piles):
            if pile.size() > 0:
                logger.debug(f'Pile {i}: {pile.peek().symbol}')
            else:
                logger.debug(f'Pile {i}: Empty')
            logger.debug(f'Size: {pile.size()}')
        
        if not self.done:
            logger.debug(f'\n\n\t\t\t-------TURN {self.turns_taken + 1}-----------')
            logger.debug(f"\t\t\tPlayer {self.current_player.id} currently has control:")
            if self.game_state == 0:
                logger.debug(f'\t\t ⚡ - They have {self.current_player.actions} action(s) to spend.')
            if self.game_state == 1:
                logger.debug(f'\t\t 🕳️ - They have no actions to spend, but may use tool powers if they can.')
            if self.game_state == 2:
                logger.debug(f'\t\t 🛠️ - They are crafting a {self.recipe_to_craft.material} {self.recipe_to_craft.tool}.')
                logger.debug(f'    - Resources owed: {self.materials_owed}')
            if self.game_state == 3:
                logger.debug(f'\t\t 🧨 - They must choose which cards to take using the TNT.')
            if self.game_state == 4:
                logger.debug('\t\t 🟩💥  ----- CREEPER ATTACK -----  💥🟩')
                logger.debug(f'- They must discard {self.num_creepers} more card(s) if they can or use a sword.')
        else:
            logger.debug(f'\n\n\t\t\t-------FINAL POSITION-----------')

        if self.verbose:
            obs_sparse = [i if o == 1 else (i,o) for i,o in enumerate(self.observation) if o != 0]
            logger.debug(f'\nObservation: \n{obs_sparse}')
            # logger.debug(f'{self.observation[-100:]}')

        if self.done:
            logger.debug(f'\n\nGAME OVER')
        else:
            logger.debug(f'\nLegal actions: {[i for i,o in enumerate(self.legal_actions) if o != 0]}')
        
        logger.debug(f'\n')

        for p in self.players:
            logger.debug(f'Player {p.id} points: {p.score}')


    def rules_move(self):
        raise Exception('Rules based agent is not yet implemented for Minecraft Card Game!')
