import random
import numpy as np
# from .classes import *
from classes import *

# python -m tensorboard.main --logdir="C:\Users\tyler\Desktop\Desktop Work\SIMPLE\app\logs"

# notes:
# - os (optimizer stepsize) I think is the 'learning rate' parameter
#        - Should be decreased linearly over training to 0 or very small
#        - Some papers have it as small as 1e-6 at the end
# start (base), avg games around 150 actions per player
# avg score of +9 ~ 60% win rate? "wins" seem to give +23 avg, losses half that, negated (-11 ish)
# docker-compose exec app mpirun -np 2 python3 train.py -e root3pdomACE -ne 12 -t 9 -ent 0.01 -os 0.0003


class RootGame:
    PHASE_SETUP_MARQUISE = 0
    PHASE_SETUP_EYRIE = 1
    PHASE_SETUP_VAGABOND = 2
    PHASE_BIRDSONG_MARQUISE = 3
    PHASE_BIRDSONG_EYRIE = 4
    PHASE_BIRDSONG_ALLIANCE = 5
    PHASE_BIRDSONG_VAGABOND = 6
    PHASE_DAYLIGHT_MARQUISE = 7
    PHASE_DAYLIGHT_EYRIE = 8
    PHASE_DAYLIGHT_ALLIANCE = 9
    PHASE_DAYLIGHT_VAGABOND = 10
    PHASE_EVENING_MARQUISE = 11
    PHASE_EVENING_EYRIE = 12
    PHASE_EVENING_ALLIANCE = 13
    PHASE_EVENING_VAGABOND = 14

    def __init__(self, board_composition:list, deck_composition:list):
        self.n_players = N_PLAYERS
        self.board = Board(board_composition)
        self.deck = Deck(deck_composition)
        self.quest_deck = Deck(QUEST_DECK_COMP)

        # self.reset_general_items()
        # self.reset_for_marquise()
        # self.reset_for_eyrie()

    def to_play(self):
        return self.current_player
    
    def index_to_id(self,x):
        "Converts 0 to 1, 1 to -1"
        return 1 if (x == 0) else -1
    
    def reset_general_items(self):
        self.num_actions_played = 0
        self.acting_player = 0
        self.outside_turn_this_action = 0
        self.legal_actions_to_get = None
        self.saved_battle_actions = []
        self.saved_battle_player = None
        self.alliance_interrupt_player = None
        self.points_scored_this_action = [0] * N_PLAYERS
        self.current_player = 0
        self.players = [Marquise(0), Eyrie(1), Alliance(2), Vagabond(3)]
        self.victory_points = [0] * N_PLAYERS

        self.dominance_win = False
        self.active_dominances = [None] * N_PLAYERS
        self.coalition_partner = [0] * N_PLAYERS
        self.available_dominances = [0,0,0,0]
        self.available_dom_card_objs = [None,None,None,None]
        self.active_quests = []
        self.ruin_items = {i:None for i in range(12) if (self.board.clearings[i].num_ruins > 0)}

        self.phase = self.PHASE_SETUP_MARQUISE
        self.phase_steps = 0

        self.field_hospitals = []
        self.outrage_offender = None
        self.outrage_suits = []

        self.persistent_used_this_turn = set()
        self.remaining_craft_power = [0]
        self.board.reset()
        self.deck.reset()
        self.battle = Battle(-1,-1,-1)
        self.battle.stage = Battle.STAGE_DONE
        self.discard_pile = []
        self.discard_array = np.zeros((42,3))
        self.available_items = {
            ITEM_COINS: 2,
            ITEM_BOOT: 2,
            ITEM_BAG: 2,
            ITEM_TEA: 2,
            ITEM_SWORD: 2,
            ITEM_CROSSBOW: 1,
            ITEM_HAMMER: 1
        }
        # random turn order
        self.turn_order = [i for i in range(N_PLAYERS)]
        random.shuffle(self.turn_order)
        # make a dict to easily find which PIND should be
        # transitioned to at the end of a turn
        self.next_player_index = {}
        for i in range(N_PLAYERS):
            place = self.turn_order.index(i)
            next_index = (place + 1) % N_PLAYERS
            self.next_player_index[i] = self.turn_order[next_index]
        
        self.public_history = [None] * TURN_MEMORY
        self.private_history = [None] * TURN_MEMORY
        self.turn_log = TurnLog()

        self.draw_cards(PIND_MARQUISE,3)
        self.draw_cards(PIND_EYRIE,3)
        self.draw_cards(PIND_ALLIANCE,3)
        self.draw_cards(PIND_VAGABOND,3)

        self.marquise_seen_hands = {
            PIND_EYRIE: [np.full((42,3),-1) for _ in range(TURN_MEMORY)],
            PIND_ALLIANCE: [np.full((42,3),-1) for _ in range(TURN_MEMORY)],
            PIND_VAGABOND: [np.full((42,3),-1) for _ in range(TURN_MEMORY)]
        }
        self.eyrie_seen_hands = {
            PIND_MARQUISE: [np.full((42,3),-1) for _ in range(TURN_MEMORY)],
            PIND_ALLIANCE: [np.full((42,3),-1) for _ in range(TURN_MEMORY)],
            PIND_VAGABOND: [np.full((42,3),-1) for _ in range(TURN_MEMORY)]
        }
        self.alliance_seen_hands = {
            PIND_EYRIE: [np.full((42,3),-1) for _ in range(TURN_MEMORY)],
            PIND_MARQUISE: [np.full((42,3),-1) for _ in range(TURN_MEMORY)],
            PIND_VAGABOND: [np.full((42,3),-1) for _ in range(TURN_MEMORY)]
        }
        self.vagabond_seen_hands = {
            PIND_EYRIE: [np.full((42,3),-1) for _ in range(TURN_MEMORY)],
            PIND_MARQUISE: [np.full((42,3),-1) for _ in range(TURN_MEMORY)],
            PIND_ALLIANCE: [np.full((42,3),-1) for _ in range(TURN_MEMORY)]
        }

    def reset_for_marquise(self):
        # print(self.board)
        self.wood_placement_started = False
        self.starting_build_spots = [0]
        self.available_wood_spots = [0]
        self.available_recruiters = [0]
        self.marquise_actions = 3
        self.marquise_moves = 2
        self.recruited_this_turn = 0
        self.remaining_wood_cost = 0

        self.persistent_used_this_turn = set()
        self.remaining_craft_power = [0]
        for k in self.marquise_seen_hands.keys():
            self.marquise_seen_hands[k].insert(0,np.full((42,3),-1))
            self.marquise_seen_hands[k].pop()
    def reset_for_eyrie(self):
        # print(self.board)
        self.eyrie_cards_added = 0
        self.eyrie_bird_added = 0
        self.remaining_decree = {
            DECREE_RECRUIT: [0,0,0,0],
            DECREE_MOVE: [0,0,0,0],
            DECREE_BATTLE: [0,0,0,0],
            DECREE_BUILD: [0,0,0,0]
        }
        self.persistent_used_this_turn = set()
        self.remaining_craft_power = [0]
        for k in self.eyrie_seen_hands.keys():
            self.eyrie_seen_hands[k].insert(0,np.full((42,3),-1))
            self.eyrie_seen_hands[k].pop()
    def reset_for_alliance(self):
        # print(self.board)
        self.remaining_supporter_cost = 0
        self.required_supporter_suit = None
        self.evening_actions_left = 0
        self.alliance_action_clearing = None
        self.persistent_used_this_turn = set()
        self.remaining_craft_power = [0]
        for k in self.alliance_seen_hands.keys():
            self.alliance_seen_hands[k].insert(0,np.full((42,3),-1))
            self.alliance_seen_hands[k].pop()
    def reset_for_vagabond(self):
        # print(self.board)
        self.refreshes_left = -1
        self.aid_target = None
        self.aids_this_turn = [0,0,0]
        self.persistent_used_this_turn = set()
        self.remaining_craft_power = [0]
        self.vagabond_battle_target = None
        self.vagabond_battle_ally = None
        for k in self.vagabond_seen_hands.keys():
            self.vagabond_seen_hands[k].insert(0,np.full((42,3),-1))
            self.vagabond_seen_hands[k].pop()

    def reset(self):
        self.reset_general_items()
        self.reset_for_marquise()
        self.reset_for_eyrie()
        self.reset_for_alliance()
        self.reset_for_vagabond()
        return self.get_observation()

    def save_to_history(self):
        """
        Saves information about the current state of the game
        and the info in the current Turn Log to the first index in
        the history.
        """
        # save all info in turn log / end of turn info
        # logger.debug(f"Current TurnLog: {vars(self.turn_log)}")
        t = self.turn_log
        aplayer = self.players[PIND_ALLIANCE]
        full_turn = np.zeros(6)
        full_turn[self.acting_player] = 1
        full_turn[3] = int(t.current_turn['dp_reset'])
        # alliance board info
        full_turn[4] = min(len(aplayer.supporters) / 25, 1)
        full_turn[5] = aplayer.num_officers / 10
        full_turn = np.append(full_turn,t.current_turn['alliance_supp_payments'])
        # save eyrie decree
        eplayer = self.players[PIND_EYRIE]
        foo = np.zeros((4,43,3))
        for dec_i in range(4):
            for c in eplayer.decree[dec_i]:
                cid = c.id
                if foo[dec_i][cid][0] == 1:
                    foo[dec_i][cid][0] = 0
                    foo[dec_i][cid][1] = 1
                elif foo[dec_i][cid][1] == 1:
                    foo[dec_i][cid][1] = 0
                    foo[dec_i][cid][2] = 1
                else:
                    foo[dec_i][cid][0] = 1
        full_turn = np.append(full_turn,foo)
        # save leader
        foo = np.zeros(4)
        foo[eplayer.chosen_leader_index] = 1
        full_turn = np.append(full_turn,foo)
        # save current discard pile and available dominances
        full_turn = np.append(full_turn,self.discard_array)
        full_turn = np.append(full_turn,np.array(self.available_dominances))
        # save info stored for each player
        for player_i in range(N_PLAYERS):
            p_info = t.current_turn[player_i]
            max_warriors = [25,20,10][player_i]
            foo = np.zeros(3)
            foo[0] = p_info['hand_size_change'] / 8
            foo[1] = p_info['warrior_supply_change'] / max_warriors
            foo[2] = min(p_info['point_change'] / 15, 1)
            for att in ('cards_lost','cards_gained','persistent_used','cards_crafted'):
                foo = np.append(foo,p_info[att])
            full_turn = np.append(full_turn,foo)
            # static / end of turn info
            foo = np.zeros(31)
            bar = np.zeros(4)
            if self.active_dominances[player_i] is None:
                foo[min(30,self.victory_points[player_i])] = 1
            else:
                bar[self.active_dominances[player_i].suit] = 1
            foo = np.append(foo,bar)
            full_turn = np.append(full_turn,foo)
        # save info stored for each clearing
        for clearing_i in range(12):
            c = self.board.clearings[clearing_i]
            c_info = t.current_turn[f'c{clearing_i}']
            foo = c_info['battles']
            end_warriors = np.zeros(3)
            end_buildings = np.zeros(7)
            end_tokens = np.zeros(3)
            for player_i in range(N_PLAYERS):
                max_warriors = [25,20,10][player_i]
                foo = np.append(foo,c_info[player_i]['warrior_change'] / max_warriors)
                foo = np.append(foo,c_info[player_i]['buildings_change'] / 3)
                bar = c_info[player_i]['tokens_change'].copy()
                if player_i == PIND_MARQUISE:
                    bar[TIND_WOOD] /= 8
                foo = np.append(foo,bar)
                end_warriors[player_i] = c.warriors[player_i] * 0.75 / max_warriors + 0.25 * int(c.warriors[player_i] > 0)
                if player_i == PIND_MARQUISE:
                    for bind in range(3):
                        num_bind = c.get_num_buildings(PIND_MARQUISE,bind)
                        if num_bind > 0:
                            end_buildings[bind] = num_bind * 0.25 + 0.25
                    end_tokens[0] = c.get_num_tokens(PIND_MARQUISE,TIND_KEEP)
                    end_tokens[1] = c.get_num_tokens(PIND_MARQUISE,TIND_WOOD) / 8
                elif player_i == PIND_EYRIE:
                    end_buildings[3] = c.get_num_buildings(PIND_EYRIE)
                else: # Alliance
                    for bind in range(3):
                        end_buildings[bind + 4] = c.get_num_buildings(PIND_ALLIANCE,bind)
                    end_tokens[2] = c.get_num_tokens(PIND_ALLIANCE,TIND_SYMPATHY)
            full_turn = np.append(full_turn,foo)
            full_turn = np.append(full_turn,end_warriors)
            full_turn = np.append(full_turn,end_buildings)
            full_turn = np.append(full_turn,end_tokens)

        # logger.debug(f"save to hist fullturn: {len(full_turn)}")

        # for i,val in enumerate(full_turn):
        #     if val != 0:
        #         logger.debug(f"{i}: {val}")

        # save this turn in the history
        marq_priv = np.append(t.current_turn['marq_supp_additions'], np.full((42,3),-1))
        eyrie_priv = np.append(np.full((42,3),-1), t.current_turn['eyrie_supp_additions'])
        all_priv = np.append(t.current_turn['marq_supp_additions'], t.current_turn['eyrie_supp_additions'])
        
        # logger.debug(f"{[marq_priv,eyrie_priv,all_priv]}")
        # logger.debug(f"\n{self.turn_log.get_array(self.acting_player)}")

        self.private_history.insert(0,[marq_priv,eyrie_priv,all_priv])
        self.private_history.pop()
        self.public_history.insert(0,full_turn)
        self.public_history.pop()
        # reset the TurnLog for the next one
        self.turn_log.reset_current_turn()
    
    def get_history(self,viewing_player:int):
        """
        Returns an array of the entire recorded history, combined with
        the private history viewable only by the given player.
        """
        ret = np.zeros(0)
        for i in range(TURN_MEMORY):
            if self.public_history[i] is not None:
                ret = np.append(ret,self.public_history[i])
                ret = np.append(ret,self.private_history[i][viewing_player])
            else:
                ret = np.append(ret,np.full(2597,0))
        return ret

    def step(self, action):
        actions_to_return = []
        self.points_scored_this_action = [0] * N_PLAYERS
        self.acting_player = self.current_player
        if self.phase in {self.PHASE_SETUP_EYRIE,self.PHASE_BIRDSONG_EYRIE,self.PHASE_DAYLIGHT_EYRIE,self.PHASE_EVENING_EYRIE}:
            self.outside_turn_this_action = PIND_EYRIE
        elif self.phase in {self.PHASE_SETUP_MARQUISE,self.PHASE_BIRDSONG_MARQUISE,self.PHASE_DAYLIGHT_MARQUISE,self.PHASE_EVENING_MARQUISE}:
            self.outside_turn_this_action = PIND_MARQUISE
        elif self.phase in {self.PHASE_BIRDSONG_ALLIANCE,self.PHASE_DAYLIGHT_ALLIANCE,self.PHASE_EVENING_ALLIANCE}:
            self.outside_turn_this_action = PIND_ALLIANCE
        else:
            self.outside_turn_this_action = PIND_VAGABOND

        aplayer = self.players[PIND_ALLIANCE]
        resolving = False
        if self.outside_turn_this_action == PIND_MARQUISE:
            # first, resolve field hospitals
            if len(self.field_hospitals) > 0:
                resolving = True
                actions_to_return = self.handle_field_hospitals(aplayer,action)
            # then resolve outrage
            elif self.outrage_offender is not None:
                resolving = True
                actions_to_return = self.handle_outrage(aplayer,action)
            # check for alliance cutting down supporters after base destroyed
            elif (sum([aplayer.get_num_buildings_on_track(bid) for bid in range(3)]) == 3 and
                    len(aplayer.supporters) > 5):
                resolving = True
                actions_to_return = self.handle_discard_supporter(aplayer,action)
        else:
            # first, check for outrage payment
            if self.outrage_offender is not None:
                resolving = True
                actions_to_return = self.handle_outrage(aplayer,action)
            # check for alliance cutting down supporters after base destroyed
            elif (sum([aplayer.get_num_buildings_on_track(bid) for bid in range(3)]) == 3 and
                    len(aplayer.supporters) > 5):
                resolving = True
                actions_to_return = self.handle_discard_supporter(aplayer,action)
            # check if we are resolving field hospitals
            elif len(self.field_hospitals) > 0:
                resolving = True
                actions_to_return = self.handle_field_hospitals(aplayer,action)

        if not resolving and self.battle.stage == Battle.STAGE_DONE:
            # logger.debug("ACTION GIVEN,NO BATTLE")
            self.resolve_action(action)
            battle = False
            # battle could have been started
            if self.battle.stage is None:
                # logger.debug("BATTLE WAS STARTED")
                # kick start the battle to see if a choice must be made
                # during it. Use the given action, although nothing will
                # really happen yet. Either a choice will be made or
                # the battle will finish by itself
                self.saved_battle_actions = self.resolve_battle_action(action)
                self.saved_battle_player = self.current_player
                # logger.debug(f"BATTLE ACTIONS RETURNED: {self.saved_battle_actions}")
                battle = True
            if self.outside_turn_this_action == PIND_MARQUISE:
                # check for field hospitals first (best for marquise)
                actions_to_return = self.field_hospitals_check()
                if not bool(actions_to_return) and self.outrage_offender is not None:
                    # logger.debug("OUTRAGE CHECK")
                    actions_to_return = self.outrage_step_check()
                if not bool(actions_to_return):
                    # logger.debug("NO BASE CHECK")
                    actions_to_return = self.alliance_no_base_check(aplayer)
                if battle and not bool(actions_to_return):
                    # logger.debug("BAT,NO ACTIONS")
                    actions_to_return = self.saved_battle_actions
            else:
                # check for field hospitals last (worst for Marquise)
                if self.outrage_offender is not None:
                    # logger.debug("OUTRAGE CHECK")
                    actions_to_return = self.outrage_step_check()
                if not bool(actions_to_return):
                    # logger.debug("NO BASE CHECK")
                    actions_to_return = self.alliance_no_base_check(aplayer)
                if not bool(actions_to_return):
                    actions_to_return = self.field_hospitals_check()
                if battle and not bool(actions_to_return):
                    # logger.debug("BAT,NO ACTIONS")
                    actions_to_return = self.saved_battle_actions
        elif not resolving: # we are in a battle
            self.saved_battle_actions = self.resolve_battle_action(action)
            self.saved_battle_player = self.current_player
            if self.outside_turn_this_action == PIND_MARQUISE:
                # check for field hospitals first (best for marquise)
                actions_to_return = self.field_hospitals_check()
                if not bool(actions_to_return) and self.outrage_offender is not None:
                    # logger.debug("OUTRAGE CHECK")
                    actions_to_return = self.outrage_step_check()
                if not bool(actions_to_return):
                    # logger.debug("NO BASE CHECK")
                    actions_to_return = self.alliance_no_base_check(aplayer)
                if not bool(actions_to_return):
                    # logger.debug("BAT,NO ACTIONS")
                    actions_to_return = self.saved_battle_actions
            else:
                # check for field hospitals last (worst for Marquise)
                if self.outrage_offender is not None:
                    # logger.debug("OUTRAGE CHECK")
                    actions_to_return = self.outrage_step_check()
                if not bool(actions_to_return):
                    # logger.debug("NO BASE CHECK")
                    actions_to_return = self.alliance_no_base_check(aplayer)
                if not bool(actions_to_return):
                    actions_to_return = self.field_hospitals_check()
                if not bool(actions_to_return):
                    # logger.debug("BAT,NO ACTIONS")
                    actions_to_return = self.saved_battle_actions

        if bool(actions_to_return):
            self.legal_actions_to_get = actions_to_return
        else:
            self.legal_actions_to_get = self.advance_game()

        # at this point, if a player has won by dominance, this flag
        # will be set to true in a step of the advance_game function
        reward = self.points_scored_this_action
        if self.dominance_win:
            done = True
            for i in range(N_PLAYERS):
                if i == self.current_player:
                    reward[i] += 8 * WIN_SCALAR
                else:
                    reward[i] -= (8 / (N_PLAYERS - 1)) * WIN_SCALAR
        else:
            done = (max(self.victory_points) >= 30) and (self.battle.stage == Battle.STAGE_DONE)
            if done:
                winlist = self.get_winner_points()
                for i,val in enumerate(winlist):
                    if val == 1:
                        reward[i] += 5 * WIN_SCALAR
                    else:
                        reward[i] -= (5 / (N_PLAYERS - 1)) * WIN_SCALAR
        
        self.num_actions_played += 1
        if (not done) and (self.num_actions_played >= 750):
            done = True
            for i in range(N_PLAYERS):
                reward[i] -= 10

        return self.get_observation(), reward, done

    def field_hospitals_check(self):
        """
        Checks if there are any field hospitals to deal with for the Marquise.
        If there is one, it checks if the Marquise have a card to pay with in their
        hand. If they have a card to pay, it returns a list of valid discard actions
        to use. Otherwise, it removes the FH from the list.

        With no field hospitals, returns an empty list.
        """
        while len(self.field_hospitals) > 0:
            suit = self.field_hospitals[-1][1]
            if self.players[PIND_MARQUISE].has_suit_in_hand(suit):
                self.current_player = PIND_MARQUISE
                return [AID_GENERIC_SKIP] + [c.id+AID_DISCARD_CARD for c in self.players[PIND_MARQUISE].hand if (c.suit in {suit,SUIT_BIRD})]
            else:
                self.field_hospitals.pop()
        return []

    def outrage_step_check(self):
        """
        Manages the extra checks for outrage payments before advancing 
        the game. Assumes that self.outrage_offender is not None and that
        there is still an outrage suit in the list to be paid.

        If there is a card to be paid from the offender's hand, then
        it returns a list of the valid Discard ID's and changes the current
        player. 

        If not, then a supporter is drawn randomly and returns an empty list.
        """
        # check if the offender has a card to pay with
        logger.debug(f"Outrage Triggered! {ID_TO_PLAYER[self.outrage_offender]} must pay the Alliance 1 {ID_TO_SUIT[self.outrage_suits[-1]]} Card")
        offender = self.players[self.outrage_offender]
        while len(self.outrage_suits) > 0:
            suit = self.outrage_suits[-1]
            if offender.has_suit_in_hand(suit):
                self.alliance_interrupt_player = self.current_player
                self.current_player = self.outrage_offender
                return [c.id+AID_DISCARD_CARD for c in offender.hand if (c.suit in {suit,SUIT_BIRD})]
            else:
                # cannot pay, so offender shows their hand and
                # alliance draws 1 card to add to supporters
                self.outrage_no_suit_helper(offender)
                self.outrage_suits.pop()
        self.outrage_offender = None
        return []
    
    def alliance_no_base_check(self,aplayer:Alliance):
        """
        Manages the case when the alliance have too many supporters and
        must discard down to 5. Returns a list of Discard supporter AIDs
        if this must be done, or an empty list if not.
        """
        if (sum([aplayer.get_num_buildings_on_track(bid) for bid in range(3)]) == 3 and
                len(aplayer.supporters) > 5 and
                self.outrage_offender is None):
            # The alliance must discard down to 5 supporters
            logger.debug(f"Alliance have no more bases, so they must discard down to 5 supporters!")
            self.alliance_interrupt_player = self.current_player
            self.current_player = PIND_ALLIANCE
            return list({c.id+AID_DISCARD_CARD for c in aplayer.supporters})
        else:
            return []

    def outrage_no_suit_helper(self,offender:Player):
        logger.debug("\tOffender has no cards to give, so they show their hand to the Alliance...")
        target_hand = np.zeros((42,3))
        for c in offender.hand:
            cid = c.id
            if target_hand[cid][0] == 1:
                target_hand[cid][0] = 0
                target_hand[cid][1] = 1
            elif target_hand[cid][1] == 1:
                target_hand[cid][1] = 0
                target_hand[cid][2] = 1
            else:
                target_hand[cid][0] = 1
        self.alliance_seen_hands[self.outrage_offender][0] = target_hand
        
        # alliance draws 1 card for supporters
        if self.deck.size() == 0:
            self.deck.add(self.discard_pile) # includes shuffling
            self.turn_log.discard_pile_was_reset()
            self.discard_pile = []
            self.discard_array = np.zeros((42,3))
        draw = self.deck.draw(1)
        if len(draw) == 0:
            # the deck is empty, and there is no discard pile to refresh it with
            logger.debug(f"--> Cannot draw any more cards, no deck / discard pile!")
            return
        c_drawn = draw[0]
        if self.deck.size() == 0:
            self.deck.add(self.discard_pile) # includes shuffling
            self.turn_log.discard_pile_was_reset()
            self.discard_pile = []
            self.discard_array = np.zeros((42,3))
        logger.debug(f"\tAlliance draws: {c_drawn.name} (added to supporters)")
        aplayer = self.players[PIND_ALLIANCE]
        self.add_to_supporters_check(aplayer,c_drawn)
    
    def add_to_supporters_check(self,aplayer:Alliance,c_to_add:Card):
        """
        Adds the given card to the Alliance player's supporters, but
        will discard it instead if the alliance have all of their bases
        unbuilt and already have 5 supporters.
        """
        if (sum([aplayer.get_num_buildings_on_track(bid) for bid in range(3)]) == 3 and
                len(aplayer.supporters) >= 5):
            # Limit of 5 supporters with no bases built
            logger.debug(f"\t\t-> Cannot add more than five supporters with no bases built.")
            self.turn_log.change_alliance_payments(c_to_add.id)
            self.add_to_discard(c_to_add)
        else:
            # add to the supporter pile
            aplayer.add_to_supporters(c_to_add)

    def handle_outrage(self,aplayer:Alliance,action:int):
        actions_to_return = []
        self.outrage_suits.pop()
        c_to_pay = self.get_card(self.outrage_offender,action-AID_DISCARD_CARD,"hand")
        self.turn_log.change_plr_hand_size(self.outrage_offender,-1)
        self.turn_log.change_alliance_supp_addition(self.outrage_offender,c_to_pay.id)
        self.add_to_supporters_check(aplayer,c_to_pay)

        offender = self.players[self.outrage_offender]
        while len(self.outrage_suits) > 0:
            logger.debug(f"Outrage Triggered! {ID_TO_PLAYER[self.outrage_offender]} must pay the Alliance 1 {ID_TO_SUIT[self.outrage_suits[-1]]} Card")
            suit = self.outrage_suits[-1]
            if offender.has_suit_in_hand(suit):
                return [c.id+AID_DISCARD_CARD for c in offender.hand if (c.suit in {suit,SUIT_BIRD})]
            else:
                # cannot pay, so offender shows their hand and
                # alliance draws 1 card to add to supporters
                self.outrage_no_suit_helper(offender)
                self.outrage_suits.pop()
        if not bool(actions_to_return):
            # we are finished paying for outrage
            self.outrage_offender = None
            self.current_player = self.alliance_interrupt_player
            actions_to_return = self.alliance_no_base_check(aplayer)
        if not bool(actions_to_return):
            actions_to_return = self.field_hospitals_check()
        if not bool(actions_to_return):
            if self.battle.stage != Battle.STAGE_DONE:
                actions_to_return = self.saved_battle_actions
                self.current_player = self.saved_battle_player
            else:
                self.current_player = self.alliance_interrupt_player
        return actions_to_return
    
    def handle_discard_supporter(self,aplayer:Alliance,action:int):
        actions_to_return = []
        self.discard_from_supporters(aplayer,action - AID_DISCARD_CARD)
        if len(aplayer.supporters) > 5:
            actions_to_return = list({c.id+AID_DISCARD_CARD for c in aplayer.supporters})
        # we are done discarding
        if not bool(actions_to_return):
            actions_to_return = self.field_hospitals_check()
        if not bool(actions_to_return):
            if self.battle.stage != Battle.STAGE_DONE:
                actions_to_return = self.saved_battle_actions
                self.current_player = self.saved_battle_player
            else:
                self.current_player = self.alliance_interrupt_player
        return actions_to_return
    
    def handle_field_hospitals(self,aplayer:Alliance,action:int):
        # the action is say if the marquise are discarding a card or not
        actions_to_return = []
        foo = self.field_hospitals.pop()
        if action == AID_GENERIC_SKIP:
            logger.debug(f"The Marquise do not use Field Hospitals on the {foo[0]} warriors in the {ID_TO_SUIT[foo[1]]} Clearing")
        else:
            # they are using Field Hospitals
            self.activate_field_hospitals(foo[0],action - AID_DISCARD_CARD)
        while len(self.field_hospitals) > 0:
            suit = self.field_hospitals[-1][1]
            if self.players[PIND_MARQUISE].has_suit_in_hand(suit):
                return [AID_GENERIC_SKIP] + [c.id+AID_DISCARD_CARD for c in self.players[PIND_MARQUISE].hand if (c.suit in {suit,SUIT_BIRD})]
            else:
                self.field_hospitals.pop()
        if not bool(actions_to_return): # no more field hospitals
            if self.outrage_offender is not None:
                actions_to_return = self.outrage_step_check()
            if not bool(actions_to_return):
                actions_to_return = self.alliance_no_base_check(aplayer)
            if not bool(actions_to_return):
                if self.battle.stage != Battle.STAGE_DONE:
                    actions_to_return = self.saved_battle_actions
                    self.current_player = self.saved_battle_player
                else:
                    if self.phase in {self.PHASE_SETUP_EYRIE,self.PHASE_BIRDSONG_EYRIE,self.PHASE_DAYLIGHT_EYRIE,self.PHASE_EVENING_EYRIE}:
                        self.current_player = PIND_EYRIE
                    elif self.phase in {self.PHASE_SETUP_MARQUISE,self.PHASE_BIRDSONG_MARQUISE,self.PHASE_DAYLIGHT_MARQUISE,self.PHASE_EVENING_MARQUISE}:
                        self.current_player = PIND_MARQUISE
                    else:
                        self.current_player = PIND_ALLIANCE
        return actions_to_return


    def get_observation(self):
        ret = np.zeros((12,3))
        for i,c in enumerate(self.board.clearings):
            ret[i][c.suit] = 1
        foo = np.zeros(54)
        if self.deck.size() > 0:
            foo[self.deck.size() - 1] = 1
        ret = np.append(ret,foo)

        ret = np.append(ret,self.discard_array)
        ret = np.append(ret,np.array(self.available_dominances))

        foo = np.zeros((7,2))
        for i,a in self.available_items.items():
            if a > 0:
                foo[i][a - 1] = 1
        ret = np.append(ret,foo)
        
        foo = np.zeros((3,31))
        bar = np.zeros((3,4))
        for i in range(3):
            if self.active_dominances[i] is None:
                foo[i][min(30,self.victory_points[i])] = 1
            else:
                bar[i][self.active_dominances[i].suit] = 1
        foo = np.append(foo,bar)
        ret = np.append(ret,foo)
        
        foo = np.zeros(18)
        foo[self.phase] = 1
        foo[self.phase_steps + 11] = 1
        ret = np.append(ret,foo)

        foo = np.zeros((5,3))
        foo[0][self.outside_turn_this_action] = 1
        foo[1][self.current_player] = 1
        for i in range(3):
            foo[i+2][self.turn_order[i]] = 1
        ret = np.append(ret,foo)

        ret = np.append(ret,self.players[0].get_obs_array())
        ret = np.append(ret,self.players[1].get_obs_array())
        ret = np.append(ret,self.players[2].get_obs_array())

        curr_player = self.players[self.current_player]
        foo = np.zeros((42,3))
        for c in curr_player.hand:
            cid = c.id
            if foo[cid][0] == 1:
                foo[cid][0] = 0
                foo[cid][1] = 1
            elif foo[cid][1] == 1:
                foo[cid][1] = 0
                foo[cid][2] = 1
            else:
                foo[cid][0] = 1
        ret = np.append(ret,foo)

        foo = np.zeros((3,6))
        if len(self.remaining_craft_power) == 3:
            for i,a in enumerate(self.remaining_craft_power):
                if a > 0:
                    try:
                        foo[i][a - 1] = 1
                    except:
                        raise Exception(f'foo is {foo}, i: {i}, a: {a}, power: {self.remaining_craft_power}')
        ret = np.append(ret,foo)
        
        foo = np.zeros((4,3))
        for i,a in enumerate(self.outrage_suits):
            foo[i][a] = 1
        ret = np.append(ret,foo)
        
        foo = np.zeros(11)
        foo.put(list(CID_TO_PERS_INDEX[i] for i in self.persistent_used_this_turn), 1)
        ret = np.append(ret,foo)

        foo = np.zeros((4,25))
        for i,a in enumerate(self.field_hospitals):
            foo[i][a[0] - 1] = 1
        ret = np.append(ret,foo)

        if self.current_player == PIND_MARQUISE or self.outside_turn_this_action == PIND_MARQUISE:
            foo = np.zeros(14)
            if self.marquise_actions > 0:
                foo[self.marquise_actions - 1] = 1
            if self.remaining_wood_cost > 0:
                foo[self.remaining_wood_cost - 1 + 10] = 1
        else:
            foo = np.full(14,-1)
        ret = np.append(ret,foo)
        
        if self.current_player == PIND_EYRIE or self.outside_turn_this_action == PIND_EYRIE:
            foo = np.zeros(2)
            if self.eyrie_cards_added > 0:
                foo[0] = 1
            if self.eyrie_bird_added > 0:
                foo[1] = 1
            bar = np.zeros((4,4,15))
            for dec_i,lst in self.remaining_decree.items():
                for i,a in enumerate(lst):
                    if a > 0:
                        bar[dec_i][i][a - 1] = 1
            foo = np.append(foo,bar)
        else:
            foo = np.full(242,-1)
        ret = np.append(ret,foo)

        if self.current_player == PIND_ALLIANCE or self.outside_turn_this_action == PIND_ALLIANCE:
            foo = np.zeros(14)
            if self.evening_actions_left > 0:
                foo[self.evening_actions_left - 1] = 1
            if self.remaining_supporter_cost > 0:
                foo[self.remaining_supporter_cost - 1 + 10] = 1
            if self.current_player == PIND_ALLIANCE:
                bar = np.zeros((3,30))
                for i in range(3):
                    bar[i][curr_player.supporter_suit_counts[i]] = 1
                ump = np.zeros((42,3))
                for c in curr_player.supporters:
                    cid = c.id
                    if ump[cid][0] == 1:
                        ump[cid][0] = 0
                        ump[cid][1] = 1
                    elif ump[cid][1] == 1:
                        ump[cid][1] = 0
                        ump[cid][2] = 1
                    else:
                        ump[cid][0] = 1
                bar = np.append(bar,ump)
            else:
                bar = np.full(216,-1)
            foo = np.append(foo,bar)
        else:
            foo = np.full(230,-1)
        ret = np.append(ret,foo)
        
        ret = np.append(ret,self.battle.get_obs_array())
        ret = np.append(ret,self.board.get_obs_array())

        ret = np.append(ret,self.turn_log.get_array(self.current_player))

        if self.current_player == PIND_MARQUISE:
            seen = self.marquise_seen_hands[PIND_EYRIE] + self.marquise_seen_hands[PIND_ALLIANCE]
        elif self.current_player == PIND_EYRIE:
            seen = self.eyrie_seen_hands[PIND_MARQUISE] + self.eyrie_seen_hands[PIND_ALLIANCE]
        else:
            seen = self.alliance_seen_hands[PIND_MARQUISE] + self.alliance_seen_hands[PIND_EYRIE]
        
        for hand in seen:
            ret = np.append(ret,hand)

        return np.append(ret,self.get_history(self.current_player))

    def legal_actions(self):
        while self.legal_actions_to_get is None:
            self.legal_actions_to_get = self.advance_game()
        return self.legal_actions_to_get
    
    def get_winner_points(self):
        """
        Assumes that somebody has won (30 pts or more).
        Returns a list with 1 at the index of the winner, and a -1
        at every other index.
        """
        ans = [-1,-1,-1]
        winners = [i for i in range(3) if self.victory_points[i] == max(self.victory_points)]
        if len(winners) == 1:
            ans[winners[0]] = 1
        else:
            for i in winners:
                if i == self.acting_player:
                    ans[i] = 1
                    break
        return ans
        
    def draw_cards(self,player_index:int,amount:int):
        """
        Draws a number of cards from the top of the deck and then
        adds them to a player's hand.

        If the deck runs out, it automatically uses up the
        discard pile to refresh the deck and then continues drawing.
        """
        p = self.players[player_index]
        # check in case the deck is empty and cards have been
        # added to the discard since the last draw attempt
        if self.deck.size() == 0:
            self.deck.add(self.discard_pile) # includes shuffling
            self.turn_log.discard_pile_was_reset()
            self.discard_pile = []
            self.discard_array = np.zeros((42,3))
        while amount:
            draw = self.deck.draw(1)
            if len(draw) == 0:
                # the deck is empty, and there is no discard pile to refresh it with
                logger.debug(f"--> Cannot draw any more cards, no deck / discard pile!")
                return
            c_drawn = draw[0]
            logger.debug(f"\t{ID_TO_PLAYER[player_index]} draws: {c_drawn.name}")
            p.hand.append(c_drawn)
            self.turn_log.change_plr_hand_size(player_index,1)
            if self.deck.size() == 0:
                self.deck.add(self.discard_pile) # includes shuffling
                self.turn_log.discard_pile_was_reset()
                self.discard_pile = []
                self.discard_array = np.zeros((42,3))
            amount -= 1
    
    def get_card(self,player_index:int,card_id:int,location:str) -> Card:
        "Removes and returns the card from the given location for the player. location can be 'hand' or 'persistent'."
        loc = self.players[player_index].hand if (location == 'hand') else self.players[player_index].persistent_cards
        for i,c in enumerate(loc):
            if c.id == card_id:
                logger.debug(f"\t{c.name} removed from {location} of {ID_TO_PLAYER[player_index]}")
                return loc.pop(i)
    
    def add_to_discard(self,c_to_discard:Card):
        "Adds the given card to the discard pile and discard_array."
        if c_to_discard.is_dominance:
            logger.debug(f"\t{c_to_discard.name} is now Available (Discarded)")
            suit = c_to_discard.suit
            self.available_dominances[suit] = 1
            self.available_dom_card_objs[suit] = c_to_discard
            return
        logger.debug(f"\t{c_to_discard.name} added to discard pile")
        self.discard_pile.append(c_to_discard)
        i = c_to_discard.id
        if self.discard_array[i][0] == 1:
            self.discard_array[i][0] = 0
            self.discard_array[i][1] = 1
        elif self.discard_array[i][1] == 1:
            self.discard_array[i][1] = 0
            self.discard_array[i][2] = 1
        else:
            self.discard_array[i][0] = 1

    def discard_from_hand(self,player_index:int,card_id:int):
        "Makes a player discard a card of the matching id from their hand, assuming they have it."
        c_to_discard = self.get_card(player_index,card_id,"hand")
        self.add_to_discard(c_to_discard)
        self.turn_log.change_plr_hand_size(player_index,-1)
        self.turn_log.change_plr_cards_lost(player_index,card_id)
    
    def discard_from_persistent(self,player_index:int,card_id:int):
        "Makes a player discard a card of the matching id from their persistent cards, assuming they have it."
        c_to_discard = self.get_card(player_index,card_id,"persistent")
        self.add_to_discard(c_to_discard)
        self.turn_log.change_plr_cards_lost(player_index,card_id)
    
    def discard_from_supporters(self,aplayer:Alliance, card_id:int):
        "Makes an alliance player discard a card from their supporters."
        for i,c in enumerate(aplayer.supporters):
            if c.id == card_id:
                logger.debug(f"\t{c.name} removed from Supporters of Woodland Alliance")
                c_to_discard = aplayer.supporters.pop(i)
                break
        self.add_to_discard(c_to_discard)
        aplayer.spend_supporter_helper(c_to_discard.suit)
        self.turn_log.change_alliance_payments(card_id)

    def draw_new_quest(self):
        """
        Draws 1 card from the top of the quest deck to become available.

        If the deck runs out, nothing happens.
        """
        # check in case the deck is empty
        if self.deck.size() == 0:
            logger.debug("--> No quests remaining to complete!")
            return
        draw = self.quest_deck.draw(1)
        c_drawn = draw[0]
        logger.debug(f"\t* New Quest Drawn: {c_drawn.name}")
        self.active_quests.append(c_drawn)
    
    def complete_quest(self,vplayer:Vagabond,quest_id:int):
        """
        Makes the given Vagabond player complete the given quest, assuming it is currently possible.

        Removes the given quest from the active list and adds it to the correct
        list for the VB's completed quests. Then, attempts to exhaust the
        required items. Finally, draws a new quest.

        Does NOT score points or make the VB draw new cards.
        """
        for i,c in enumerate(self.active_quests):
            if c.id == quest_id:
                quest_card = self.active_quests.pop(i)
                break
        logger.debug(f"\tQuest Completed: {quest_card.name} ({ID_TO_SUIT[quest_card.suit]})")
        vplayer.completed_quests[quest_card.suit].append(quest_card)
        for item,amount in quest_card.requirements.items():
            for _ in range(amount):
                vplayer.exhaust_item(item)
        self.draw_new_quest()

    def change_score(self,player_index:int,amount:int):
        "Makes a player score some amount of points. Use a negative amount to lose points."
        if amount == 0:
            return
        if self.active_dominances[player_index] is not None:
            logger.debug(f"\t{amount} Point(s) not scored due to active dominance.")
            return
        
        old_points = self.victory_points[player_index]
        new_points = max(0, old_points + amount)
        real_change = new_points - old_points

        self.victory_points[player_index] = new_points
        logger.debug(f"\t{ID_TO_PLAYER[player_index]} Points changed by {amount}")
        logger.debug(f"\t\tNew Score:")
        for i in range(3):
            logger.debug(f"\t\t> {ID_TO_PLAYER[i]}: {self.victory_points[i]}")
            # change the points scored for this action / balanced for other players
            if i == player_index:
                self.points_scored_this_action[i] += real_change * GAME_SCALAR
            else:
                self.points_scored_this_action[i] -= (real_change / (N_PLAYERS - 1)) * GAME_SCALAR
            if self.active_dominances[i] is not None:
                logger.debug(f"\t\t\t- {ID_TO_SUIT[self.active_dominances[i].suit]} Dominance Active")
        self.turn_log.change_plr_points(player_index,real_change)

    def craft_card(self,player_index:int,card_id:int):
        "Makes the player craft the given card, assuming the action is legal."
        card_to_craft = None
        p = self.players[player_index]
        for i,c in enumerate(p.hand):
            if c.id == card_id:
                card_to_craft = c
                hand_i = i
                break
        
        if card_to_craft is None:
            print(f"ERROR: Could not find card id {card_id} in Player {player_index}'s hand: {p.hand}")

        self.turn_log.change_plr_cards_crafted(player_index,card_id)
        item_id = card_to_craft.crafting_item
        if item_id != ITEM_NONE:
            # we are crafting an item
            logger.debug(f"\t{ID_TO_PLAYER[player_index]} crafts: {ID_TO_ITEM[item_id]}")
            self.available_items[item_id] -= 1
            if player_index != PIND_VAGABOND:
                p.crafted_items[item_id] += 1
            else:
                p.add_item(item_id,0,0)
            
            # Disdain for Trade for the Eyrie (unless they have Builder leader)
            points_scored = 1 if (player_index == PIND_EYRIE and p.chosen_leader_index != LEADER_BUILDER) else card_to_craft.points
            self.change_score(player_index,points_scored)
            self.discard_from_hand(player_index,card_id)
        elif card_to_craft.is_persistent:
            # we are crafting a persistent card
            logger.debug(f"\t{ID_TO_PLAYER[player_index]} crafts: {card_to_craft.name}")
            p.persistent_cards.append(card_to_craft)
            p.hand.pop(hand_i)
            self.turn_log.change_plr_hand_size(player_index,-1)
        elif card_id in CID_FAVORS:
            # a favor card has been activated
            logger.debug(f"\t{ID_TO_PLAYER[player_index]} crafts: {card_to_craft.name}")
            points_scored,fh_list = self.board.resolve_favor(player_index,CLEARING_SUITS[card_to_craft.suit])
            # clear out each clearing
            for cid in CLEARING_SUITS[card_to_craft.suit]:
                clearing = self.board.clearings[cid]
                for faction_i in {j for j in range(N_PLAYERS) if j != player_index}:
                    player = self.players[faction_i]
                    foo = clearing.get_num_warriors(faction_i)
                    logger.debug(f"\t\tRemoving {foo} {ID_TO_PLAYER[faction_i]} warriors in clearing {cid}")
                    clearing.change_num_warriors(faction_i,-foo)
                    self.turn_log.change_clr_warriors(cid,faction_i,-foo)
                    player.change_num_warriors(foo)
                    self.turn_log.change_plr_warrior_supply(faction_i,foo)

                    while len(clearing.buildings[faction_i]) > 0:
                        foo = clearing.buildings[faction_i].pop()
                        player.change_num_buildings(foo,1)
                        self.turn_log.change_clr_building(cid,faction_i,foo,-1)
                        logger.debug(f"\t\tBuilding of {ID_TO_PLAYER[faction_i]} removed from clearing {cid}")
                        if faction_i == PIND_ALLIANCE:
                            self.base_removal_helper(player,clearing.suit)

                    while len(clearing.tokens[faction_i]) > 0:
                        foo = clearing.tokens[faction_i].pop()
                        player.change_num_tokens(foo,1)
                        self.turn_log.change_clr_tokens(cid,faction_i,foo,-1)
                        logger.debug(f"\t\tToken of {ID_TO_PLAYER[faction_i]} removed from clearing {cid}")
                        if faction_i == PIND_ALLIANCE:
                            self.outrage_offender = player_index
                            self.outrage_suits.append(clearing.suit)

            self.change_score(player_index,points_scored)
            if bool(fh_list) and self.keep_is_up():
                self.field_hospitals += sorted(fh_list)
            self.discard_from_hand(player_index, card_id)
        for i in range(3):
            self.remaining_craft_power[i] -= card_to_craft.crafting_recipe[i]

    def craft_royal_claim(self,player_index:int,action:int):
        "Crafts Royal Claim for the given player using a specific crafting power."
        p = self.players[player_index]
        for i,c in enumerate(p.hand):
            if c.id == CID_ROYAL_CLAIM:
                card_to_craft = c
                hand_i = i
                break
        p.persistent_cards.append(card_to_craft)
        self.turn_log.change_plr_cards_crafted(player_index,CID_ROYAL_CLAIM)
        p.hand.pop(hand_i)
        self.turn_log.change_plr_hand_size(player_index,-1)
        recipe_used = AID_CRAFT_RC_MAPPING[action]
        logger.debug(f"\t{ID_TO_PLAYER[player_index]} crafts: {card_to_craft.name} with recipe {recipe_used} (Mouse,Rabbit,Fox)")
        for i in range(3):
            self.remaining_craft_power[i] -= recipe_used[i]

    # BATTLE FUNCTIONS
    def keep_is_up(self):
        "Returns True only if the Marquise's Keep token is still on the map."
        mplayer = self.players[PIND_MARQUISE]
        return self.board.clearings[mplayer.keep_clearing_id].get_num_tokens(PIND_MARQUISE,TIND_KEEP) > 0

    def activate_field_hospitals(self,amount:int,payment_card_id:int):
        "Places 'amount' of warriors at the Marquise's Keep and discards the given card from their hand."
        logger.debug(f"\tField Hospitals activated, Marquise recovered {amount} warrior(s)")
        keep_clearing = self.players[PIND_MARQUISE].keep_clearing_id
        self.players[PIND_MARQUISE].change_num_warriors(-amount)
        self.turn_log.change_plr_warrior_supply(PIND_MARQUISE,-amount)
        self.board.place_warriors(PIND_MARQUISE,amount,keep_clearing)
        self.turn_log.change_clr_warriors(keep_clearing,PIND_MARQUISE,amount)
        self.discard_from_hand(PIND_MARQUISE,payment_card_id)

    def score_battle_points(self,faction_index:int,is_attacker:bool,cardboard_removed:int):
        """
        Given the number of cardboard pieces removed, make the given faction
        score the right number of points. Accounts for the Eyrie Despot with
        extra info stored in the current Battle object.
        """
        points = cardboard_removed
        if (faction_index == PIND_EYRIE) and (self.players[faction_index].chosen_leader_index == LEADER_DESPOT) and cardboard_removed:
            # if the Eyrie with Despot are removing cardboard, score them an extra point if they haven't
            # received an extra point thus far for doing so
            if (not self.battle.att_cardboard_removed) and is_attacker:
                logger.debug("> Bonus point scored from Despot!")
                points += 1
                self.battle.att_cardboard_removed = True
            elif (not self.battle.def_cardboard_removed) and (not is_attacker):
                logger.debug("> Bonus point scored from Despot!")
                points += 1
                self.battle.def_cardboard_removed = True
        self.change_score(faction_index,points)

    def deal_hits(self,faction_index:int,amount:int,clearing_index:int):
        """
        Make the given faction take a certain number of hits in the given clearing.
        Removes warriors first, then buildings/tokens if necessary.

        Returns a 3-tuple:
        - 1. The number of hits left to deal. A positive amount means a choice must be made.
        - 2. The number of warriors killed.
        - 3. The number of pieces of cardboard removed (buildings + tokens)
        """
        target_clearing = self.board.clearings[clearing_index]
        target_faction = self.players[faction_index]
        warriors_removed = cardboard_removed = 0
        while amount > 0:
            # first take out warriors
            if target_clearing.get_num_warriors(faction_index) > 0:
                logger.debug(f"- Removed 1 {ID_TO_PLAYER[faction_index]} warrior from clearing {clearing_index}")
                target_clearing.change_num_warriors(faction_index,-1)
                self.turn_log.change_clr_warriors(clearing_index,faction_index,-1)
                target_faction.change_num_warriors(1)
                self.turn_log.change_plr_warrior_supply(faction_index,1)
                warriors_removed += 1
                amount -= 1
            # then check if there is a choice of building / token
            elif faction_index == PIND_MARQUISE:
                # find how many choices the Marquise have of removing tokens/buildings
                building_choices = sum((target_clearing.get_num_buildings(faction_index,bid) > 0) for bid in range(3))
                token_choices = sum((target_clearing.get_num_tokens(faction_index,tid) > 0) for tid in range(2))
                # if they have a choice at all, then leave with 'amount' > 0 to indicate this
                if (building_choices + token_choices) > 1:
                    break
                # otherwise, there is no choice in what to remove
                # if it is a building, remove one of them
                elif building_choices == 1:
                    for bid in range(3):
                        if target_clearing.get_num_buildings(faction_index,bid) > 0:
                            logger.debug(f"-- {ID_TO_MBUILD[bid]} destroyed in clearing {clearing_index}")
                            target_clearing.remove_building(faction_index,bid)
                            target_faction.change_num_buildings(bid,1)
                            self.turn_log.change_clr_building(clearing_index,faction_index,bid,-1)
                            cardboard_removed += 1
                            amount -= 1
                # if it is a token, remove one of them
                elif token_choices == 1:
                    for tid in range(2):
                        if target_clearing.get_num_tokens(faction_index,tid) > 0:
                            logger.debug(f"-- {ID_TO_MTOKEN[tid]} destroyed in clearing {clearing_index}")
                            target_clearing.remove_token(faction_index,tid)
                            target_faction.change_num_tokens(tid,1)
                            self.turn_log.change_clr_tokens(clearing_index,faction_index,tid,-1)
                            cardboard_removed += 1
                            amount -= 1
                # if no other item is present, the remaining hits are lost
                else:
                    amount = 0
            elif faction_index == PIND_EYRIE:
                # the Eyrie can only have roosts, and have no tokens
                if target_clearing.get_num_buildings(faction_index,BIND_ROOST) > 0:
                    logger.debug(f"-- Roost destroyed in clearing {clearing_index}")
                    target_clearing.remove_building(faction_index,BIND_ROOST)
                    target_faction.change_num_buildings(BIND_ROOST,1)
                    self.turn_log.change_clr_building(clearing_index,faction_index,BIND_ROOST,-1)
                    cardboard_removed += 1
                    amount -= 1
                else:
                    amount = 0
            elif faction_index == PIND_ALLIANCE:
                # find how many choices the Alliance have of removing tokens/buildings
                building_choices = sum((target_clearing.get_num_buildings(faction_index,bid) > 0) for bid in range(3))
                token_choices = 1 if (target_clearing.get_num_tokens(faction_index,TIND_SYMPATHY) > 0) else 0
                # if they have a choice at all, then leave with 'amount' > 0 to indicate this
                if (building_choices + token_choices) > 1:
                    break
                # otherwise, there is no choice in what to remove
                # if it is a building, remove one of them
                elif building_choices == 1:
                    for bid in range(3):
                        if target_clearing.get_num_buildings(faction_index,bid) > 0:
                            logger.debug(f"-- {ID_TO_ABUILD[bid]} destroyed in clearing {clearing_index}")
                            target_clearing.remove_building(faction_index,bid)
                            self.turn_log.change_clr_building(clearing_index,faction_index,bid,-1)
                            self.base_removal_helper(target_faction,bid)
                            cardboard_removed += 1
                            amount -= 1
                # if it is a token, remove one of them
                elif token_choices == 1:
                    logger.debug(f"-- Sympathy destroyed in clearing {clearing_index}")
                    target_clearing.remove_token(faction_index,TIND_SYMPATHY)
                    self.turn_log.change_clr_tokens(clearing_index,faction_index,TIND_SYMPATHY,-1)
                    target_faction.change_num_tokens(TIND_SYMPATHY,1)
                    self.outrage_offender = self.battle.attacker_id if (self.battle.defender_id == PIND_ALLIANCE) else self.battle.defender_id
                    self.outrage_suits.append(target_clearing.suit)
                    logger.debug(f"AUTO - offender:{self.outrage_offender}, suits:{self.outrage_suits}")
                    cardboard_removed += 1
                    amount -= 1
                # if no other item is present, the remaining hits are lost
                else:
                    amount = 0
        return amount, warriors_removed, cardboard_removed

    def resolve_battle_action(self,action):
        """
        Given an action number, performs the given action given the
        current information stored about any battles currently going on.

        Assumes that self.battle points to an existing Battle object.
        """
        defender = self.players[self.battle.defender_id]
        clearing = self.board.clearings[self.battle.clearing_id]
        if self.battle.stage is None:
            # the battle just started, assume a brand new Battle object was just created
            logger.debug(f"\t--- BATTLE STARTED: {ID_TO_PLAYER[self.battle.attacker_id]} Attacks {ID_TO_PLAYER[self.battle.defender_id]} in clearing {self.battle.clearing_id}")
            ans = defender.get_ambush_actions(clearing.suit)
            if bool(ans):
                # the defender chooses to ambush or not
                self.battle.stage = Battle.STAGE_DEF_AMBUSH
                self.current_player = (self.battle.defender_id)
                return ans
            # no ambush is possible, so we move straight to the dice roll
            logger.debug(f"{ID_TO_PLAYER[self.battle.defender_id]} chose not to ambush")
            self.battle.stage = Battle.STAGE_DICE_ROLL

        attacker = self.players[self.battle.attacker_id]
        if self.battle.stage == Battle.STAGE_DEF_ORDER:
            # action is what defender building/token to hit with the next hit
            if self.battle.defender_id == PIND_MARQUISE:
                if action in {AID_ORDER_KEEP,AID_ORDER_WOOD}:
                    clearing.remove_token(self.battle.defender_id,action - AID_ORDER_KEEP)
                    self.turn_log.change_clr_tokens(self.battle.clearing_id,self.battle.defender_id,action - AID_ORDER_KEEP,-1)
                    defender.change_num_tokens(action - AID_ORDER_KEEP,1)
                    item = ID_TO_MTOKEN[action - AID_ORDER_KEEP]
                else:
                    clearing.remove_building(self.battle.defender_id,action - AID_ORDER_SAWMILL)
                    self.turn_log.change_clr_building(self.battle.clearing_id,self.battle.defender_id,action - AID_ORDER_SAWMILL,-1)
                    defender.change_num_buildings(action - AID_ORDER_SAWMILL,1)
                    item = ID_TO_MBUILD[action - AID_ORDER_SAWMILL]
            elif self.battle.defender_id == PIND_ALLIANCE:
                if action == AID_ORDER_SYMPATHY:
                    clearing.remove_token(PIND_ALLIANCE,TIND_SYMPATHY)
                    self.turn_log.change_clr_tokens(self.battle.clearing_id,self.battle.defender_id,TIND_SYMPATHY,-1)
                    defender.change_num_tokens(TIND_SYMPATHY,1)
                    item = ID_TO_ATOKEN[TIND_SYMPATHY]
                    self.outrage_offender = self.battle.attacker_id
                    self.outrage_suits.append(clearing.suit)
                else:
                    clearing.remove_building(PIND_ALLIANCE,action - AID_ORDER_BASE_MOUSE)
                    self.turn_log.change_clr_building(self.battle.clearing_id,self.battle.defender_id,action - AID_ORDER_BASE_MOUSE,-1)
                    self.base_removal_helper(defender,clearing.suit)
                    item = ID_TO_ABUILD[action - AID_ORDER_BASE_MOUSE]
            logger.debug(f"{ID_TO_PLAYER[self.battle.defender_id]} chose to destroy {item}")
            self.score_battle_points(self.battle.attacker_id,True,1)
            # see if there is a choice anymore
            self.battle.att_hits_to_deal,warriors_killed,cardboard_removed = self.deal_hits(self.battle.defender_id, self.battle.att_hits_to_deal - 1, self.battle.clearing_id)
            if cardboard_removed:
                self.score_battle_points(self.battle.attacker_id,True,cardboard_removed)
            if warriors_killed and self.battle.defender_id == PIND_MARQUISE and defender.has_suit_in_hand(clearing.suit) and self.keep_is_up():
                self.field_hospitals.append((warriors_killed,clearing.suit))

            if self.battle.att_hits_to_deal > 0:
                # defender still has a choice of what to remove
                self.current_player = (self.battle.defender_id)
                if self.battle.defender_id == PIND_MARQUISE:
                    building_choices = [bid+AID_ORDER_SAWMILL for bid in range(3) if (clearing.get_num_buildings(self.battle.defender_id,bid) > 0)]
                    token_choices = [tid+AID_ORDER_KEEP for tid in range(2) if (clearing.get_num_tokens(self.battle.defender_id,tid) > 0)]
                elif self.battle.defender_id == PIND_ALLIANCE:
                    building_choices = [bid+AID_ORDER_BASE_MOUSE for bid in range(3) if (clearing.get_num_buildings(self.battle.defender_id,bid) > 0)]
                    token_choices = [AID_ORDER_SYMPATHY for tid in range(1) if (clearing.get_num_tokens(self.battle.defender_id,tid) > 0)]
                return building_choices + token_choices
            
            # all hits needed have been dealt to defender
            # it is now the attacker's turn to take hits
            self.battle.def_hits_to_deal,warriors_killed,cardboard_removed = self.deal_hits(self.battle.attacker_id, self.battle.def_hits_to_deal, self.battle.clearing_id)
            if cardboard_removed:
                self.score_battle_points(self.battle.defender_id,False,cardboard_removed)
            if warriors_killed and self.battle.attacker_id == PIND_MARQUISE and attacker.has_suit_in_hand(clearing.suit) and self.keep_is_up():
                self.field_hospitals.append((warriors_killed,clearing.suit))
            
            self.current_player = (self.battle.attacker_id)
            if self.battle.def_hits_to_deal > 0:
                # attacker has a choice on what to remove
                self.battle.stage = Battle.STAGE_ATT_ORDER
                if self.battle.attacker_id == PIND_MARQUISE:
                    building_choices = [bid+AID_ORDER_SAWMILL for bid in range(3) if (clearing.get_num_buildings(self.battle.attacker_id,bid) > 0)]
                    token_choices = [tid+AID_ORDER_KEEP for tid in range(2) if (clearing.get_num_tokens(self.battle.attacker_id,tid) > 0)]
                elif self.battle.attacker_id == PIND_ALLIANCE:
                    building_choices = [bid+AID_ORDER_BASE_MOUSE for bid in range(3) if (clearing.get_num_buildings(self.battle.attacker_id,bid) > 0)]
                    token_choices = [AID_ORDER_SYMPATHY for tid in range(1) if (clearing.get_num_tokens(self.battle.attacker_id,tid) > 0)]
                return building_choices + token_choices
            else:
                # the battle is over
                logger.debug(f"--- BATTLE FINISHED")
                self.battle.stage = Battle.STAGE_DONE
                return []
            
        if self.battle.stage == Battle.STAGE_ATT_ORDER:
            # action is what attacker building/token to hit with the next hit
            if self.battle.attacker_id == PIND_MARQUISE:
                if action in {AID_ORDER_KEEP,AID_ORDER_WOOD}:
                    clearing.remove_token(self.battle.attacker_id,action - AID_ORDER_KEEP)
                    self.turn_log.change_clr_tokens(self.battle.clearing_id,self.battle.defender_id,action - AID_ORDER_KEEP,-1)
                    attacker.change_num_tokens(action - AID_ORDER_KEEP,1)
                    item = ID_TO_MTOKEN[action - AID_ORDER_KEEP]
                else:
                    clearing.remove_building(self.battle.attacker_id,action - AID_ORDER_SAWMILL)
                    self.turn_log.change_clr_building(self.battle.clearing_id,self.battle.defender_id,action - AID_ORDER_SAWMILL,-1)
                    attacker.change_num_buildings(action - AID_ORDER_SAWMILL,1)
                    item = ID_TO_MBUILD[action - AID_ORDER_SAWMILL]
            elif self.battle.attacker_id == PIND_ALLIANCE:
                if action == AID_ORDER_SYMPATHY:
                    clearing.remove_token(PIND_ALLIANCE,TIND_SYMPATHY)
                    self.turn_log.change_clr_tokens(self.battle.clearing_id,self.battle.defender_id,TIND_SYMPATHY,-1)
                    attacker.change_num_tokens(TIND_SYMPATHY,1)
                    item = ID_TO_ATOKEN[TIND_SYMPATHY]
                    self.outrage_offender = self.battle.defender_id
                    self.outrage_suits.append(clearing.suit)
                else:
                    clearing.remove_building(PIND_ALLIANCE,action - AID_ORDER_BASE_MOUSE)
                    self.turn_log.change_clr_building(self.battle.clearing_id,self.battle.defender_id,action - AID_ORDER_BASE_MOUSE,-1)
                    self.base_removal_helper(attacker,clearing.suit)
                    item = ID_TO_ABUILD[action - AID_ORDER_BASE_MOUSE]
            logger.debug(f"{ID_TO_PLAYER[self.battle.attacker_id]} chose to destroy {item}")
            self.score_battle_points(self.battle.defender_id,False,1)
            # see if there is a choice anymore
            self.battle.def_hits_to_deal,warriors_killed,cardboard_removed = self.deal_hits(self.battle.attacker_id, self.battle.def_hits_to_deal - 1, self.battle.clearing_id)
            if cardboard_removed:
                self.score_battle_points(self.battle.defender_id,False,cardboard_removed)
            if warriors_killed and self.battle.attacker_id == PIND_MARQUISE and attacker.has_suit_in_hand(clearing.suit) and self.keep_is_up():
                self.field_hospitals.append((warriors_killed,clearing.suit))
            
            self.current_player = (self.battle.attacker_id)
            if self.battle.def_hits_to_deal > 0:
                # attacker still has a choice of what to remove
                self.battle.stage = Battle.STAGE_ATT_ORDER
                if self.battle.attacker_id == PIND_MARQUISE:
                    building_choices = [bid+AID_ORDER_SAWMILL for bid in range(3) if (clearing.get_num_buildings(self.battle.attacker_id,bid) > 0)]
                    token_choices = [tid+AID_ORDER_KEEP for tid in range(2) if (clearing.get_num_tokens(self.battle.attacker_id,tid) > 0)]
                elif self.battle.attacker_id == PIND_ALLIANCE:
                    building_choices = [bid+AID_ORDER_BASE_MOUSE for bid in range(3) if (clearing.get_num_buildings(self.battle.attacker_id,bid) > 0)]
                    token_choices = [AID_ORDER_SYMPATHY for tid in range(1) if (clearing.get_num_tokens(self.battle.attacker_id,tid) > 0)]
                return building_choices + token_choices
            
            # All hits have been dealt, so we are in one of two possible situations:
            # 1. The dice have not been rolled. If the -attacker- is choosing buildings to destroy,
            #    then that means they attacked with 1 warrior and were ambushed, and now had to
            #    choose one of their buildings to remove. They have no warriors and the battle is over.
            # 2. The dice have been rolled and extra effects have been chosen. However, the attacker is
            #    last to pick which things get hit in what order, so the battle must be over.
            logger.debug(f"--- BATTLE FINISHED")
            self.battle.stage = Battle.STAGE_DONE
            return []
        
        if self.battle.stage == Battle.STAGE_DEF_AMBUSH:
            # action is the defender's choice to ambush or not
            if action == AID_AMBUSH_NONE:
                logger.debug(f"{ID_TO_PLAYER[self.battle.defender_id]} chose not to ambush")
                # we immediately go to the dice roll
                self.battle.stage = Battle.STAGE_DICE_ROLL
            else:
                # save which ambush card is played
                logger.debug(f"{ID_TO_PLAYER[self.battle.defender_id]} chooses to AMBUSH!")
                if action == AID_AMBUSH_BIRD:
                    self.battle.def_ambush_id = CID_AMBUSH_BIRD
                elif action == AID_AMBUSH_MOUSE:
                    self.battle.def_ambush_id = CID_AMBUSH_MOUSE
                elif action == AID_AMBUSH_RABBIT:
                    self.battle.def_ambush_id = CID_AMBUSH_RABBIT
                elif action == AID_AMBUSH_FOX:
                    self.battle.def_ambush_id = CID_AMBUSH_FOX
                # make the defender discard this card
                self.discard_from_hand(self.battle.defender_id,self.battle.def_ambush_id)

                # check if the attacker has Scouting Party (nullifies ambush cards used up)
                if any((c.id == CID_SCOUTING_PARTY) for c in attacker.persistent_cards):
                    logger.debug("The ambush is thwarted by a Scouting Party!")
                    self.turn_log.change_plr_pers_used(self.battle.attacker_id, CID_TO_PERS_INDEX[CID_SCOUTING_PARTY],self.battle.defender_id)
                    self.battle.stage = Battle.STAGE_DICE_ROLL
                # otherwise, see if attacker can choose to counter ambush
                else:
                    ans = attacker.get_ambush_actions(clearing.suit)
                    if bool(ans):
                        self.battle.stage = Battle.STAGE_ATT_AMBUSH
                        self.current_player = (self.battle.attacker_id)
                        return ans
                    # otherwise, the ambush triggers and 2 hits are dealt
                    logger.debug(f"{ID_TO_PLAYER[self.battle.attacker_id]} chose not to counter-ambush")
                    self.battle.def_hits_to_deal,warriors_killed,cardboard_removed = self.deal_hits(self.battle.attacker_id, 2, self.battle.clearing_id)
                    if cardboard_removed:
                        self.score_battle_points(self.battle.defender_id,False,cardboard_removed)
                    if warriors_killed and self.battle.attacker_id == PIND_MARQUISE and attacker.has_suit_in_hand(clearing.suit) and self.keep_is_up():
                        self.field_hospitals.append((warriors_killed,clearing.suit))

                    # check if a choice must be made from hits
                    if self.battle.def_hits_to_deal > 0:
                        self.current_player = (self.battle.attacker_id)
                        self.battle.stage = Battle.STAGE_ATT_ORDER
                        if self.battle.attacker_id == PIND_MARQUISE:
                            building_choices = [bid+AID_ORDER_SAWMILL for bid in range(3) if (clearing.get_num_buildings(self.battle.attacker_id,bid) > 0)]
                            token_choices = [tid+AID_ORDER_KEEP for tid in range(2) if (clearing.get_num_tokens(self.battle.attacker_id,tid) > 0)]
                        elif self.battle.attacker_id == PIND_ALLIANCE:
                            building_choices = [bid+AID_ORDER_BASE_MOUSE for bid in range(3) if (clearing.get_num_buildings(self.battle.attacker_id,bid) > 0)]
                            token_choices = [AID_ORDER_SYMPATHY for tid in range(1) if (clearing.get_num_tokens(self.battle.attacker_id,tid) > 0)]
                        return building_choices + token_choices
                    # if the hits are all dealt, we check if a battle can still occur
                    elif clearing.get_num_warriors(self.battle.attacker_id) > 0:
                        logger.debug("Continuing to the dice roll...")
                        self.battle.stage = Battle.STAGE_DICE_ROLL
                    # otherwise, the ambush wiped out all attackers
                    else:
                        logger.debug("Ouch, should have brought more backup...")
                        logger.debug(f"--- BATTLE FINISHED")
                        self.battle.stage = Battle.STAGE_DONE
                        self.current_player = (self.battle.attacker_id)
                        return []

        if self.battle.stage == Battle.STAGE_ATT_AMBUSH:
            # action is the attacker's choice to counter ambush or not
            if action == AID_AMBUSH_NONE:
                logger.debug(f"{ID_TO_PLAYER[self.battle.attacker_id]} chose not to counter-ambush")
                # the ambush triggers and 2 hits are dealt
                # deal_hits returns the number of remaining hits there are; if >0, it means a choice is possible for the one getting hit
                self.battle.def_hits_to_deal,warriors_killed,cardboard_removed = self.deal_hits(self.battle.attacker_id, 2, self.battle.clearing_id)
                if cardboard_removed:
                    self.score_battle_points(self.battle.defender_id,False,cardboard_removed)
                if warriors_killed and self.battle.attacker_id == PIND_MARQUISE and attacker.has_suit_in_hand(clearing.suit) and self.keep_is_up():
                    self.field_hospitals.append((warriors_killed,clearing.suit))

                # check if a choice must be made from hits
                if self.battle.def_hits_to_deal > 0:
                    self.current_player = (self.battle.attacker_id)
                    self.battle.stage = Battle.STAGE_ATT_ORDER
                    if self.battle.attacker_id == PIND_MARQUISE:
                        building_choices = [bid+AID_ORDER_SAWMILL for bid in range(3) if (clearing.get_num_buildings(self.battle.attacker_id,bid) > 0)]
                        token_choices = [tid+AID_ORDER_KEEP for tid in range(2) if (clearing.get_num_tokens(self.battle.attacker_id,tid) > 0)]
                    elif self.battle.attacker_id == PIND_ALLIANCE:
                        building_choices = [bid+AID_ORDER_BASE_MOUSE for bid in range(3) if (clearing.get_num_buildings(self.battle.attacker_id,bid) > 0)]
                        token_choices = [AID_ORDER_SYMPATHY for tid in range(1) if (clearing.get_num_tokens(self.battle.attacker_id,tid) > 0)]
                    return building_choices + token_choices
                # if the hits are all dealt, we check if a battle can still occur
                elif clearing.get_num_warriors(self.battle.attacker_id) > 0:
                    logger.debug("Continuing to the dice roll...")
                    self.battle.stage = Battle.STAGE_DICE_ROLL
                # otherwise, the ambush wiped out all attackers
                else:
                    logger.debug("Ouch, should have brought more backup...")
                    logger.debug(f"--- BATTLE FINISHED")
                    self.battle.stage = Battle.STAGE_DONE
                    self.current_player = (self.battle.attacker_id)
                    return []
            else:
                # save which ambush card is played
                logger.debug(f"{ID_TO_PLAYER[self.battle.attacker_id]} chooses to COUNTER-AMBUSH!")
                if action == AID_AMBUSH_BIRD:
                    self.battle.att_ambush_id = CID_AMBUSH_BIRD
                elif action == AID_AMBUSH_MOUSE:
                    self.battle.att_ambush_id = CID_AMBUSH_MOUSE
                elif action == AID_AMBUSH_RABBIT:
                    self.battle.att_ambush_id = CID_AMBUSH_RABBIT
                elif action == AID_AMBUSH_FOX:
                    self.battle.att_ambush_id = CID_AMBUSH_FOX
                # make the attacker discard this card
                self.discard_from_hand(self.battle.attacker_id,self.battle.att_ambush_id)
                # we immediately go to the dice roll, since the defender's ambush is cancelled
                logger.debug("Continuing to the dice roll...")
                self.battle.stage = Battle.STAGE_DICE_ROLL

        if self.battle.stage == Battle.STAGE_DICE_ROLL:
            # the dice must be rolled before continuing
            roll = [random.randint(0,3) for i in range(2)]
            logger.debug(f"--- DICE ROLL: {roll}")
            if self.battle.defender_id == PIND_ALLIANCE: # Guerrilla Tactics
                self.battle.att_rolled_hits = min(clearing.get_num_warriors(self.battle.attacker_id), min(roll))
                self.battle.def_rolled_hits = min(clearing.get_num_warriors(self.battle.defender_id), max(roll))
            else:
                self.battle.att_rolled_hits = min(clearing.get_num_warriors(self.battle.attacker_id), max(roll))
                self.battle.def_rolled_hits = min(clearing.get_num_warriors(self.battle.defender_id), min(roll))
            # defenseless
            if clearing.get_num_warriors(self.battle.defender_id) == 0:
                logger.debug(f"{ID_TO_PLAYER[self.battle.defender_id]} is defenseless (+1 hit taken)")
                self.battle.att_extra_hits += 1
            # Eyrie Commander Leader
            if (self.battle.attacker_id == PIND_EYRIE) and (attacker.chosen_leader_index == LEADER_COMMANDER):
                logger.debug(f"{ID_TO_PLAYER[self.battle.attacker_id]} is led by the Commander (+1 hit dealt)")
                self.battle.att_extra_hits += 1

            # check if the attacker can choose extra effects
            ans = attacker.get_attacker_card_actions()
            if bool(ans):
                self.current_player = (self.battle.attacker_id)
                self.battle.stage = Battle.STAGE_ATT_EFFECTS
                return ans
            # check if the defender can choose extra effects
            ans = defender.get_defender_card_actions()
            if bool(ans):
                self.current_player = (self.battle.defender_id)
                self.battle.stage = Battle.STAGE_DEF_EFFECTS
                return ans
            
            # no extra effects can be chosen, so deal the hits next
            # next, the defender takes hits first
            logger.debug("--- Dealing hits to defender...")
            self.battle.def_hits_to_deal = self.battle.def_rolled_hits
            self.battle.att_hits_to_deal,warriors_killed,cardboard_removed = self.deal_hits(self.battle.defender_id,self.battle.att_extra_hits+self.battle.att_rolled_hits,self.battle.clearing_id)
            if cardboard_removed:
                self.score_battle_points(self.battle.attacker_id,True,cardboard_removed)
            if warriors_killed and self.battle.defender_id == PIND_MARQUISE and defender.has_suit_in_hand(clearing.suit) and self.keep_is_up():
                self.field_hospitals.append((warriors_killed,clearing.suit))

            if self.battle.att_hits_to_deal > 0:
                # defender has a choice of what to remove
                self.current_player = (self.battle.defender_id)
                self.battle.stage = Battle.STAGE_DEF_ORDER
                if self.battle.defender_id == PIND_MARQUISE:
                    building_choices = [bid+AID_ORDER_SAWMILL for bid in range(3) if (clearing.get_num_buildings(self.battle.defender_id,bid) > 0)]
                    token_choices = [tid+AID_ORDER_KEEP for tid in range(2) if (clearing.get_num_tokens(self.battle.defender_id,tid) > 0)]
                elif self.battle.defender_id == PIND_ALLIANCE:
                    building_choices = [bid+AID_ORDER_BASE_MOUSE for bid in range(3) if (clearing.get_num_buildings(self.battle.defender_id,bid) > 0)]
                    token_choices = [AID_ORDER_SYMPATHY for tid in range(1) if (clearing.get_num_tokens(self.battle.defender_id,tid) > 0)]
                return building_choices + token_choices
            # lastly, attacker takes hits
            logger.debug("--- Dealing hits to attacker...")
            self.battle.def_hits_to_deal,warriors_killed,cardboard_removed = self.deal_hits(self.battle.attacker_id,self.battle.def_hits_to_deal,self.battle.clearing_id)
            if cardboard_removed:
                self.score_battle_points(self.battle.defender_id,False,cardboard_removed)
            if warriors_killed and self.battle.attacker_id == PIND_MARQUISE and attacker.has_suit_in_hand(clearing.suit) and self.keep_is_up():
                self.field_hospitals.append((warriors_killed,clearing.suit))

            if self.battle.def_hits_to_deal > 0:
                # attacker has a choice of what to remove
                self.current_player = (self.battle.attacker_id)
                self.battle.stage = Battle.STAGE_ATT_ORDER
                if self.battle.attacker_id == PIND_MARQUISE:
                    building_choices = [bid+AID_ORDER_SAWMILL for bid in range(3) if (clearing.get_num_buildings(self.battle.attacker_id,bid) > 0)]
                    token_choices = [tid+AID_ORDER_KEEP for tid in range(2) if (clearing.get_num_tokens(self.battle.attacker_id,tid) > 0)]
                elif self.battle.attacker_id == PIND_ALLIANCE:
                    building_choices = [bid+AID_ORDER_BASE_MOUSE for bid in range(3) if (clearing.get_num_buildings(self.battle.attacker_id,bid) > 0)]
                    token_choices = [AID_ORDER_SYMPATHY for tid in range(1) if (clearing.get_num_tokens(self.battle.attacker_id,tid) > 0)]
                return building_choices + token_choices
            # battle is over
            logger.debug(f"--- BATTLE FINISHED")
            self.battle.stage = Battle.STAGE_DONE
            self.current_player = (self.battle.attacker_id)
            return []
        
        if self.battle.stage == Battle.STAGE_ATT_EFFECTS:
            # the attacker has chosen what extra effects to use
            if action in {AID_EFFECTS_ARMORERS,AID_EFFECTS_ARM_BT}:
                # Armorers is used up
                logger.debug(f"{ID_TO_PLAYER[self.battle.attacker_id]} activates Armorers (ignores rolled hits)")
                self.battle.def_rolled_hits = 0
                self.discard_from_persistent(self.battle.attacker_id,CID_ARMORERS)
                self.turn_log.change_plr_pers_used(self.battle.attacker_id, CID_TO_PERS_INDEX[CID_ARMORERS],self.battle.defender_id)
            if action in {AID_EFFECTS_BRUTTACT,AID_EFFECTS_ARM_BT}:
                # brutal tactics is used
                logger.debug(f"{ID_TO_PLAYER[self.battle.attacker_id]} activates Brutal Tactics (+1 hit dealt)")
                self.battle.att_extra_hits += 1
                self.turn_log.change_plr_pers_used(self.battle.attacker_id, CID_TO_PERS_INDEX[CID_BRUTAL_TACTICS],self.battle.defender_id)
                self.change_score(self.battle.defender_id,1)
            # check if the defender can choose extra effects
            ans = defender.get_defender_card_actions()
            if bool(ans):
                self.current_player = (self.battle.defender_id)
                self.battle.stage = Battle.STAGE_DEF_EFFECTS
                return ans
            
            # next, the defender takes hits first
            logger.debug("--- Dealing hits to defender...")
            self.battle.def_hits_to_deal = self.battle.def_rolled_hits
            self.battle.att_hits_to_deal,warriors_killed,cardboard_removed = self.deal_hits(self.battle.defender_id,self.battle.att_extra_hits+self.battle.att_rolled_hits,self.battle.clearing_id)
            if cardboard_removed:
                self.score_battle_points(self.battle.attacker_id,True,cardboard_removed)
            if warriors_killed and self.battle.defender_id == PIND_MARQUISE and defender.has_suit_in_hand(clearing.suit) and self.keep_is_up():
                self.field_hospitals.append((warriors_killed,clearing.suit))

            if self.battle.att_hits_to_deal > 0:
                # defender has a choice of what to remove
                self.current_player = (self.battle.defender_id)
                self.battle.stage = Battle.STAGE_DEF_ORDER
                if self.battle.defender_id == PIND_MARQUISE:
                    building_choices = [bid+AID_ORDER_SAWMILL for bid in range(3) if (clearing.get_num_buildings(self.battle.defender_id,bid) > 0)]
                    token_choices = [tid+AID_ORDER_KEEP for tid in range(2) if (clearing.get_num_tokens(self.battle.defender_id,tid) > 0)]
                elif self.battle.defender_id == PIND_ALLIANCE:
                    building_choices = [bid+AID_ORDER_BASE_MOUSE for bid in range(3) if (clearing.get_num_buildings(self.battle.defender_id,bid) > 0)]
                    token_choices = [AID_ORDER_SYMPATHY for tid in range(1) if (clearing.get_num_tokens(self.battle.defender_id,tid) > 0)]
                return building_choices + token_choices
            # lastly, attacker takes hits
            logger.debug("--- Dealing hits to attacker...")
            self.battle.def_hits_to_deal,warriors_killed,cardboard_removed = self.deal_hits(self.battle.attacker_id,self.battle.def_hits_to_deal,self.battle.clearing_id)
            if cardboard_removed:
                self.score_battle_points(self.battle.defender_id,False,cardboard_removed)
            if warriors_killed and self.battle.attacker_id == PIND_MARQUISE and attacker.has_suit_in_hand(clearing.suit) and self.keep_is_up():
                self.field_hospitals.append((warriors_killed,clearing.suit))

            if self.battle.def_hits_to_deal > 0:
                # attacker has a choice of what to remove
                self.current_player = (self.battle.attacker_id)
                self.battle.stage = Battle.STAGE_ATT_ORDER
                if self.battle.attacker_id == PIND_MARQUISE:
                    building_choices = [bid+AID_ORDER_SAWMILL for bid in range(3) if (clearing.get_num_buildings(self.battle.attacker_id,bid) > 0)]
                    token_choices = [tid+AID_ORDER_KEEP for tid in range(2) if (clearing.get_num_tokens(self.battle.attacker_id,tid) > 0)]
                elif self.battle.attacker_id == PIND_ALLIANCE:
                    building_choices = [bid+AID_ORDER_BASE_MOUSE for bid in range(3) if (clearing.get_num_buildings(self.battle.attacker_id,bid) > 0)]
                    token_choices = [AID_ORDER_SYMPATHY for tid in range(1) if (clearing.get_num_tokens(self.battle.attacker_id,tid) > 0)]
                return building_choices + token_choices
            # battle is over
            logger.debug(f"--- BATTLE FINISHED")
            self.battle.stage = Battle.STAGE_DONE
            self.current_player = (self.battle.attacker_id)
            return []
        
        if self.battle.stage == Battle.STAGE_DEF_EFFECTS:
            # the defender has chosen what extra effects to use
            if action in {AID_EFFECTS_ARMORERS,AID_EFFECTS_ARMSAP}:
                # Armorers is used up
                logger.debug(f"{ID_TO_PLAYER[self.battle.defender_id]} activates Armorers (ignores rolled hits)")
                self.battle.att_rolled_hits = 0
                self.turn_log.change_plr_pers_used(self.battle.defender_id, CID_TO_PERS_INDEX[CID_ARMORERS],self.battle.attacker_id)
                self.discard_from_persistent(self.battle.defender_id,CID_ARMORERS)
            if action in {AID_EFFECTS_SAPPERS,AID_EFFECTS_ARMSAP}:
                # sappers is used
                logger.debug(f"{ID_TO_PLAYER[self.battle.defender_id]} activates Sappers (+1 hit dealt)")
                self.battle.def_extra_hits += 1
                self.turn_log.change_plr_pers_used(self.battle.defender_id, CID_TO_PERS_INDEX[CID_SAPPERS],self.battle.attacker_id)
                self.discard_from_persistent(self.battle.defender_id,CID_SAPPERS)
            # next, the defender takes hits first
            logger.debug("--- Dealing hits to defender...")
            self.battle.def_hits_to_deal = self.battle.def_rolled_hits + self.battle.def_extra_hits
            self.battle.att_hits_to_deal,warriors_killed,cardboard_removed = self.deal_hits(self.battle.defender_id,self.battle.att_extra_hits+self.battle.att_rolled_hits,self.battle.clearing_id)
            if cardboard_removed:
                self.score_battle_points(self.battle.attacker_id,True,cardboard_removed)
            if warriors_killed and self.battle.defender_id == PIND_MARQUISE and defender.has_suit_in_hand(clearing.suit) and self.keep_is_up():
                self.field_hospitals.append((warriors_killed,clearing.suit))

            if self.battle.att_hits_to_deal > 0:
                # defender has a choice of what to remove
                self.current_player = (self.battle.defender_id)
                self.battle.stage = Battle.STAGE_DEF_ORDER
                if self.battle.defender_id == PIND_MARQUISE:
                    building_choices = [bid+AID_ORDER_SAWMILL for bid in range(3) if (clearing.get_num_buildings(self.battle.defender_id,bid) > 0)]
                    token_choices = [tid+AID_ORDER_KEEP for tid in range(2) if (clearing.get_num_tokens(self.battle.defender_id,tid) > 0)]
                elif self.battle.defender_id == PIND_ALLIANCE:
                    building_choices = [bid+AID_ORDER_BASE_MOUSE for bid in range(3) if (clearing.get_num_buildings(self.battle.defender_id,bid) > 0)]
                    token_choices = [AID_ORDER_SYMPATHY for tid in range(1) if (clearing.get_num_tokens(self.battle.defender_id,tid) > 0)]
                return building_choices + token_choices
            # lastly, attacker takes hits
            logger.debug("--- Dealing hits to attacker...")
            self.battle.def_hits_to_deal,warriors_killed,cardboard_removed = self.deal_hits(self.battle.attacker_id,self.battle.def_hits_to_deal,self.battle.clearing_id)
            if cardboard_removed:
                self.score_battle_points(self.battle.defender_id,False,cardboard_removed)
            if warriors_killed and self.battle.attacker_id == PIND_MARQUISE and attacker.has_suit_in_hand(clearing.suit) and self.keep_is_up():
                self.field_hospitals.append((warriors_killed,clearing.suit))

            if self.battle.def_hits_to_deal > 0:
                # attacker has a choice of what to remove
                self.current_player = (self.battle.attacker_id)
                self.battle.stage = Battle.STAGE_ATT_ORDER
                if self.battle.attacker_id == PIND_MARQUISE:
                    building_choices = [bid+AID_ORDER_SAWMILL for bid in range(3) if (clearing.get_num_buildings(self.battle.attacker_id,bid) > 0)]
                    token_choices = [tid+AID_ORDER_KEEP for tid in range(2) if (clearing.get_num_tokens(self.battle.attacker_id,tid) > 0)]
                elif self.battle.attacker_id == PIND_ALLIANCE:
                    building_choices = [bid+AID_ORDER_BASE_MOUSE for bid in range(3) if (clearing.get_num_buildings(self.battle.attacker_id,bid) > 0)]
                    token_choices = [AID_ORDER_SYMPATHY for tid in range(1) if (clearing.get_num_tokens(self.battle.attacker_id,tid) > 0)]
                return building_choices + token_choices
            # battle is over
            logger.debug(f"--- BATTLE FINISHED")
            self.battle.stage = Battle.STAGE_DONE
            self.current_player = (self.battle.attacker_id)
            return []


    # Activating Card Effects
    def activate_royal_claim(self,player_index:int):
        "In Birdsong, may discard this to score one point per clearing you rule."
        logger.debug(f"\t{ID_TO_PLAYER[player_index]} activates Royal Claim...")
        points = sum((i == player_index) for i in self.board.get_rulers())
        self.change_score(player_index,points)
        self.discard_from_persistent(player_index, CID_ROYAL_CLAIM)
        self.turn_log.change_plr_pers_used(player_index, CID_TO_PERS_INDEX[CID_ROYAL_CLAIM])
    
    def activate_stand_and_deliver(self,player_index:int,target_index:int):
        "In Birdsong, may take a random card from another player. That player scores one point."
        logger.debug(f"\t{ID_TO_PLAYER[player_index]} activates Stand and Deliver on {ID_TO_PLAYER[target_index]}...")
        target_p_hand = self.players[target_index].hand
        chosen_i = random.randint(0,len(target_p_hand) - 1)
        chosen_card = target_p_hand.pop(chosen_i)
        self.turn_log.change_plr_hand_size(target_index,-1)
        logger.debug(f"\t\tCard Taken: {chosen_card.name}")

        self.players[player_index].hand.append(chosen_card)
        self.turn_log.change_plr_hand_size(player_index,1)
        self.change_score(target_index,1)
        self.turn_log.change_plr_pers_used(player_index, CID_TO_PERS_INDEX[CID_STAND_AND_DELIVER],target_index)

    def activate_tax_collector(self,player_index:int,clearing_index:int):
        "Once in Daylight, may remove one of your warriors from the map to draw a card."
        logger.debug(f"\t{ID_TO_PLAYER[player_index]} activates Tax Collector...")
        self.board.place_warriors(player_index,-1,clearing_index)
        self.turn_log.change_clr_warriors(clearing_index,player_index,-1)
        
        self.players[player_index].change_num_warriors(1)
        self.turn_log.change_plr_warrior_supply(player_index,1)

        self.draw_cards(player_index,1)
        self.turn_log.change_plr_pers_used(player_index, CID_TO_PERS_INDEX[CID_TAX_COLLECTOR])
        if player_index == PIND_MARQUISE and self.keep_is_up():
            self.field_hospitals.append((1,self.board.clearings[clearing_index].suit))

    def activate_better_burrow(self,player_index:int,target_index:int):
        "At start of Birdsong, you and another player draw a card."
        logger.debug(f"\t{ID_TO_PLAYER[player_index]} activates Better Burrow Bank...")
        self.draw_cards(player_index,1)
        self.draw_cards(target_index,1)
        self.turn_log.change_plr_pers_used(player_index, CID_TO_PERS_INDEX[CID_BBB],target_index)
    
    def activate_codebreakers(self,player_index:int,target_index:int):
        "Once in Daylight, may look at another player's hand."
        logger.debug(f"\t{ID_TO_PLAYER[player_index]} activates Codebreakers on {ID_TO_PLAYER[target_index]}")
        target_hand = np.zeros((42,3))
        for c in self.players[target_index].hand:
            cid = c.id
            if target_hand[cid][0] == 1:
                target_hand[cid][0] = 0
                target_hand[cid][1] = 1
            elif target_hand[cid][1] == 1:
                target_hand[cid][1] = 0
                target_hand[cid][2] = 1
            else:
                target_hand[cid][0] = 1
        if player_index == PIND_MARQUISE:
            self.marquise_seen_hands[target_index][0] = target_hand
        elif player_index == PIND_EYRIE:
            self.eyrie_seen_hands[target_index][0] = target_hand
        elif player_index == PIND_ALLIANCE:
            self.alliance_seen_hands[target_index][0] = target_hand
        elif player_index == PIND_VAGABOND:
            self.vagabond_seen_hands[target_index][0] = target_hand
        self.turn_log.change_plr_pers_used(player_index, CID_TO_PERS_INDEX[CID_CODEBREAKERS], target_index)

    def can_craft(self,card:Card,player:Player):
        "Returns True only if the current player can currently craft the given card. Checks the remaining crafting power."
        r = card.crafting_recipe
        if max(r) == 0:
            return False # card cannot be crafted at all
        if card.is_persistent and card.id in {c.id for c in player.persistent_cards}:
            return False # we already have this card crafted
        if card.crafting_item != ITEM_NONE and self.available_items[card.crafting_item] == 0:
            return False # enough of this item has been crafted already
        
        wild = r[3]
        if wild > 0:
            # wildcards can use any crafting power
            return sum(self.remaining_craft_power) >= wild
        for i in range(3):
            if r[i] > self.remaining_craft_power[i]:
                return False # not enough crafting power left
        return True
    
    def get_craftable_ids(self,player:Player):
        "Returns a list of every card ID that the given player can currently craft, given the remaining crafting power."
        return [c.id for c in player.hand if self.can_craft(c,player)]

    def get_royal_claim_craft_ids(self):
        """
        Returns a list of integers: each is the Action ID of a valid way
        that Royal Claim can be crafted with the current remaining crafting power.
        """
        ans = []
        for aid,recipe in AID_CRAFT_RC_MAPPING.items():
            if all(self.remaining_craft_power[i] >= recipe[i] for i in range(3)):
                ans.append(aid)
        return ans

    def place_marquise_wood(self,mplayer:Marquise):
        """
        Places 1 wood token at each Sawmill, unless there is not enough wood.

        Returns True only if all of the wood could be automatically placed, False otherwise.
        """
        sawmill_counts = self.board.get_total_building_counts(PIND_MARQUISE,BIND_SAWMILL)
        wood_in_store = mplayer.get_num_tokens_in_store(TIND_WOOD)
        if wood_in_store == 0 or sum(sawmill_counts) == 0:
            logger.debug("\tNo wood to be placed / No sawmills to generate wood at")
            return True
        # there is at least 1 wood and at least 1 sawmill to place at
        if sum(sawmill_counts) > wood_in_store:
            logger.debug("\tWood cannot be placed automatically...")
            self.available_wood_spots = sawmill_counts
            return False
        # place a wood token for each sawmill at each clearing with a sawmill
        logger.debug(f"\t{sum(sawmill_counts)} Wood can be placed automatically...")
        clearings_with_sawmill_ids = {i for i,n in enumerate(sawmill_counts) if n > 0}
        for i in clearings_with_sawmill_ids:
            n_sawmills = sawmill_counts[i]
            while n_sawmills:
                mplayer.change_num_tokens(TIND_WOOD, -1)
                self.board.place_token(PIND_MARQUISE,TIND_WOOD,i)
                self.turn_log.change_clr_tokens(i,PIND_MARQUISE,TIND_WOOD,1)
                n_sawmills -= 1
        return True

    def place_marquise_warriors(self,mplayer:Marquise):
        """
        Places 1 Marquise warrior at each Recruiter, unless there is not enough warriors in store.

        Returns True only if all of the warriors could be automatically placed, False otherwise.
        """
        recruiter_counts = self.board.get_total_building_counts(PIND_MARQUISE,BIND_RECRUITER)
        if sum(recruiter_counts) > mplayer.warrior_storage:
            logger.debug("\tWarriors cannot be recruited automatically...")
            self.available_recruiters = recruiter_counts
            return False
        # place a warrior for each recruiter at each clearing with a recruiter
        logger.debug("\tWarriors can be recruited automatically...")
        clearings_with_recruiter_ids = {i for i,n in enumerate(recruiter_counts) if n > 0}
        for i in clearings_with_recruiter_ids:
            n_recruiters = recruiter_counts[i]
            mplayer.change_num_warriors(-n_recruiters)
            self.turn_log.change_plr_warrior_supply(PIND_MARQUISE,-n_recruiters)
            self.board.place_warriors(PIND_MARQUISE,n_recruiters,i)
            self.turn_log.change_clr_warriors(i,PIND_MARQUISE,n_recruiters)
        return True
    
    def place_marquise_building(self,mplayer:Marquise,building_index:int,clearing_index:int):
        """
        Tries to automatically place the given building in the given clearing. Assumes that
        there is an open building slot, that the Marquise player can build this building,
        that there is enough connected wood, etc. Places the given building / adjusts points
        and everything automatically before checking wood.

        Returns True only if there was no choice in where to spend wood:
        - All the wood connected would have to be spent
        - All the available wood is in only one clearing
        """
        # place the building
        wood_cost,points_scored = mplayer.update_from_building_placed(building_index)
        logger.debug(f"Building a {ID_TO_MBUILD[building_index]} in clearing {clearing_index}")
        logger.debug(f"\tWood Cost: {wood_cost}")
        self.board.place_building(PIND_MARQUISE,building_index,clearing_index)
        self.turn_log.change_clr_building(clearing_index,PIND_MARQUISE,building_index,1)
        self.change_score(PIND_MARQUISE,points_scored)
        if wood_cost == 0:
            return True
        
        usable_wood = self.board.get_wood_to_build_in(clearing_index)
        one_clearing_id = -1
        total_usable = 0
        for i,amount in enumerate(usable_wood):
            if amount > 0:
                total_usable += amount
                if one_clearing_id == -1:
                    one_clearing_id = i
                else:
                    one_clearing_id = None
        # if there is only one clearing with wood, just take the cost from that clearing
        if one_clearing_id is not None:
            logger.debug(f"\tRemoving {wood_cost} wood solely from clearing {one_clearing_id}...")
            for i in range(wood_cost):
                self.board.clearings[one_clearing_id].remove_token(PIND_MARQUISE,TIND_WOOD)
                self.turn_log.change_clr_tokens(one_clearing_id,PIND_MARQUISE,TIND_WOOD,-1)
            mplayer.change_num_tokens(TIND_WOOD,wood_cost)
            return True
        # if we have exactly enough wood to pay, then use up all of the usable wood
        if total_usable == wood_cost:
            logger.debug(f"\tRemoving all usable wood to pay...")
            for i,amount in enumerate(usable_wood):
                while amount > 0:
                    self.board.clearings[i].remove_token(PIND_MARQUISE,TIND_WOOD)
                    self.turn_log.change_clr_tokens(i,PIND_MARQUISE,TIND_WOOD,-1)
                    amount -= 1
            mplayer.change_num_tokens(TIND_WOOD,wood_cost)
            return True
        # wood is in more than 1 clearing and we have too much, so
        # we have to choose where to take it from
        logger.debug("\t\tWood cannot be automatically taken")
        # logger.debug(f"\t\tNeed to spend {wood_cost} from: {usable_wood}")
        self.available_wood_spots = usable_wood
        self.remaining_wood_cost = wood_cost
        return False
    
    def get_marquise_building_actions(self,mplayer:Marquise):
        """
        Finds all of the AID's for building each building for the Marquise.
        Returns a list of integers.
        """
        ans = []
        usable_wood_per_clearing = self.board.get_wood_available()
        empty_slots = self.board.get_empty_building_slot_counts()
        ids = [(BIND_SAWMILL,AID_BUILD1),(BIND_WORKSHOP,AID_BUILD2),(BIND_RECRUITER,AID_BUILD3)]
        for bid,aid in ids:
            n_left_to_build = mplayer.get_num_buildings_on_track(bid)
            if n_left_to_build > 0:
                building_cost = mplayer.building_costs[6 - n_left_to_build]
                ans += [aid+i for i,amount in enumerate(usable_wood_per_clearing) if (amount >= building_cost and empty_slots[i] > 0)]
        return ans
    
    def get_marquise_overwork_actions(self,mplayer:Marquise):
        "Returns a list of integers: All the Overwork AID's for the Marquise."
        ans = []
        seen = set()
        sawmill_clearings = [i for i,count in enumerate(self.board.get_total_building_counts(PIND_MARQUISE,BIND_SAWMILL)) if (count > 0)]
        for card in mplayer.hand:
            if card.id not in seen:
                seen.add(card.id)
                for i in sawmill_clearings:
                    clearing = self.board.clearings[i]
                    if (card.suit == SUIT_BIRD) or (clearing.suit == card.suit):
                        ans.append(AID_OVERWORK + i + card.id*12)
        return ans
    
    def reduce_decree_count(self,decree_index:int,suit:int):
        """
        Marks a certain required action on the current decree as completed
        by subtracting one from the counter, given the suit of the clearing.

        Prioritizes the exact suit before counting an action as a bird action.
        """
        if self.remaining_decree[decree_index][suit] > 0:
            logger.debug(f"\tFulfilled {ID_TO_SUIT[suit]} {ID_TO_DECREE[decree_index]} requirement on Decree")
            self.remaining_decree[decree_index][suit] -= 1
        else:
            logger.debug(f"\tFulfilled Bird {ID_TO_DECREE[decree_index]} requirement on Decree")
            self.remaining_decree[decree_index][SUIT_BIRD] -= 1
    
    def setup_decree_counter(self,eplayer:Eyrie):
        "Sets up the self.remaining_decree object given the eplayer's decree."
        logger.debug("\tTallying required decree...")
        for decree_index,card_list in eplayer.decree.items():
            for card in card_list:
                self.remaining_decree[decree_index][card.suit] += 1

    def get_eyrie_decree_add_actions(self,eplayer:Eyrie):
        "Returns a list of int: All of the legal Add to Decree AID's for the Eyrie."
        ans = []
        ids = [AID_DECREE_RECRUIT,AID_DECREE_MOVE,AID_DECREE_BATTLE,AID_DECREE_BUILD]
        for card in eplayer.hand:
            if (card.suit == SUIT_BIRD) and (self.eyrie_bird_added == 1):
                continue
            ans += [i+card.id for i in ids]
        return ans

    def get_decree_resolving_actions(self,eplayer:Eyrie):
        """
        Assuming that there are still actions to do in the decree, finds all
        of the AID's that will help resolve the current step in the decree, if
        any are currently possible. 
        """
        ans = []
        valid_suits = set()
        if sum(self.remaining_decree[DECREE_RECRUIT]) > 0:
            if eplayer.warrior_storage == 0:
                return ans
            remaining = self.remaining_decree[DECREE_RECRUIT]
            if remaining[SUIT_BIRD] > 0:
                valid_suits = {SUIT_MOUSE,SUIT_RABBIT,SUIT_FOX}
            else:
                for i in range(3):
                    if remaining[i] > 0:
                        valid_suits.add(i)
            roost_clearing_indices = [i for i,count in enumerate(self.board.get_total_building_counts(PIND_EYRIE,BIND_ROOST)) if (count > 0)]
            for i in roost_clearing_indices:
                if self.board.clearings[i].suit in valid_suits:
                    ans.append(i + AID_CHOOSE_CLEARING)

        elif sum(self.remaining_decree[DECREE_MOVE]) > 0:
            remaining = self.remaining_decree[DECREE_MOVE]
            if remaining[SUIT_BIRD] > 0:
                valid_suits = {SUIT_MOUSE,SUIT_RABBIT,SUIT_FOX}
            else:
                for i in range(3):
                    if remaining[i] > 0:
                        valid_suits.add(i)
            ans += self.board.get_legal_move_actions(PIND_EYRIE,valid_suits)

        elif sum(self.remaining_decree[DECREE_BATTLE]) > 0:
            remaining = self.remaining_decree[DECREE_BATTLE]
            if remaining[SUIT_BIRD] > 0:
                valid_suits = {SUIT_MOUSE,SUIT_RABBIT,SUIT_FOX}
            else:
                for i in range(3):
                    if remaining[i] > 0:
                        valid_suits.add(i)
            for enemy_id,battle_aid in [(PIND_MARQUISE,AID_BATTLE_MARQUISE),(PIND_ALLIANCE,AID_BATTLE_ALLIANCE),(PIND_VAGABOND,AID_BATTLE_VAGABOND)]:
                possible_battle_clearings = [i for i,x in enumerate(self.board.get_possible_battles(PIND_EYRIE,enemy_id)) if x]
                for i in possible_battle_clearings:
                    if self.board.clearings[i].suit in valid_suits:
                        ans.append(i + battle_aid)

        elif sum(self.remaining_decree[DECREE_BUILD]) > 0:
            if eplayer.get_num_buildings_on_track(BIND_ROOST) == 0:
                return ans
            remaining = self.remaining_decree[DECREE_BUILD]
            if remaining[SUIT_BIRD] > 0:
                valid_suits = {SUIT_MOUSE,SUIT_RABBIT,SUIT_FOX}
            else:
                for i in range(3):
                    if remaining[i] > 0:
                        valid_suits.add(i)
            for i in range(12):
                c = self.board.clearings[i]
                if (c.suit in valid_suits and
                        c.is_ruler(PIND_EYRIE) and 
                        c.get_num_buildings(PIND_EYRIE,BIND_ROOST) == 0 and 
                        c.get_num_empty_slots() > 0 and 
                        c.can_place(PIND_EYRIE)):
                    ans.append(i + AID_BUILD1)
        return ans
    
    def get_revolt_actions(self,aplayer:Alliance):
        "Returns a list of all valid revolt AIDs for the Alliance."
        ans = []
        for i,clearing in enumerate(self.board.clearings):
            csuit = clearing.suit
            if (clearing.is_sympathetic() and
                    aplayer.get_num_buildings_on_track(csuit) > 0 and
                    aplayer.supporter_suit_counts[csuit] >= 2):
                ans.append(i + AID_REVOLT)
        return ans
    
    def get_spread_sym_actions(self,aplayer:Alliance):
        "Returns a list of all valid 'spread sympathy' AIDs for the Alliance."
        tokens_left = aplayer.get_num_tokens_in_store(TIND_SYMPATHY)
        next_cost = Alliance.sympathy_costs[10 - tokens_left]
        # no spreading sympathy possible if
        # - The alliance do not have enough supporters for any token placement
        # - The alliance have placed all of their sympathy tokens
        if (len(aplayer.supporters) < next_cost or
                tokens_left == 0):
            return []
        # finding all clearings the alliance could spread sympathy to
        sympathetic_clearings = {c.id for c in self.board.clearings if c.is_sympathetic()}
        if len(sympathetic_clearings) == 0:
            clearings_to_check = set(range(12))
        else:
            clearings_to_check = set()
            for i in sympathetic_clearings:
                adj_cids = self.board.clearings[i].adjacent_clearing_ids
                clearings_to_check.update(adj_cids - sympathetic_clearings)
        # finding which AIDs are legal
        ans = []
        for i in clearings_to_check:
            clearing = self.board.clearings[i]
            real_cost = next_cost + int(clearing.has_martial_law())
            if (aplayer.supporter_suit_counts[clearing.suit] >= real_cost and clearing.can_place(PIND_ALLIANCE)):
                ans.append(i + AID_SPREAD_SYMPATHY)
        return ans
    
    def get_train_actions(self,aplayer:Alliance):
        "Get a list of all of the AIDs for training new officers for the Alliance."
        if aplayer.warrior_storage == 0:
            return []
        valid_suits = set()
        for bid in range(3):
            if aplayer.get_num_buildings_on_track(bid) == 0:
                valid_suits.add(bid)
                valid_suits.add(SUIT_BIRD)
        if len(valid_suits) == 0:
            return []
        ans = set()
        for c in aplayer.hand:
            if c.suit in valid_suits:
                ans.add(c.id + AID_TRAIN)
        return list(ans)
    
    def get_dom_payment_actions(self,player:Player):
        "Get a list of all of the AIDs for spending a card to take an available dominance card."
        if max(self.available_dominances) == 0:
            return []
        ans = set()
        avdoms = [i for i,available in enumerate(self.available_dominances) if available == 1]
        for dom_suit in avdoms:
            for c in player.hand:
                if c.suit == dom_suit or c.suit == SUIT_BIRD:
                    ans.add(AID_TAKE_DOM + dom_suit * 42 + c.id)
        return list(ans)
    
    def get_dom_activation_actions(self,player:Player,player_index:int):
        "Get a list of all the valid AIDs for the player to activate a dominance card in their hand."
        if (self.active_dominances[player_index] is not None) or (self.victory_points[player_index] < 10):
            return []
        ans = []

        if player_index == PIND_VAGABOND:
            # find all players with the lowest score among them
            # but ignore those who activated a dominance already
            lowest_score = 31
            lowest_player_ids = []
            for pid in range(3):
                pscore = self.victory_points[pid]
                if pscore != -1:
                    if pscore < lowest_score:
                        lowest_score = pscore
                        lowest_player_ids = [pid]
                    elif pscore == lowest_score:
                        lowest_player_ids.append(pid)
            # find all legal coalition actions
            for c in player.hand:
                if c.is_dominance:
                    ans += [(AID_ACTIVATE_COALITION + 3*c.suit + pid) for pid in lowest_player_ids]
        else:
            # Find all dominance cards that can be activated
            for c in player.hand:
                if c.is_dominance:
                    ans.append(AID_ACTIVATE_DOM + c.suit)

        return ans
    
    def has_dominance_win(self,player_index:int):
        "Checks if the player with the given index can win now by their currently activated dominance."
        dom_suit = self.active_dominances[player_index].suit
        clearing_rulers = self.board.get_rulers()
        if dom_suit == SUIT_BIRD:
            for i,j in [(0,11),(2,8)]:
                if clearing_rulers[i] == player_index and clearing_rulers[j] == player_index:
                    logger.debug(f">>> The {ID_TO_PLAYER[player_index]} rule clearings {i} and {j}!")
                    return True
            logger.debug(f">>> Bird Dominance NOT fulfilled for {ID_TO_PLAYER[player_index]}...")
            return False
        # not bird dominance
        dom_suit_clearings = CLEARING_SUITS[dom_suit]
        if sum([(clearing_rulers[i] == player_index) for i in dom_suit_clearings]) >= 3:
            logger.debug(f">>> The {ID_TO_PLAYER[player_index]} rule enough clearings!")
            return True
        else:
            logger.debug(f">>> {ID_TO_SUIT[dom_suit]} Dominance NOT fulfilled for {ID_TO_PLAYER[player_index]}...")
            return False
    
    def adjust_reward_for_dom_activation(self,player_index:int):
        """
        Adjusts the 'points_scored_this_action' to signal to a player
        activating a dominance how good their play was.

        Since they no longer can score points, we must come up with some
        way to indicate (compared to everyone else's score) how close
        they are to winning/how good of a move it was to activate the
        dominance in their position. I will attempt to emulate this
        effect by rewarding them larger amounts for playing a dominance
        card while being closer to having its winning condition fulfilled.
        Playing one without having the condition fulfilled will either
        give no reward or a negative reward, hopefully signalling how
        their chances of winning have changed.
        """
        dom_suit = self.active_dominances[player_index].suit
        clearing_rulers = self.board.get_rulers()
        change = 0
        if dom_suit == SUIT_BIRD:
            owned_corners = [(clearing_rulers[i] == player_index) for i in (0,2,8,11)]
            if sum(owned_corners) == 0:
                change = -2
            elif sum(owned_corners) == 1:
                change = -1
            elif sum(owned_corners) == 4:
                change = 8
            elif sum(owned_corners) == 3:
                change = 6
            # owns two corners, find if they're opposite
            elif (owned_corners[0] and owned_corners[3]) or (owned_corners[1] and owned_corners[2]):
                change = 5
            else: # own two corners, but not opposite
                change = 0
        else:
            # not bird dominance
            dom_suit_clearings = CLEARING_SUITS[dom_suit]
            num_clearings_owned = sum([(clearing_rulers[i] == player_index) for i in dom_suit_clearings])
            if num_clearings_owned == 4:
                change = 8
            elif num_clearings_owned == 3:
                change = 6
            elif num_clearings_owned == 2:
                change = 2
            elif num_clearings_owned == 1:
                change = 0
            else: # none of that type owned, yikes
                change = -2
        for i in range(N_PLAYERS):
            if i == player_index:
                self.points_scored_this_action[i] += change * GAME_SCALAR
            else:
                self.points_scored_this_action[i] += (-change / (N_PLAYERS - 1)) * GAME_SCALAR

    def adjust_reward_for_dom_turn(self,player_index:int):
        """
        Adjusts the 'points_scored_this_action' to signal to dominance
        players how good or bad their position is at the end of one
        of their turns. 
        
        Since they can no longer score points, they will
        instead be rewarded/punished based on how close they are to
        obtaining a domininance win. The more clearings of the correct
        type they rule, the higher their reward will be for their turn,
        hopefully signifying that they have done the right thing that turn
        by obtaining or holding enough presence in certain clearings.
        """
        dom_suit = self.active_dominances[player_index].suit
        clearing_rulers = self.board.get_rulers()
        change = 0
        if dom_suit == SUIT_BIRD:
            owned_corners = [(clearing_rulers[i] == player_index) for i in (0,2,8,11)]
            if sum(owned_corners) == 0:
                change = -1
            elif sum(owned_corners) == 1:
                change = 0
            elif sum(owned_corners) == 4:
                change = 6
            elif sum(owned_corners) == 3:
                change = 4
            # owns two corners, find if they're opposite
            elif (owned_corners[0] and owned_corners[3]) or (owned_corners[1] and owned_corners[2]):
                change = 3
            else: # own two corners, but not opposite
                change = 0
        else:
            # not bird dominance
            dom_suit_clearings = CLEARING_SUITS[dom_suit]
            num_clearings_owned = sum([(clearing_rulers[i] == player_index) for i in dom_suit_clearings])
            if num_clearings_owned == 4:
                change = 6
            elif num_clearings_owned == 3:
                change = 4
            elif num_clearings_owned == 2:
                change = 0
            elif num_clearings_owned == 1:
                change = -1
            else: # none of that type owned, yikes
                change = -1
        for i in range(N_PLAYERS):
            if i == player_index:
                self.points_scored_this_action[i] += change * GAME_SCALAR
            else:
                self.points_scored_this_action[i] += (-change / (N_PLAYERS - 1)) * GAME_SCALAR
    
    def revolt_helper(self,aplayer:Alliance,clearing_index:int):
        """
        Assumes that the two matching supporters have been spent already.
        Helps with making a revolt happen in the given clearing:
        - Removes all enemy pieces there (adds Field Hospitals if needed)
        - Places matching base
        - Places warriors (# = # of sympathetic clearings)
        - Recruits 1 officer
        This scores points for removing buildings/tokens and will recruit as many
        as possible if the alliance warrior supply runs out.
        """
        logger.debug(f"\tREVOLT started by the Woodland Alliance in clearing {clearing_index}!")
        # Remove all enemy pieces
        clearing = self.board.clearings[clearing_index]
        for faction_i in {j for j in range(N_PLAYERS) if j != PIND_ALLIANCE}:
            player = self.players[faction_i]
            foo = clearing.get_num_warriors(faction_i)
            logger.debug(f"\t\tRemoving {foo} {ID_TO_PLAYER[faction_i]} warriors in clearing {clearing_index}")
            clearing.change_num_warriors(faction_i,-foo)
            self.turn_log.change_clr_warriors(clearing_index,faction_i,-foo)
            player.change_num_warriors(foo)
            self.turn_log.change_plr_warrior_supply(faction_i,foo)

            if (faction_i == PIND_MARQUISE and
                    foo > 0 and
                    self.keep_is_up()):
                self.field_hospitals.append((foo,clearing.suit))
            while len(clearing.buildings[faction_i]) > 0:
                foo = clearing.buildings[faction_i].pop()
                player.change_num_buildings(foo,1)
                self.turn_log.change_clr_building(clearing_index,faction_i,foo,-1)
                logger.debug(f"\t\tDestroyed a building of {ID_TO_PLAYER[faction_i]}")
                self.change_score(PIND_ALLIANCE,1)
            while len(clearing.tokens[faction_i]) > 0:
                foo = clearing.tokens[faction_i].pop()
                player.change_num_tokens(foo,1)
                self.turn_log.change_clr_tokens(clearing_index,faction_i,foo,-1)
                logger.debug(f"\t\tDestroyed a token of {ID_TO_PLAYER[faction_i]}")
                self.change_score(PIND_ALLIANCE,1)
        # place matching base
        csuit = clearing.suit
        aplayer.change_num_buildings(csuit,-1)
        self.board.place_building(PIND_ALLIANCE,csuit,clearing_index)
        self.turn_log.change_clr_building(clearing_index,PIND_ALLIANCE,csuit,1)
        # place recruits
        n_recruits = min(self.board.get_num_sympathetic(csuit),aplayer.warrior_storage)
        aplayer.change_num_warriors(-n_recruits)
        self.turn_log.change_plr_warrior_supply(PIND_ALLIANCE,-n_recruits)
        self.board.place_warriors(PIND_ALLIANCE,n_recruits,clearing_index)
        self.turn_log.change_clr_warriors(clearing_index,PIND_ALLIANCE,n_recruits)
        # add 1 officer
        if aplayer.warrior_storage > 0:
            logger.debug("\tRecruiting 1 Officer...")
            aplayer.change_num_warriors(-1)
            self.turn_log.change_plr_warrior_supply(PIND_ALLIANCE,-1)
            aplayer.num_officers += 1

    def base_removal_helper(self,aplayer:Alliance,suit:int):
        "Does the work as if a base of the given type was removed from the board."
        logger.debug(f"> A {ID_TO_SUIT[suit]} Base was removed, losing supporters...")
        aplayer.change_num_buildings(suit,1)
        ids_to_discard = []
        for c in aplayer.supporters:
            if c.suit == SUIT_BIRD or c.suit == suit:
                ids_to_discard.append(c.id)
        for i in ids_to_discard:
            self.discard_from_supporters(aplayer,i)
        n = aplayer.num_officers
        n_to_remove = n//2 if (n % 2 == 0) else (n//2) + 1
        logger.debug(f"\tThe Alliance loses {n_to_remove} officer(s)")
        aplayer.num_officers -= n_to_remove
        aplayer.change_num_warriors(n_to_remove)
        self.turn_log.change_plr_warrior_supply(PIND_ALLIANCE,n_to_remove)
    
    def get_vagabond_move_actions(self,vplayer:Vagabond,base_boots:int):
        """
        Returns a list of AIDs for each currently valid move possible
        for the Vagabond player. 
        
        Given their current position, they can only move from a clearing
        or forest to an adjacent clearing. They must exhaust 'base_boot' boots to do so,
        unless the target clearing has any Hostile warriors, which then
        increases the cost by 1 boot.
        """
        ans = []

        # check for available boots
        if vplayer.has_exhaustable(ITEM_BOOT,base_boots+1):
            can_move_hostile = True
        elif vplayer.has_exhaustable(ITEM_BOOT,base_boots):
            can_move_hostile = False
        else:
            return ans
        
        # get possible clearings to move to
        start_loc = vplayer.location
        if start_loc <= 11:
            possible_clearings = self.board.clearings[start_loc].adjacent_clearing_ids
        else:
            possible_clearings = self.board.forests[start_loc - 12].adjacent_clearing_ids
        
        hostile_ids = {i for i,rel in vplayer.relationships.items() if rel == 0}
        
        for end_i in possible_clearings:
            end_c = self.board.clearings[end_i]
            # check if moving to hostile clearing and if we are able to
            if (not can_move_hostile and 
                    any(end_c.get_num_warriors(p_index) > 0 for p_index in hostile_ids)):
                continue
            ans.append(AID_VB_MOVE + 19*start_loc + end_i)

        return ans
    
    def get_vagabond_clearing_actions(self,vplayer:Vagabond):
        """
        Given that the Vagabond is located in a clearing, returns a
        list of all legal AIDs possible for:
        - Battling
        - Exploring
        - Aiding
        - Questing
        - Striking
        - Crafting
        - Special Actions (Thief,Tinker)
        
        All of the above require the VB to be in a clearing.
        """
        ans = []
        loc = vplayer.location
        clearing = self.board.clearings[loc]
        has_exhaustable = False

        # Battle (must have sword to use/be in a clearing)
        if vplayer.has_exhaustable(ITEM_SWORD):
            has_exhaustable = True
            for enemy_id,battle_aid in [(PIND_EYRIE,AID_BATTLE_EYRIE),(PIND_MARQUISE,AID_BATTLE_MARQUISE),(PIND_ALLIANCE,AID_BATTLE_ALLIANCE)]:
                if clearing.can_start_battle(PIND_VAGABOND,enemy_id):
                    ans.append(loc + battle_aid)

        # Explore / Special Action
        if vplayer.has_exhaustable(ITEM_TORCH):
            has_exhaustable = True
            if clearing.num_ruins > 0:
                ans.append(AID_EXPLORE)

            if vplayer.chosen_character == CHAR_THIEF:
                for pid in range(3):
                    if clearing.has_presence(pid) and len(self.players[pid].hand) > 0:
                        ans.append(AID_THIEF_ABILITY + pid)
            elif vplayer.chosen_character == CHAR_TINKER:
                card_ids_in_discard = np.where(np.any(self.discard_array!=0, 1))[0]
                for cid in card_ids_in_discard:
                    csuit = self.deck.deck_comp[cid][0].suit
                    if csuit == clearing.suit or csuit == SUIT_BIRD:
                        ans.append(AID_TINKER_ABILITY + cid)
        # Quest
        for qcard in self.active_quests:
            if (qcard.suit == clearing.suit and 
                    all(vplayer.has_exhaustable(i,a) for i,a in qcard.requirements.items())):
                has_exhaustable = True
                ans.append(AID_COMPLETE_QUEST + 2*qcard.id)
                ans.append(AID_COMPLETE_QUEST + 2*qcard.id + 1)
        
        # Strike
        if vplayer.has_exhaustable(ITEM_CROSSBOW):
            has_exhaustable = True
            # Marquise
            if clearing.get_num_warriors(PIND_MARQUISE) > 0:
                ans.append(AID_STRIKE)
            else:
                for bid in range(3):
                    if clearing.get_num_buildings(PIND_MARQUISE,bid) > 0:
                        ans.append(AID_STRIKE + 1 + bid)
                for tid in range(2):
                    if clearing.get_num_tokens(PIND_MARQUISE,tid) > 0:
                        ans.append(AID_STRIKE + 4 + tid)
            # Eyrie
            if clearing.get_num_warriors(PIND_EYRIE) > 0:
                ans.append(AID_STRIKE + 6)
            elif clearing.get_num_buildings(PIND_EYRIE) > 0:
                ans.append(AID_STRIKE + 7)
            # Alliance
            if clearing.get_num_warriors(PIND_ALLIANCE) > 0:
                ans.append(AID_STRIKE + 8)
            else:
                for bid in range(3):
                    if clearing.get_num_buildings(PIND_ALLIANCE,bid) > 0:
                        ans.append(AID_STRIKE + 9 + bid)
                if clearing.get_num_tokens(PIND_ALLIANCE) > 0:
                    ans.append(AID_STRIKE + 12)
        
        # Craft
        self.remaining_craft_power = [0,0,0]
        self.remaining_craft_power[clearing.suit] = vplayer.satchel_undamaged.count((ITEM_HAMMER,0))
        logger.debug(f"\tCrafting power: {self.remaining_craft_power}")
        ans += list({i+AID_CRAFT_CARD for i in self.get_craftable_ids(vplayer)})

        if not has_exhaustable:
            has_exhaustable = vplayer.has_any_exhaustable()

        # Aid
        # must be able to exhaust an item and have a matching card to give
        if (has_exhaustable and
                vplayer.has_suit_in_hand(clearing.suit)):
            aid_targets = [player_i for player_i in range(3) if clearing.has_presence(player_i)]
            aid_cards = [card.id for card in vplayer.hand if (card.suit == SUIT_BIRD or card.suit == clearing.suit)]
            ans += [(AID_START_AIDING + 42*pi + ci) for pi in aid_targets for ci in aid_cards]
        
        return ans

    # GAME ADVANCEMENT
    def advance_game(self):
        """
        Assumes that an action has just been resolved, and advances the game
        to the next required choice, skipping over steps / stages where
        there is no choice (as much as possible).
        """
        if self.phase == self.PHASE_SETUP_MARQUISE:
            if self.phase_steps == 0:
                logger.debug(f"\t\t--- GAME START --- Turn Order: {[ID_TO_PLAYER[i] for i in self.turn_order]}")
                return [i+AID_CHOOSE_CLEARING for i,x in enumerate(self.board.clearings) if (x.opposite_corner_id >= 0)]
            if self.phase_steps == 1:
                i = self.players[PIND_MARQUISE].keep_clearing_id
                starting_clearing = self.board.clearings[i]
                self.starting_build_spots = list(starting_clearing.adjacent_clearing_ids) + [i]
                return [x + AID_BUILD1 for x in self.starting_build_spots]
            if self.phase_steps == 2:
                return [x + AID_BUILD2 for x in self.starting_build_spots if (self.board.clearings[x].get_num_empty_slots() > 0)]
            if self.phase_steps == 3:
                return [x + AID_BUILD3 for x in self.starting_build_spots if (self.board.clearings[x].get_num_empty_slots() > 0)]
            if self.phase_steps == 4:
                self.phase = self.PHASE_SETUP_EYRIE
                self.phase_steps = 0
                self.current_player = PIND_EYRIE
                return [x for x in range(AID_CHOOSE_LEADER,AID_CHOOSE_LEADER + 4)]
        
        if self.phase == self.PHASE_SETUP_EYRIE and self.phase_steps == 1:
            
            self.alliance_setup(self.players[PIND_ALLIANCE])
            
            self.phase = self.PHASE_SETUP_VAGABOND
            self.phase_steps = 0
            self.current_player = PIND_VAGABOND
            return [i+AID_CHOOSE_VB_CLASS for i in range(3)]
        
        if self.phase == self.PHASE_SETUP_VAGABOND:
            if self.phase_steps == 1:
                return [i+AID_STARTING_FOREST for i in range(7)]
            if self.phase_steps == 2:
                # START GAME - Random Starting Player?
                logger.debug(f"--- STARTING TURN 1 ---")
                self.phase_steps = 0
                first_player = self.turn_order[0]
                self.current_player = first_player
                if first_player == PIND_MARQUISE: # Marquise start
                    self.phase = self.PHASE_BIRDSONG_MARQUISE
                elif first_player == PIND_EYRIE: # Eyrie start
                    self.phase = self.PHASE_BIRDSONG_EYRIE
                elif first_player == PIND_ALLIANCE: # Alliance start
                    self.phase = self.PHASE_BIRDSONG_ALLIANCE
                elif first_player == PIND_VAGABOND:
                    self.phase = self.PHASE_BIRDSONG_VAGABOND

        if self.current_player == PIND_MARQUISE:
            return self.advance_marquise(self.players[self.current_player])
        if self.current_player == PIND_EYRIE:
            return self.advance_eyrie(self.players[self.current_player])
        elif self.current_player == PIND_ALLIANCE:
            return self.advance_alliance(self.players[self.current_player])
        elif self.current_player == PIND_VAGABOND:
            return self.advance_vagabond(self.players[self.current_player])
    
    def advance_marquise(self,current_player:Marquise):
        "Advances the game assuming we are in the middle of the Marquise's turn."
        if self.phase == self.PHASE_BIRDSONG_MARQUISE:
            if self.phase_steps == 0: # Start of Birdsong
                if self.active_dominances[PIND_MARQUISE] is not None:
                    logger.debug("Checking for dominance victory...")
                    if self.has_dominance_win(PIND_MARQUISE):
                        self.dominance_win = True
                        return [0]
                logger.debug(f"\tResetting for Marquise's Turn...")
                self.reset_for_marquise()
                # can they use BBB?
                if CID_BBB in {c.id for c in current_player.persistent_cards}:
                    logger.debug("Checking for use of BBB...")
                    return [AID_GENERIC_SKIP,AID_CARD_BBB+PIND_EYRIE,AID_CARD_BBB+PIND_ALLIANCE]
                self.phase_steps = 1
            if self.phase_steps == 1:
                if not self.wood_placement_started:
                    self.wood_placement_started = True
                    logger.debug(f"\tProducing wood at sawmills...")
                    wood_placement_done = self.place_marquise_wood(current_player)
                    if not wood_placement_done:
                        logger.debug(f"> Choice exists in placing wood")
                        return [i+AID_CHOOSE_CLEARING for i,count in enumerate(self.available_wood_spots) if (count > 0)]
                    # we are finished placing wood
                    logger.debug(f"Finished placing wood.")
                    self.phase_steps = 2
                else:
                    # we are not finished placing wood
                    # and we still have a choice of placement
                    return [i+AID_CHOOSE_CLEARING for i,count in enumerate(self.available_wood_spots) if (count > 0)]
            if self.phase_steps == 2:
                # can they use Royal Claim or Stand/Deliver?
                unused_pers = {c.id for c in current_player.persistent_cards} - self.persistent_used_this_turn
                ans = []
                if CID_STAND_AND_DELIVER in unused_pers:
                    for pid in (PIND_EYRIE,PIND_ALLIANCE):
                        if len(self.players[pid].hand) > 0:
                            ans.append(AID_CARD_STAND_DELIVER + pid)
                if CID_ROYAL_CLAIM in unused_pers:
                    ans.append(AID_CARD_ROYAL_CLAIM)
                if bool(ans):
                    logger.debug(f"Checking for use of Birdsong Cards...")
                    return [AID_GENERIC_SKIP] + ans
                self.phase_steps = 3
            if self.phase_steps == 3:
                logger.debug(f"--- Moving on to Daylight ---")
                self.phase_steps = 0
                self.phase = self.PHASE_DAYLIGHT_MARQUISE
                
        if self.phase == self.PHASE_DAYLIGHT_MARQUISE:
            if self.phase_steps == 0:
                # can they use Command Warren?
                if CID_COMMAND_WARREN in {c.id for c in current_player.persistent_cards}:
                    ans = []
                    for enemy_id,battle_aid in [(PIND_EYRIE,AID_BATTLE_EYRIE),(PIND_ALLIANCE,AID_BATTLE_ALLIANCE)]:
                        possible_battle_clearings = [i for i,x in enumerate(self.board.get_possible_battles(PIND_MARQUISE,enemy_id)) if x]
                        for i in possible_battle_clearings:
                            ans.append(i + battle_aid)
                    if bool(ans):
                        logger.debug(f"Checking for use of Command Warren...")
                        return [AID_GENERIC_SKIP] + ans
                self.phase_steps = 1
            if self.phase_steps == 1:
                if len(self.remaining_craft_power) == 1:
                    self.remaining_craft_power = self.board.get_crafting_power(PIND_MARQUISE)
                    logger.debug(f"\tCrafting power: {self.remaining_craft_power}")
                # check for crafting
                ans = list({i+AID_CRAFT_CARD for i in self.get_craftable_ids(current_player)})
                if (CID_ROYAL_CLAIM+AID_CRAFT_CARD) in ans:
                    # find all ways to craft royal claim
                    ans.remove(CID_ROYAL_CLAIM+AID_CRAFT_CARD)
                    ans += self.get_royal_claim_craft_ids()
                if bool(ans):
                    # also allow use of daylight cards
                    unused_pers = {c.id for c in current_player.persistent_cards} - self.persistent_used_this_turn
                    if CID_CODEBREAKERS in unused_pers:
                        for pid in (PIND_EYRIE,PIND_ALLIANCE):
                                ans.append(AID_CARD_CODEBREAKERS + pid)
                    if CID_TAX_COLLECTOR in unused_pers:
                        foo = self.board.get_num_warriors(PIND_MARQUISE)
                        ans += [i+AID_CARD_TAX_COLLECTOR for i,amount in enumerate(foo) if (amount > 0)]
                    # check for dominance-related actions
                    ans += self.get_dom_payment_actions(current_player)
                    ans += self.get_dom_activation_actions(current_player, PIND_MARQUISE)
                    logger.debug(f"Checking for Crafting / use of Birdsong Cards...")
                    return [AID_GENERIC_SKIP] + ans
                else:
                    # no crafting possible, so move onto the main phase
                    logger.debug(f"> Moving onto step 2 of Daylight...")
                    self.phase_steps = 2
            while self.phase_steps < 6:
                if self.phase_steps == 2:
                    ans = []
                    # check for persistent cards to use
                    unused_pers = {c.id for c in current_player.persistent_cards} - self.persistent_used_this_turn
                    if CID_CODEBREAKERS in unused_pers:
                        for pid in (PIND_EYRIE,PIND_ALLIANCE):
                            ans.append(AID_CARD_CODEBREAKERS + pid)
                    if CID_TAX_COLLECTOR in unused_pers:
                        foo = self.board.get_num_warriors(PIND_MARQUISE)
                        ans += [i+AID_CARD_TAX_COLLECTOR for i,amount in enumerate(foo) if (amount > 0)]
                    # check for spending bird cards
                    seen = set()
                    for c in current_player.hand:
                        if (c.suit == SUIT_BIRD) and (c.id not in seen):
                            ans.append(BIRD_ID_TO_ACTION[c.id])
                            seen.add(c.id)
                    # dominance-related actions
                    ans += self.get_dom_payment_actions(current_player)
                    ans += self.get_dom_activation_actions(current_player, PIND_MARQUISE)
                    # standard actions
                    if self.marquise_actions > 0:
                        # starting a battle
                        for enemy_id,battle_aid in [(PIND_EYRIE,AID_BATTLE_EYRIE),(PIND_ALLIANCE,AID_BATTLE_ALLIANCE)]:
                            possible_battle_clearings = [i for i,x in enumerate(self.board.get_possible_battles(PIND_MARQUISE,enemy_id)) if x]
                            for i in possible_battle_clearings:
                                ans.append(i + battle_aid)
                        # starting a march
                        ans += self.board.get_legal_move_actions(PIND_MARQUISE,{0,1,2})
                        # recruiting
                        if (not self.recruited_this_turn) and (current_player.warrior_storage > 0) and (current_player.get_num_buildings_on_track(BIND_RECRUITER) < 6):
                            ans.append(AID_RECRUIT)
                        # building
                        ans += self.get_marquise_building_actions(current_player)
                        # overworking
                        if current_player.get_num_tokens_in_store(TIND_WOOD) > 0:
                            ans += self.get_marquise_overwork_actions(current_player)
                    if bool(ans):
                        logger.debug(f"Checking for next Daylight Step 2 action ({self.marquise_actions} actions left)...")
                        return [AID_GENERIC_SKIP] + ans
                    # if we get here, then we are done with the daylight phase
                    logger.debug("\tEnd of Daylight phase (no more actions)")
                    self.phase_steps = 6
                if self.phase_steps == 3: # we are mid-march
                    ans = self.board.get_legal_move_actions(PIND_MARQUISE,{0,1,2})
                    if not bool(ans):
                        self.marquise_moves = 2
                        self.phase_steps = 2
                        logger.debug("\t> No legal moves! March ended...")
                    else:
                        logger.debug("Finding 2nd move of march...")
                        return [AID_GENERIC_SKIP] + ans
                if self.phase_steps == 4: # choosing where to recruit
                    logger.debug(f"Choosing where to recruit...")
                    return [i+AID_CHOOSE_CLEARING for i,count in enumerate(self.available_recruiters) if (count > 0)]
                if self.phase_steps == 5:
                    logger.debug(f"Choosing where to take wood from...")
                    return [i+AID_CHOOSE_CLEARING for i,count in enumerate(self.available_wood_spots) if (count > 0)]
            if self.phase_steps == 6:
                logger.debug("--- Moving on to Evening ---")
                self.phase_steps = 0
                self.phase = self.PHASE_EVENING_MARQUISE
        if self.phase == self.PHASE_EVENING_MARQUISE:
            if self.phase_steps == 0:
                unused_pers = {c.id for c in current_player.persistent_cards} - self.persistent_used_this_turn
                if CID_COBBLER in unused_pers:
                    ans = self.board.get_legal_move_actions(PIND_MARQUISE,{0,1,2})
                    if bool(ans):
                        logger.debug("Checking for use of Cobbler...")
                        return [AID_GENERIC_SKIP] + ans
                self.phase_steps = 1
            # Evening Phase
            if self.phase_steps == 1:
                # if self.outrage_offender is not None:
                #     # prevent drawing cards before paying outrage
                #     return [AID_GENERIC_SKIP]
                self.draw_cards(PIND_MARQUISE,current_player.get_num_cards_to_draw())
                self.phase_steps = 2
            if self.phase_steps == 2 and len(current_player.hand) > 5:
                ans = {c.id+AID_DISCARD_CARD for c in current_player.hand}
                logger.debug("Marquise must discard from hand...")
                return list(ans)
            # turn done!
            if max(self.victory_points) >= 30:
                return [0]
            logger.debug("--- End of Marquise's Turn ---\n")
            self.phase_steps = 0
            self.save_to_history()
            if self.active_dominances[PIND_MARQUISE] is not None:
                self.adjust_reward_for_dom_turn(PIND_MARQUISE)

            self.current_player = self.next_player_index[PIND_MARQUISE]
            if self.current_player == PIND_EYRIE:
                self.phase = self.PHASE_BIRDSONG_EYRIE
                self.outside_turn_this_action = PIND_EYRIE
                return self.advance_eyrie(self.players[PIND_EYRIE])
            elif self.current_player == PIND_ALLIANCE:
                self.phase = self.PHASE_BIRDSONG_ALLIANCE
                self.outside_turn_this_action = PIND_ALLIANCE
                return self.advance_alliance(self.players[PIND_ALLIANCE])
    
    def advance_eyrie(self,current_player:Eyrie):
        "Advances the game assuming we are in the middle of the Eyrie's turn."
        if self.phase == self.PHASE_BIRDSONG_EYRIE:
            if self.phase_steps == 0: # Start of Birdsong
                if self.active_dominances[PIND_EYRIE] is not None:
                    logger.debug("Checking for dominance victory...")
                    if self.has_dominance_win(PIND_EYRIE):
                        self.dominance_win = True
                        return [0]
                logger.debug(f"\tResetting for Eyrie's Turn...")
                self.reset_for_eyrie()
                # can they use BBB?
                if CID_BBB in {c.id for c in current_player.persistent_cards}:
                    logger.debug("Checking for use of BBB...")
                    return [AID_GENERIC_SKIP,AID_CARD_BBB+PIND_MARQUISE,AID_CARD_BBB+PIND_ALLIANCE]
                self.phase_steps = 1
            if self.phase_steps == 1: # drawing emergency card
                if len(current_player.hand) == 0:
                    logger.debug("\tDrawing Emergency Card...")
                    self.draw_cards(PIND_EYRIE,1)
                self.phase_steps = 2
            if self.phase_steps == 2: # adding to decree
                unused_pers = {c.id for c in current_player.persistent_cards} - self.persistent_used_this_turn
                ans = []
                if CID_STAND_AND_DELIVER in unused_pers:
                    for pid in (PIND_MARQUISE,PIND_ALLIANCE):
                        if len(self.players[pid].hand) > 0:
                            ans.append(AID_CARD_STAND_DELIVER + pid)
                if CID_ROYAL_CLAIM in unused_pers:
                    ans.append(AID_CARD_ROYAL_CLAIM)
                ans += self.get_eyrie_decree_add_actions(current_player)
                if not bool(ans):
                    logger.debug("\tCannot add more to decree")
                    self.phase_steps = 3
                elif self.eyrie_cards_added > 0:
                    logger.debug("Choosing whether to add 2nd card to decree...")
                    return [AID_GENERIC_SKIP] + ans
                else:
                    logger.debug("Choosing what to add to decree...")
                    return ans
            if self.phase_steps == 3:
                # setup decree to complete
                self.setup_decree_counter(current_player)
                # a new roost (no roosts on map)
                if current_player.get_num_buildings_on_track(BIND_ROOST) == 7:
                    logger.debug("Activating 'A New Roost'...")
                    # find the clearing(s) with the fewest warriors where a building can be placed
                    ans = []
                    total_warriors = sorted([(sum(self.board.get_num_warriors(fac_i)[i] for fac_i in range(N_PLAYERS)), i) for i in range(12)])
                    empty_slot_counts = self.board.get_empty_building_slot_counts()
                    last_seen = total_warriors[0][0]
                    for count,i in total_warriors:
                        if count > last_seen:
                            if len(ans) > 0:
                                break
                            else:
                                last_seen = count
                        if empty_slot_counts[i] > 0 and self.board.clearings[i].can_place(PIND_EYRIE):
                            ans.append(i + AID_BUILD1)

                    if len(ans) == 1:
                        # automatically place the roost
                        current_player.place_roost()
                        self.board.place_building(PIND_EYRIE, BIND_ROOST, ans[0] - AID_BUILD1)
                        self.turn_log.change_clr_building(ans[0] - AID_BUILD1,PIND_EYRIE,BIND_ROOST,1)
                        n_warriors = min(3,current_player.warrior_storage)
                        self.board.place_warriors(PIND_EYRIE, n_warriors, ans[0] - AID_BUILD1)
                        self.turn_log.change_clr_warriors(ans[0] - AID_BUILD1,PIND_EYRIE,n_warriors)
                        current_player.change_num_warriors(-n_warriors)
                        self.turn_log.change_plr_warrior_supply(PIND_EYRIE,-n_warriors)
                    else:
                        logger.debug("Choosing where to add new roost...")
                        return ans
                self.phase_steps = 4
            if self.phase_steps == 4:
                # can they use Royal Claim or Stand/Deliver?
                unused_pers = {c.id for c in current_player.persistent_cards} - self.persistent_used_this_turn
                ans = []
                if CID_STAND_AND_DELIVER in unused_pers:
                    for pid in (PIND_MARQUISE,PIND_ALLIANCE):
                        if len(self.players[pid].hand) > 0:
                            ans.append(AID_CARD_STAND_DELIVER + pid)
                if CID_ROYAL_CLAIM in unused_pers:
                    ans.append(AID_CARD_ROYAL_CLAIM)
                if bool(ans):
                    logger.debug(f"Checking for use of Birdsong Cards...")
                    return [AID_GENERIC_SKIP] + ans
                self.phase_steps = 5
            if self.phase_steps == 5:
                logger.debug(f"--- Moving on to Daylight ---")
                self.phase_steps = 0
                self.phase = self.PHASE_DAYLIGHT_EYRIE
        if self.phase == self.PHASE_DAYLIGHT_EYRIE:
            if self.phase_steps == 0:
                # can they use Command Warren?
                if CID_COMMAND_WARREN in {c.id for c in current_player.persistent_cards}:
                    ans = []
                    for enemy_id,battle_aid in [(PIND_MARQUISE,AID_BATTLE_MARQUISE),(PIND_ALLIANCE,AID_BATTLE_ALLIANCE)]:
                        possible_battle_clearings = [i for i,x in enumerate(self.board.get_possible_battles(PIND_EYRIE,enemy_id)) if x]
                        for i in possible_battle_clearings:
                            ans.append(i + battle_aid)
                    if bool(ans):
                        logger.debug(f"Checking for use of Command Warren...")
                        return [AID_GENERIC_SKIP] + ans
                self.phase_steps = 1
            if self.phase_steps == 1:
                if len(self.remaining_craft_power) == 1:
                    self.remaining_craft_power = self.board.get_crafting_power(PIND_EYRIE)
                    logger.debug(f"\tCrafting power: {self.remaining_craft_power}")
                ans = list({i+AID_CRAFT_CARD for i in self.get_craftable_ids(current_player)})
                if (CID_ROYAL_CLAIM+AID_CRAFT_CARD) in ans:
                    # find all ways to craft royal claim
                    ans.remove(CID_ROYAL_CLAIM+AID_CRAFT_CARD)
                    ans += self.get_royal_claim_craft_ids()
                if bool(ans):
                    unused_pers = {c.id for c in current_player.persistent_cards} - self.persistent_used_this_turn
                    if CID_CODEBREAKERS in unused_pers:
                        for pid in (PIND_MARQUISE,PIND_ALLIANCE):
                            ans.append(AID_CARD_CODEBREAKERS + pid)
                    if CID_TAX_COLLECTOR in unused_pers:
                        foo = self.board.get_num_warriors(PIND_EYRIE)
                        ans += [i+AID_CARD_TAX_COLLECTOR for i,amount in enumerate(foo) if (amount > 0)]
                    # check for dominance-related actions
                    ans += self.get_dom_payment_actions(current_player)
                    ans += self.get_dom_activation_actions(current_player, PIND_EYRIE)
                    logger.debug(f"Checking for Crafting / use of Birdsong Cards...")
                    return [AID_GENERIC_SKIP] + ans
                else:
                    # no crafting possible, so move onto resolving the decree
                    logger.debug(f"> Moving onto step 2 of Daylight...")
                    self.phase_steps = 2
            if self.phase_steps == 2:
                # RESOLVING THE DECREE
                ans = []
                # check for persistent cards to use
                unused_pers = {c.id for c in current_player.persistent_cards} - self.persistent_used_this_turn
                if CID_CODEBREAKERS in unused_pers:
                    for pid in (PIND_MARQUISE,PIND_ALLIANCE):
                        ans.append(AID_CARD_CODEBREAKERS + pid)
                if CID_TAX_COLLECTOR in unused_pers:
                    foo = self.board.get_num_warriors(PIND_EYRIE)
                    ans += [i+AID_CARD_TAX_COLLECTOR for i,amount in enumerate(foo) if (amount > 0)]
                # check for dominance-related actions
                ans += self.get_dom_payment_actions(current_player)
                ans += self.get_dom_activation_actions(current_player, PIND_EYRIE)
                if any(any(x) for x in self.remaining_decree.values()):
                    # an action must be taken if possible!
                    # some of decree still remains to resolve
                    decree_actions = self.get_decree_resolving_actions(current_player)
                    if bool(decree_actions): # there is an action to take for the decree
                        logger.debug(f"Choosing decree-resolving action...")
                        return ans + decree_actions
                    # otherwise, we will turmoil
                    if bool(ans): # give them one last chance to use cards
                        logger.debug("\tTurmoiling, but given last chance to use cards...")
                        return [AID_GENERIC_SKIP] + ans
                    # if no persistent to use, turmoil now
                    logger.debug(">> TURMOILING")
                    self.phase_steps = 3
                else:
                    # decree is done
                    if bool(ans):
                        logger.debug(f">>> DECREE COMPLETED, but given last chance to use cards...")
                        return [AID_GENERIC_SKIP] + ans
                    logger.debug(">>> DECREE COMPLETED")
                    self.phase_steps = 4
            if self.phase_steps == 3: # we are turmoiling
                if max(self.victory_points) >= 30: # make sure we don't take away a win
                    return [0]
                to_discard,pts = current_player.turmoil_helper()
                self.change_score(PIND_EYRIE,-pts)
                for c in to_discard:
                    # self.turn_log.change_plr_cards_lost(PIND_EYRIE,c.id)
                    self.add_to_discard(c)

                logger.debug("Choosing new leader after turmoil...")
                return [i+AID_CHOOSE_LEADER for i in current_player.available_leaders]
            if self.phase_steps == 4: # End of daylight
                logger.debug("--- Moving on to Evening ---")
                self.phase_steps = 0
                self.phase = self.PHASE_EVENING_EYRIE
        if self.phase == self.PHASE_EVENING_EYRIE:
            if self.phase_steps == 0:
                unused_pers = {c.id for c in current_player.persistent_cards} - self.persistent_used_this_turn
                if CID_COBBLER in unused_pers:
                    ans = self.board.get_legal_move_actions(PIND_EYRIE,{0,1,2})
                    if bool(ans):
                        logger.debug("Checking for use of Cobbler...")
                        return [AID_GENERIC_SKIP] + ans
                self.phase_steps = 1
            # Evening Phase
            if self.phase_steps == 1:
                # if self.outrage_offender is not None:
                #     # prevent drawing cards before paying outrage
                #     return [AID_GENERIC_SKIP]
                self.change_score(PIND_EYRIE,current_player.get_points_to_score())
                self.draw_cards(PIND_EYRIE,current_player.get_num_cards_to_draw())
                self.phase_steps = 2
            if self.phase_steps == 2 and len(current_player.hand) > 5:
                ans = {c.id+AID_DISCARD_CARD for c in current_player.hand}
                logger.debug("Eyrie must discard from hand...")
                return list(ans)
            # turn done!
            if max(self.victory_points) >= 30:
                return [0]
            logger.debug("--- End of Eyrie's Turn ---\n")
            self.phase_steps = 0
            self.save_to_history()
            if self.active_dominances[PIND_EYRIE] is not None:
                self.adjust_reward_for_dom_turn(PIND_EYRIE)

            self.current_player = self.next_player_index[PIND_EYRIE]
            if self.current_player == PIND_MARQUISE:
                self.phase = self.PHASE_BIRDSONG_MARQUISE
                self.outside_turn_this_action = PIND_MARQUISE
                return self.advance_marquise(self.players[PIND_MARQUISE])
            elif self.current_player == PIND_ALLIANCE:
                self.phase = self.PHASE_BIRDSONG_ALLIANCE
                self.outside_turn_this_action = PIND_ALLIANCE
                return self.advance_alliance(self.players[PIND_ALLIANCE])
        
    def advance_alliance(self,current_player:Alliance):
        "Advances the game assuming we are in the middle of the Alliance's turn."
        if self.phase == self.PHASE_BIRDSONG_ALLIANCE:
            if self.phase_steps == 0: # Start of Birdsong
                if self.active_dominances[PIND_ALLIANCE] is not None:
                    logger.debug("Checking for dominance victory...")
                    if self.has_dominance_win(PIND_ALLIANCE):
                        self.dominance_win = True
                        return [0]
                logger.debug(f"\tResetting for Alliance's Turn...")
                self.reset_for_alliance()
                # can they use BBB?
                if CID_BBB in {c.id for c in current_player.persistent_cards}:
                    logger.debug("Checking for use of BBB...")
                    return [AID_GENERIC_SKIP,AID_CARD_BBB+PIND_MARQUISE,AID_CARD_BBB+PIND_EYRIE]
                self.phase_steps = 1
            while self.phase_steps < 3:
                if self.phase_steps == 1: # Starting Revolts 
                    ans = self.get_revolt_actions(current_player)
                    if not bool(ans):
                        logger.debug("\tCannot start any revolts...")
                        self.phase_steps = 3
                    else:
                        unused_pers = {c.id for c in current_player.persistent_cards} - self.persistent_used_this_turn
                        if CID_STAND_AND_DELIVER in unused_pers:
                            for pid in (PIND_EYRIE,PIND_MARQUISE):
                                if len(self.players[pid].hand) > 0:
                                    ans.append(AID_CARD_STAND_DELIVER + pid)
                        if CID_ROYAL_CLAIM in unused_pers:
                            ans.append(AID_CARD_ROYAL_CLAIM)
                        logger.debug("- Choosing whether or not to revolt...")
                        return [AID_GENERIC_SKIP] + ans
                if self.phase_steps == 2: # choosing what to spend for revolt
                    if self.remaining_supporter_cost != 0:
                        ans = {(c.id + AID_SPEND_SUPPORTER) for c in current_player.supporters if (c.suit in {SUIT_BIRD,self.required_supporter_suit})}
                        if bool(ans):
                            logger.debug(f"\tChoosing supporter to spend...")
                            return list(ans)
                    self.phase_steps = 1
            while self.phase_steps < 5:
                if self.phase_steps == 3: # spreading sympathy
                    ans = self.get_spread_sym_actions(current_player)
                    if not bool(ans):
                        logger.debug("\tCannot spready sympathy...")
                        self.phase_steps = 5
                    else:
                        unused_pers = {c.id for c in current_player.persistent_cards} - self.persistent_used_this_turn
                        if CID_STAND_AND_DELIVER in unused_pers:
                            for pid in (PIND_EYRIE,PIND_MARQUISE):
                                if len(self.players[pid].hand) > 0:
                                    ans.append(AID_CARD_STAND_DELIVER + pid)
                        if CID_ROYAL_CLAIM in unused_pers:
                            ans.append(AID_CARD_ROYAL_CLAIM)
                        logger.debug(f"- Checking for spreading sympathy...")
                        return [AID_GENERIC_SKIP] + ans
                if self.phase_steps == 4: # choosing what to spend for spreading sympathy
                    if self.remaining_supporter_cost != 0:
                        ans = {(c.id + AID_SPEND_SUPPORTER) for c in current_player.supporters if (c.suit in {SUIT_BIRD,self.required_supporter_suit})}
                        if bool(ans):
                            logger.debug(f"\tChoosing supporter to spend ({self.remaining_supporter_cost} needed)")
                            return list(ans)
                    self.phase_steps = 3
            if self.phase_steps == 5:
                unused_pers = {c.id for c in current_player.persistent_cards} - self.persistent_used_this_turn
                ans = []
                if CID_STAND_AND_DELIVER in unused_pers:
                    for pid in (PIND_EYRIE,PIND_MARQUISE):
                        if len(self.players[pid].hand) > 0:
                            ans.append(AID_CARD_STAND_DELIVER + pid)
                if CID_ROYAL_CLAIM in unused_pers:
                    ans.append(AID_CARD_ROYAL_CLAIM)
                if ans:
                    logger.debug("f\tChoosing use of persistent effects...")
                    return [AID_GENERIC_SKIP] + ans
                else:
                    self.phase_steps = 6
            if self.phase_steps == 6:
                logger.debug("--- Moving on to Daylight ---")
                self.phase_steps = 0
                self.phase = self.PHASE_DAYLIGHT_ALLIANCE

        if self.phase == self.PHASE_DAYLIGHT_ALLIANCE:
            if self.phase_steps == 0:
                # can they use Command Warren?
                if CID_COMMAND_WARREN in {c.id for c in current_player.persistent_cards}:
                    ans = []
                    for enemy_id,battle_aid in [(PIND_EYRIE,AID_BATTLE_EYRIE),(PIND_MARQUISE,AID_BATTLE_MARQUISE)]:
                        possible_battle_clearings = [i for i,x in enumerate(self.board.get_possible_battles(PIND_ALLIANCE,enemy_id)) if x]
                        for i in possible_battle_clearings:
                            ans.append(i + battle_aid)
                    if bool(ans):
                        logger.debug(f"Checking for use of Command Warren...")
                        return [AID_GENERIC_SKIP] + ans
                self.phase_steps = 1
            if self.phase_steps == 1: # Complete any of the following in any order / number
                # Craft
                if len(self.remaining_craft_power) == 1:
                    self.remaining_craft_power = self.board.get_crafting_power(PIND_ALLIANCE)
                    logger.debug(f"\tCrafting power: {self.remaining_craft_power}")
                ans = list({i+AID_CRAFT_CARD for i in self.get_craftable_ids(current_player)})
                if (CID_ROYAL_CLAIM+AID_CRAFT_CARD) in ans:
                    # find all ways to craft royal claim
                    ans.remove(CID_ROYAL_CLAIM+AID_CRAFT_CARD)
                    ans += self.get_royal_claim_craft_ids()
                # Use Persistent Cards
                unused_pers = {c.id for c in current_player.persistent_cards} - self.persistent_used_this_turn
                if CID_CODEBREAKERS in unused_pers:
                    for pid in (PIND_EYRIE,PIND_MARQUISE):
                        ans.append(AID_CARD_CODEBREAKERS + pid)
                if CID_TAX_COLLECTOR in unused_pers:
                    foo = self.board.get_num_warriors(PIND_ALLIANCE)
                    ans += [i+AID_CARD_TAX_COLLECTOR for i,amount in enumerate(foo) if (amount > 0)]
                # Mobilize
                ids_in_hand = {c.id for c in current_player.hand}
                ans += [i + AID_MOBILIZE for i in ids_in_hand]
                # Train
                ans += self.get_train_actions(current_player)
                # check for dominance-related actions
                ans += self.get_dom_payment_actions(current_player)
                ans += self.get_dom_activation_actions(current_player, PIND_ALLIANCE)
                if ans:
                    logger.debug("Choosing next Daylight action...")
                    return [AID_GENERIC_SKIP] + ans
                logger.debug("No possible daylight actions remaining...")
                self.phase_steps = 2
            if self.phase_steps == 2: # end of daylight
                logger.debug("--- Moving on to Evening ---")
                self.phase_steps = 0
                self.phase = self.PHASE_EVENING_ALLIANCE

        if self.phase == self.PHASE_EVENING_ALLIANCE:
            if self.phase_steps == 0:
                self.evening_actions_left = current_player.num_officers
                unused_pers = {c.id for c in current_player.persistent_cards} - self.persistent_used_this_turn
                if CID_COBBLER in unused_pers:
                    ans = self.board.get_legal_move_actions(PIND_ALLIANCE,{0,1,2})
                    if bool(ans):
                        logger.debug("Checking for use of Cobbler...")
                        return [AID_GENERIC_SKIP] + ans
                self.phase_steps = 1
            # Evening Phase
            if self.phase_steps == 1: # Military Operations
                if self.evening_actions_left > 0:
                    # Move
                    ans = self.board.get_legal_move_actions(PIND_ALLIANCE,{0,1,2})
                    # Battle
                    for enemy_id,battle_aid in [(PIND_EYRIE,AID_BATTLE_EYRIE),(PIND_MARQUISE,AID_BATTLE_MARQUISE)]:
                        possible_battle_clearings = [i for i,x in enumerate(self.board.get_possible_battles(PIND_ALLIANCE,enemy_id)) if x]
                        for i in possible_battle_clearings:
                            ans.append(i + battle_aid)
                    # Recruit
                    bases = self.board.get_total_building_counts(PIND_ALLIANCE)
                    if (sum(bases) > 0) and (current_player.warrior_storage > 0):
                        ans += [i + AID_RECRUIT_ALLIANCE for i,c in enumerate(bases) if c > 0]
                    # Organize
                    if current_player.get_num_tokens_in_store(TIND_SYMPATHY) > 0:
                        for i,c in enumerate(self.board.clearings):
                            if (c.get_num_warriors(PIND_ALLIANCE) > 0 and
                                    not c.is_sympathetic() and
                                    c.can_place(PIND_ALLIANCE)):
                                ans.append(i + AID_ORGANIZE)
                    if ans:
                        logger.debug(f"Choosing Military Operation ({self.evening_actions_left} Remaining)...")
                        return [AID_GENERIC_SKIP] + ans
                    logger.debug(f"No military operations possible...")
                    self.phase_steps = 2
                else:
                    logger.debug("No military operations left to use...")
                    self.phase_steps = 2
            if self.phase_steps == 2:
                # if self.outrage_offender is not None:
                #     # prevent drawing cards before someone pays outrage
                #     return [AID_GENERIC_SKIP]
                self.draw_cards(PIND_ALLIANCE,current_player.get_num_cards_to_draw())
                self.phase_steps = 3
            if self.phase_steps == 3 and len(current_player.hand) > 5:
                ans = {c.id+AID_DISCARD_CARD for c in current_player.hand}
                logger.debug("Alliance must discard from hand...")
                return list(ans)
            # turn done!
            if max(self.victory_points) >= 30:
                return [0]
            logger.debug("--- End of Alliance's Turn ---\n")
            self.phase_steps = 0
            self.save_to_history()
            if self.active_dominances[PIND_ALLIANCE] is not None:
                self.adjust_reward_for_dom_turn(PIND_ALLIANCE)

            self.current_player = self.next_player_index[PIND_ALLIANCE]
            if self.current_player == PIND_EYRIE:
                self.phase = self.PHASE_BIRDSONG_EYRIE
                self.outside_turn_this_action = PIND_EYRIE
                return self.advance_eyrie(self.players[PIND_EYRIE])
            elif self.current_player == PIND_MARQUISE:
                self.phase = self.PHASE_BIRDSONG_MARQUISE
                self.outside_turn_this_action = PIND_MARQUISE
                return self.advance_marquise(self.players[PIND_MARQUISE])
    
    def advance_vagabond(self,vplayer:Vagabond):
        "Advances the game assuming we are in the middle of the Vagabond's turn."
        if self.phase == self.PHASE_BIRDSONG_VAGABOND:
            if self.phase_steps == 0: # Start of Birdsong
                logger.debug(f"\tResetting for Vagabond's Turn...")
                self.reset_for_vagabond()
                # can they use BBB?
                if CID_BBB in {c.id for c in vplayer.persistent_cards}:
                    logger.debug("Checking for use of BBB...")
                    return [AID_GENERIC_SKIP,AID_CARD_BBB+PIND_MARQUISE,AID_CARD_BBB+PIND_EYRIE,AID_CARD_BBB+PIND_ALLIANCE]
                self.phase_steps = 1
            if self.phase_steps == 1: # Refreshing items
                ans = []
                if self.refreshes_left == -1:
                    # set initial refresh number
                    self.refreshes_left = 3 + 2 * vplayer.tea_track
                logger.debug(f"Refreshes remaining to use: {self.refreshes_left}")
                if self.refreshes_left > 0:
                    ans = vplayer.get_refresh_actions()
                if not bool(ans):
                    logger.debug("\tCannot refresh any more items...")
                    self.refreshes_left = -1
                    self.phase_steps = 2
                else:
                    return ans
            if self.phase_steps == 2: # slipping
                unused_pers = {c.id for c in vplayer.persistent_cards} - self.persistent_used_this_turn
                logger.debug("Choosing where to slip...")
                ans = self.board.get_slip_actions(vplayer.location)
                if CID_STAND_AND_DELIVER in unused_pers:
                    for pid in (PIND_MARQUISE,PIND_EYRIE,PIND_ALLIANCE):
                        if len(self.players[pid].hand) > 0:
                            ans.append(AID_CARD_STAND_DELIVER + pid)
                # NOTE: VB cannot craft royal claim
                # if CID_ROYAL_CLAIM in unused_pers:
                #     ans.append(AID_CARD_ROYAL_CLAIM)
                return [AID_GENERIC_SKIP] + ans
            if self.phase_steps == 3: # other birdsong cards double check
                unused_pers = {c.id for c in vplayer.persistent_cards} - self.persistent_used_this_turn
                ans = []
                if CID_STAND_AND_DELIVER in unused_pers:
                    for pid in (PIND_MARQUISE,PIND_EYRIE,PIND_ALLIANCE):
                        if len(self.players[pid].hand) > 0:
                            ans.append(AID_CARD_STAND_DELIVER + pid)
                # if CID_ROYAL_CLAIM in unused_pers:
                #     ans.append(AID_CARD_ROYAL_CLAIM)
                if bool(ans):
                    logger.debug("Checking for use of Birdsong Cards...")
                    return [AID_GENERIC_SKIP] + ans
                self.phase_steps = 4
            if self.phase_steps == 4:
                logger.debug("--- Moving on to Daylight ---")
                self.phase_steps = 0
                self.phase = self.PHASE_DAYLIGHT_VAGABOND

        if self.phase == self.PHASE_DAYLIGHT_VAGABOND:
            if self.phase_steps == 0:
                # can they use Command Warren?
                if CID_COMMAND_WARREN in {c.id for c in vplayer.persistent_cards}:
                    ans = []
                    if vplayer.location <= 11:
                        for enemy_id,battle_aid in [(PIND_EYRIE,AID_BATTLE_EYRIE),(PIND_MARQUISE,AID_BATTLE_MARQUISE),(PIND_ALLIANCE,AID_BATTLE_ALLIANCE)]:
                            if self.board.clearings[vplayer.location].can_start_battle(PIND_VAGABOND,enemy_id):
                                ans.append(vplayer.location + battle_aid)
                    if bool(ans):
                        logger.debug(f"Checking for use of Command Warren...")
                        return [AID_GENERIC_SKIP] + ans
                self.phase_steps = 1
            while self.phase_steps < 5:
                if self.phase_steps == 1:
                    # Complete any of the following in any order / number (exhausting the right item)
                    # Use Persistent Cards
                    unused_pers = {c.id for c in vplayer.persistent_cards} - self.persistent_used_this_turn
                    ans = []
                    if CID_CODEBREAKERS in unused_pers:
                        for pid in (PIND_EYRIE,PIND_MARQUISE,PIND_ALLIANCE):
                            ans.append(AID_CARD_CODEBREAKERS + pid)
                    # NOTE: VB cannot craft Tax Collector...
                    # if CID_TAX_COLLECTOR in unused_pers:
                    #     foo = self.board.get_num_warriors(PIND_ALLIANCE)
                    #     ans += [i+AID_CARD_TAX_COLLECTOR for i,amount in enumerate(foo) if (amount > 0)]
                    
                    # Move (checks for boots)
                    ans += self.get_vagabond_move_actions(vplayer,1)
                    # Actions requiring being in a clearing
                    if vplayer.location <= 11:
                        ans += self.get_vagabond_clearing_actions(vplayer)
                    # Special Abilities usable at any time
                    if vplayer.chosen_character == CHAR_RANGER and vplayer.has_exhaustable(ITEM_TORCH):
                        ans.append(AID_RANGER_ABILITY)
                    # Repair
                    if vplayer.has_exhaustable(ITEM_HAMMER):
                        repair_actions = set()
                        for i,exh in vplayer.satchel_damaged:
                            if exh == 1:
                                repair_actions.add(AID_REPAIR_EXH + i)
                            else:
                                repair_actions.add(AID_REPAIR_UNEXH + i)
                        ans += list(repair_actions)

                    # check for dominance-related actions
                    ans += self.get_dom_payment_actions(vplayer)
                    ans += self.get_dom_activation_actions(vplayer, PIND_VAGABOND)
                    if ans:
                        logger.debug("Choosing next Daylight action...")
                        return [AID_GENERIC_SKIP] + ans
                    logger.debug("No possible daylight actions remaining...")
                    self.phase_steps = 5

                if self.phase_steps == 2:
                    # second half of Aid action
                    target = self.players[self.aid_target]
                    exhaustable_items = [i for i in range(8) if vplayer.has_exhaustable(i)]
                    takeable_items = [7] + [i for i in range(7) if target.crafted_items[i] > 0]
                    return [(AID_CHOOSE_AID_ITEMS + ei*8 + ti) for ei in exhaustable_items for ti in takeable_items]

                if self.phase_steps == 3:
                    # choosing whether to move with ally warriors
                    pass
                
                if self.phase_steps == 4:
                    # choosing how many ally warriors to move with
                    pass

            if self.phase_steps == 5: # end of daylight
                logger.debug("--- Moving on to Evening ---")
                self.phase_steps = 0
                self.phase = self.PHASE_EVENING_VAGABOND

        if self.phase == self.PHASE_EVENING_VAGABOND:
            # Evening Phase
            if self.phase_steps == 0:
                unused_pers = {c.id for c in vplayer.persistent_cards} - self.persistent_used_this_turn
                if CID_COBBLER in unused_pers:
                    ans = self.get_vagabond_move_actions(vplayer,0)
                    if bool(ans):
                        logger.debug("Checking for use of Cobbler...")
                        return [AID_GENERIC_SKIP] + ans
                self.phase_steps = 1

            if self.phase_steps == 1:
                # Rest
                if vplayer.location > 11:
                    logger.debug("> The Vagabond rests for the night...")
                    for i,exh in vplayer.satchel_damaged:
                        vplayer.remove_item(i,1,exh)
                        if i == ITEM_BAG and vplayer.bag_track == 3:
                            vplayer.add_item(ITEM_BAG,0,0)
                        elif i in Vagabond.TRACK_IDS:
                            vplayer.change_track_amount(i,1)
                        else:
                            vplayer.add_item(i,0,0)
                        logger.debug(f"\t\tVagabond repairs and refreshes a {ID_TO_ITEM[i]}")
                self.phase_steps = 2
            
            if self.phase_steps == 2:
                # Draw cards
                self.draw_cards(PIND_VAGABOND,(1 + vplayer.coins_track))
                self.phase_steps = 3

            if self.phase_steps == 3:
                if len(vplayer.hand) > 5:
                    # Discard down to 5 cards
                    ans = {c.id+AID_DISCARD_CARD for c in vplayer.hand}
                    logger.debug("Vagabond must discard from hand...")
                    return list(ans)
                self.phase_steps = 4
            
            item_limit = 6 + 2 * vplayer.bag_track
            if self.phase_steps == 4 and (len(vplayer.satchel_undamaged) + len(vplayer.satchel_damaged)) > item_limit:
                # Discard items down to limit
                logger.debug(f"> Vagabond has too many items! They must discard down to {item_limit}")
                ans = set()
                for i,exh in vplayer.satchel_undamaged:
                    ans.add(AID_DISCARD_ITEM + 4*i + exh)
                for i,exh in vplayer.satchel_damaged:
                    ans.add(AID_DISCARD_ITEM + 4*i + 2 + exh)
                return list(ans)

            # turn done!
            if max(self.victory_points) >= 30:
                return [0]
            logger.debug("--- End of Vagabond's Turn ---\n")
            self.phase_steps = 0
            self.save_to_history()

            # if self.active_dominances[PIND_MARQUISE] is not None:
            #     self.adjust_reward_for_dom_turn(PIND_MARQUISE)

            self.current_player = self.next_player_index[PIND_VAGABOND]
            if self.current_player == PIND_EYRIE:
                self.phase = self.PHASE_BIRDSONG_EYRIE
                self.outside_turn_this_action = PIND_EYRIE
                return self.advance_eyrie(self.players[PIND_EYRIE])
            elif self.current_player == PIND_ALLIANCE:
                self.phase = self.PHASE_BIRDSONG_ALLIANCE
                self.outside_turn_this_action = PIND_ALLIANCE
                return self.advance_alliance(self.players[PIND_ALLIANCE])
            elif self.current_player == PIND_MARQUISE:
                self.phase = self.PHASE_BIRDSONG_MARQUISE
                self.outside_turn_this_action = PIND_MARQUISE
                return self.advance_marquise(self.players[PIND_MARQUISE])

    # ACTION RESOLUTION
    def resolve_action(self,action:int):
        """
        One of the big ones.
        Given an action number, alters the board itself according
        to the current player. It uses the saved information about the
        state of the board to exactly update the state as if only the
        given action was performed.

        Does NOT try to advance the game, or find who should make
        the next action. That should be done in the advancement function. TODO
        """
        current_player = self.players[self.current_player]
        # go by current phase of the current turn
        ### STANDARD TURNS
        if self.phase == self.PHASE_DAYLIGHT_MARQUISE:
            self.marquise_daylight(action,current_player)
        elif self.phase == self.PHASE_DAYLIGHT_EYRIE:
            self.eyrie_daylight(action,current_player)
        elif self.phase == self.PHASE_DAYLIGHT_ALLIANCE:
            self.alliance_daylight(action,current_player)

        elif self.phase == self.PHASE_BIRDSONG_MARQUISE:
            self.marquise_birdsong(action,current_player)
        elif self.phase == self.PHASE_BIRDSONG_EYRIE:
            self.eyrie_birdsong(action,current_player)
        elif self.phase == self.PHASE_BIRDSONG_ALLIANCE:
            self.alliance_birdsong(action,current_player)

        elif self.phase == self.PHASE_EVENING_MARQUISE:
            self.marquise_evening(action)
        elif self.phase == self.PHASE_EVENING_EYRIE:
            self.eyrie_evening(action)
        elif self.phase == self.PHASE_EVENING_ALLIANCE:
            self.alliance_evening(action,current_player)
        ### INITIAL SETUP
        elif self.phase == self.PHASE_SETUP_MARQUISE:
            self.marquise_setup(action,current_player)
        elif self.phase == self.PHASE_SETUP_EYRIE:
            self.eyrie_setup(action,current_player)

    def marquise_setup(self,action:int,current_player:Marquise):
        "Performs the corresponding Marquise setup action."
        s = self.phase_steps
        if s == 0: # choosing where to put the Keep
            chosen_clearing = action - AID_CHOOSE_CLEARING
            logger.debug(f"Keep placed in clearing {chosen_clearing}")
            current_player.keep_clearing_id = chosen_clearing
            current_player.change_num_tokens(TIND_KEEP,-1)
            self.board.place_token(PIND_MARQUISE,TIND_KEEP,chosen_clearing)
            self.turn_log.change_clr_tokens(chosen_clearing,PIND_MARQUISE,TIND_KEEP,1)
            # Garrison
            skip = self.board.clearings[chosen_clearing].opposite_corner_id
            for i in range(12):
                if i != skip:
                    self.board.place_warriors(PIND_MARQUISE,1,i)
                    self.turn_log.change_clr_warriors(i,PIND_MARQUISE,1)
            current_player.change_num_warriors(-11)
            self.turn_log.change_plr_warrior_supply(PIND_MARQUISE,-11)
            logger.debug(f"1 Marquise warrior placed in each clearing except {skip}")
        elif s == 1: # choosing where to place a sawmill
            chosen_clearing = action - AID_BUILD1
            current_player.update_from_building_placed(BIND_SAWMILL)
            self.board.place_building(PIND_MARQUISE,BIND_SAWMILL,chosen_clearing)
            self.turn_log.change_clr_building(chosen_clearing,PIND_MARQUISE,BIND_SAWMILL,1)
        elif s == 2: # choosing where to place a workshop
            chosen_clearing = action - AID_BUILD2
            current_player.update_from_building_placed(BIND_WORKSHOP)
            self.board.place_building(PIND_MARQUISE,BIND_WORKSHOP,chosen_clearing)
            self.turn_log.change_clr_building(chosen_clearing,PIND_MARQUISE,BIND_WORKSHOP,1)
        elif s == 3: # choosing where to place a recruiter
            chosen_clearing = action - AID_BUILD3
            current_player.update_from_building_placed(BIND_RECRUITER)
            self.board.place_building(PIND_MARQUISE,BIND_RECRUITER,chosen_clearing)
            self.turn_log.change_clr_building(chosen_clearing,PIND_MARQUISE,BIND_RECRUITER,1)
        self.phase_steps += 1

    def marquise_birdsong(self,action:int,current_player:Marquise):
        "Performs the action during the Marquise's birdsong / changes the turn stage."
        if action >= AID_CHOOSE_CLEARING and action <= AID_CHOOSE_CLEARING + 11: # choose where to place wood
            logger.debug(f"\tChose to place wood in clearing {action - AID_CHOOSE_CLEARING}")
            self.available_wood_spots[action - AID_CHOOSE_CLEARING] -= 1
            current_player.change_num_tokens(TIND_WOOD,-1)
            self.board.place_token(PIND_MARQUISE,TIND_WOOD,action - AID_CHOOSE_CLEARING)
            self.turn_log.change_clr_tokens(action - AID_CHOOSE_CLEARING,PIND_MARQUISE,TIND_WOOD,1)
            can_place_wood = [(x > 0) for x in self.available_wood_spots]
            if (current_player.get_num_tokens_in_store(TIND_WOOD) == 0) or (sum(can_place_wood) == 0):
                # finished placing wood
                # (no wood / no sawmills left to cover)
                self.phase_steps = 2
            elif sum(can_place_wood) == 1:
                # there is no choice left of where to place
                logger.debug(f"\tRemaining wood can be placed automatically:")
                i = can_place_wood.index(True)
                foo = self.available_wood_spots[i]
                amount_to_place = min(foo, current_player.get_num_tokens_in_store(TIND_WOOD))
                while amount_to_place > 0:
                    current_player.change_num_tokens(TIND_WOOD,-1)
                    self.board.place_token(PIND_MARQUISE,TIND_WOOD,i)
                    self.turn_log.change_clr_tokens(i,PIND_MARQUISE,TIND_WOOD,1)
                    amount_to_place -= 1
                self.phase_steps = 2
            # if neither of the two conditions above are
            # satisfied, then we still have a choice and
            # do not move on to phase_steps 2
        elif action >= AID_CARD_BBB and action <= AID_CARD_BBB + 2:
            self.activate_better_burrow(PIND_MARQUISE,action - AID_CARD_BBB)
            self.persistent_used_this_turn.add(CID_BBB)
            self.phase_steps = 1
        elif action == AID_GENERIC_SKIP:
            logger.debug("- Chose to Skip")
            self.phase_steps = 3
        elif action >= AID_CARD_STAND_DELIVER and action <= AID_CARD_STAND_DELIVER + 2:
            self.activate_stand_and_deliver(PIND_MARQUISE,action - AID_CARD_STAND_DELIVER)
            self.persistent_used_this_turn.add(CID_STAND_AND_DELIVER)
        elif action == AID_CARD_ROYAL_CLAIM:
            self.activate_royal_claim(PIND_MARQUISE)
            self.persistent_used_this_turn.add(CID_ROYAL_CLAIM)
    
    def marquise_daylight(self,action:int,current_player:Marquise):
        "Performs the given daylight action for the Marquise."
        if action >= AID_CRAFT_CARD and action <= AID_CRAFT_CARD + 40: # craft a card
            self.craft_card(PIND_MARQUISE,action - AID_CRAFT_CARD)
        elif action == AID_GENERIC_SKIP:
            if self.phase_steps in {0,1}: # skipping using CWARREN / crafting / using cards 
                logger.debug("- Chose to Skip")
                self.phase_steps += 1
            elif self.phase_steps == 2:
                logger.debug("- Forfeiting remaining daylight actions...")
                self.phase_steps = 6
            elif self.phase_steps == 3: # we forfeit second move of a march
                logger.debug("- Forfeiting second march move...")
                self.phase_steps = 2
                self.marquise_moves = 2

        elif action >= AID_SPEND_BIRD and action <= AID_SPEND_BIRD + 9:
            logger.debug("Spending a bird card to gain an action:")
            self.discard_from_hand(PIND_MARQUISE, ACTION_TO_BIRD_ID[action])
            self.marquise_actions += 1
        elif action >= AID_BATTLE_EYRIE and action <= AID_BATTLE_EYRIE + 11:
            self.battle = Battle(PIND_MARQUISE,PIND_EYRIE,action - AID_BATTLE_EYRIE)
            self.turn_log.record_battle(action - AID_BATTLE_EYRIE,PIND_MARQUISE,PIND_EYRIE)
            if self.phase_steps == 0:
                logger.debug("Command Warren Activated:")
                self.phase_steps = 1
                self.persistent_used_this_turn.add(CID_COMMAND_WARREN)
            else:
                self.marquise_actions -= 1
        elif action >= AID_BATTLE_ALLIANCE and action <= AID_BATTLE_ALLIANCE + 11:
            self.battle = Battle(PIND_MARQUISE,PIND_ALLIANCE,action - AID_BATTLE_ALLIANCE)
            self.turn_log.record_battle(action - AID_BATTLE_ALLIANCE,PIND_MARQUISE,PIND_ALLIANCE)
            if self.phase_steps == 0:
                logger.debug("Command Warren Activated:")
                self.phase_steps = 1
                self.persistent_used_this_turn.add(CID_COMMAND_WARREN)
            else:
                self.marquise_actions -= 1
        elif action >= AID_MOVE and action <= AID_MOVE + 3599:
            start,foo = divmod(action - AID_MOVE,300)
            end,amount = divmod(foo,25)
            self.board.move_warriors(PIND_MARQUISE,amount + 1,start,end)
            self.turn_log.change_clr_warriors(start,PIND_MARQUISE,-amount-1)
            self.turn_log.change_clr_warriors(end,PIND_MARQUISE,amount+1)
            
            dest = self.board.clearings[end]
            if dest.is_sympathetic():
                self.outrage_offender = PIND_MARQUISE
                self.outrage_suits.append(dest.suit)
            
            if self.marquise_moves == 1: # we just marched a second time
                self.marquise_moves = 2
                self.phase_steps = 2
            else: # we just made the first move in a march
                self.marquise_actions -= 1
                self.marquise_moves = 1
                self.phase_steps = 3
        elif action == AID_RECRUIT:
            logger.debug("Attempting to Recruit:")
            all_recruited = self.place_marquise_warriors(current_player)
            self.recruited_this_turn = 1
            self.marquise_actions -= 1
            if not all_recruited:
                self.phase_steps = 4
        elif action >= AID_BUILD1 and action <= AID_BUILD1 + 11:
            # building a Sawmill
            wood_spent = self.place_marquise_building(current_player, BIND_SAWMILL, action - AID_BUILD1)
            self.marquise_actions -= 1
            if not wood_spent:
                self.phase_steps = 5
        elif action >= AID_BUILD2 and action <= AID_BUILD2 + 11:
            # building a Workshop
            wood_spent = self.place_marquise_building(current_player, BIND_WORKSHOP, action - AID_BUILD2)
            self.marquise_actions -= 1
            if not wood_spent:
                self.phase_steps = 5
        elif action >= AID_BUILD3 and action <= AID_BUILD3 + 11:
            # building a Recruiter
            wood_spent = self.place_marquise_building(current_player, BIND_RECRUITER, action - AID_BUILD3)
            self.marquise_actions -= 1
            if not wood_spent:
                self.phase_steps = 5
        elif action >= AID_OVERWORK and action <= AID_OVERWORK + 503:
            card_id,clearing_id = divmod(action - AID_OVERWORK, 12)
            logger.debug(f"Overworking the Sawmill in clearing {clearing_id}")
            self.marquise_actions -= 1
            self.discard_from_hand(PIND_MARQUISE,card_id)
            current_player.change_num_tokens(TIND_WOOD,-1)
            self.board.place_token(PIND_MARQUISE,TIND_WOOD,clearing_id)
            self.turn_log.change_clr_tokens(clearing_id,PIND_MARQUISE,TIND_WOOD,1)
        elif self.phase_steps == 5:
            # we are choosing where to take wood from
            logger.debug(f"\tChose wood from clearing {action - AID_CHOOSE_CLEARING}")
            self.board.clearings[action - AID_CHOOSE_CLEARING].remove_token(PIND_MARQUISE,TIND_WOOD)
            current_player.change_num_tokens(TIND_WOOD,1)
            self.turn_log.change_clr_tokens(action - AID_CHOOSE_CLEARING,PIND_MARQUISE,TIND_WOOD,-1)
            self.remaining_wood_cost -= 1
            logger.debug(f"\t\tRemaining wood cost: {self.remaining_wood_cost}")
            self.available_wood_spots[action - AID_CHOOSE_CLEARING] -= 1

            logger.debug(f"\t\tNew available_wood_spots: {self.available_wood_spots}")
            can_spend = [(x > 0) for x in self.available_wood_spots]
            logger.debug(f"\t\tcan_spend: {can_spend}")
            if self.remaining_wood_cost == 0:
                # we have taken the last wood needed to spend
                self.phase_steps = 2
            elif sum(can_spend) == 1:
                # we can only take wood from this one spot
                i = can_spend.index(True)
                logger.debug(f"\tWe can take the rest from clearing {i}")
                while self.remaining_wood_cost:
                    current_player.change_num_tokens(TIND_WOOD,1)
                    self.board.clearings[i].remove_token(PIND_MARQUISE,TIND_WOOD)
                    self.turn_log.change_clr_tokens(i,PIND_MARQUISE,TIND_WOOD,-1)
                    self.remaining_wood_cost -= 1
                self.phase_steps = 2
            # if neither of the two conditions above are
            # satisfied, then we still have a choice and
            # do not move on to phase_steps 2

        elif self.phase_steps == 4:
            # we are choosing where to recruit
            logger.debug(f"Chose to recruit in clearing {action - AID_CHOOSE_CLEARING}")
            current_player.change_num_warriors(-1)
            self.turn_log.change_plr_warrior_supply(PIND_MARQUISE,-1)
            self.board.place_warriors(PIND_MARQUISE,1,action - AID_CHOOSE_CLEARING)
            self.turn_log.change_clr_warriors(action - AID_CHOOSE_CLEARING,PIND_MARQUISE,1)
            self.available_recruiters[action - AID_CHOOSE_CLEARING] -= 1
            can_recruit = [(x > 0) for x in self.available_recruiters]
            if (current_player.warrior_storage == 0) or (sum(can_recruit) == 0):
                # we are done recruiting
                self.phase_steps = 2
            elif sum(can_recruit) == 1:
                # there is no choice left of where to place
                i = can_recruit.index(True)
                logger.debug(f"\tWe can recruit the rest in clearing {i}")
                foo = self.available_recruiters[i]
                amount_to_place = min(foo, current_player.warrior_storage)
                current_player.change_num_warriors(-amount_to_place)
                self.turn_log.change_plr_warrior_supply(PIND_MARQUISE,-amount_to_place)
                self.board.place_warriors(PIND_MARQUISE,amount_to_place,action - AID_CHOOSE_CLEARING)
                self.turn_log.change_clr_warriors(action - AID_CHOOSE_CLEARING,PIND_MARQUISE,amount_to_place)
                self.phase_steps = 2
            # if neither of the two conditions above are
            # satisfied, then we still have a choice and
            # do not move on to phase_steps 2

        elif action >= AID_TAKE_DOM and action <= AID_TAKE_DOM + 167:
            suit, target_id = divmod(action - AID_TAKE_DOM,42)
            logger.debug(f"\t> Spending a card to take the available {ID_TO_SUIT[suit]} Dominance card...")
            self.discard_from_hand(PIND_MARQUISE, target_id)
            self.available_dominances[suit] = 0
            dom_card = self.available_dom_card_objs[suit]
            self.available_dom_card_objs[suit] = None
            current_player.hand.append(dom_card)
            self.turn_log.change_plr_hand_size(PIND_MARQUISE,1)
            self.turn_log.change_plr_cards_gained(PIND_MARQUISE,dom_card.id)
        elif action >= AID_ACTIVATE_DOM and action <= AID_ACTIVATE_DOM + 3:
            suit = action - AID_ACTIVATE_DOM
            target_id = suit + 38
            logger.debug(f">>> The Marquise activates the {ID_TO_SUIT[suit]} Dominance Card! <<<")
            dom_card = self.get_card(PIND_MARQUISE,target_id,'hand')
            self.turn_log.change_plr_hand_size(PIND_MARQUISE,-1)
            self.active_dominances[PIND_MARQUISE] = dom_card
            self.victory_points[PIND_MARQUISE] = -1
            self.adjust_reward_for_dom_activation(PIND_MARQUISE)

        elif action >= AID_CARD_CODEBREAKERS and action <= AID_CARD_CODEBREAKERS + 2:
            self.persistent_used_this_turn.add(CID_CODEBREAKERS)
            self.activate_codebreakers(PIND_MARQUISE,action - AID_CARD_CODEBREAKERS)
        elif action >= AID_CARD_TAX_COLLECTOR and action <= AID_CARD_TAX_COLLECTOR + 11: # activate tax collector
            self.persistent_used_this_turn.add(CID_TAX_COLLECTOR)
            self.activate_tax_collector(PIND_MARQUISE,action - AID_CARD_TAX_COLLECTOR)
        elif action >= AID_CRAFT_ROYAL_CLAIM and action <= AID_CRAFT_ROYAL_CLAIM + 14: # craft Royal Claim
            self.craft_royal_claim(PIND_MARQUISE,action)
    
    def marquise_evening(self,action:int):
        "Performs the given action for the Marquise in Evening."
        if action >= AID_MOVE and action <= AID_MOVE + 3599: # Cobbler
            logger.debug("Cobbler Activated:")
            start,foo = divmod(action - AID_MOVE,300)
            end,amount = divmod(foo,25)
            self.board.move_warriors(PIND_MARQUISE,amount + 1,start,end)
            self.turn_log.change_clr_warriors(start,PIND_MARQUISE,-amount-1)
            self.turn_log.change_clr_warriors(end,PIND_MARQUISE,amount+1)

            dest = self.board.clearings[end]
            if dest.is_sympathetic():
                self.outrage_offender = PIND_MARQUISE
                self.outrage_suits.append(dest.suit)

            self.persistent_used_this_turn.add(CID_COBBLER)
            self.phase_steps = 1
        elif action == AID_GENERIC_SKIP: # choose not to use cobbler
            logger.debug("- Chose to Skip")
            self.phase_steps = 1
        elif action >= AID_DISCARD_CARD and action <= AID_DISCARD_CARD + 41: # Discard excess card
            self.discard_from_hand(PIND_MARQUISE, action - AID_DISCARD_CARD)


    def eyrie_setup(self,action:int,current_player:Eyrie):
        "Performs the corresponding Eyrie setup action."
        s = self.phase_steps
        if s == 0: # choosing which leader to setup
            # initial setup
            keep_id = self.players[PIND_MARQUISE].keep_clearing_id
            setup_id = self.board.clearings[keep_id].opposite_corner_id
            current_player.place_roost()
            current_player.change_num_warriors(-6)
            self.turn_log.change_plr_warrior_supply(PIND_EYRIE,-6)
            self.board.place_building(PIND_EYRIE,BIND_ROOST,setup_id)
            self.turn_log.change_clr_building(setup_id,PIND_EYRIE,BIND_ROOST,1)
            self.board.place_warriors(PIND_EYRIE,6,setup_id)
            self.turn_log.change_clr_warriors(setup_id,PIND_EYRIE,6)
            # action is the leader that was chosen
            current_player.choose_new_leader(action - AID_CHOOSE_LEADER)
        self.phase_steps += 1
    
    def eyrie_birdsong(self,action:int,current_player:Eyrie):
        "Performs the action during the Eyrie's birdsong / changes the turn stage."
        if action >= AID_CARD_BBB and action <= AID_CARD_BBB + 2:
            self.activate_better_burrow(PIND_EYRIE,action - AID_CARD_BBB)
            self.persistent_used_this_turn.add(CID_BBB)
            self.phase_steps = 1
        elif action == AID_GENERIC_SKIP: # don't use BBB OR Don't add second card to decree
            logger.debug("- Chose to Skip")
            self.phase_steps += 1

        elif action >= AID_DECREE_RECRUIT and action <= AID_DECREE_RECRUIT + 41: # add card to RECRUIT
            c = self.get_card(PIND_EYRIE,action - AID_DECREE_RECRUIT,'hand')
            self.turn_log.change_plr_hand_size(PIND_EYRIE,-1)
            current_player.add_to_decree(c, DECREE_RECRUIT)
            self.eyrie_cards_added += 1
            if c.suit == SUIT_BIRD:
                self.eyrie_bird_added = 1
            if self.eyrie_cards_added == 2:
                self.phase_steps = 3
        elif action >= AID_DECREE_MOVE and action <= AID_DECREE_MOVE + 41: # add card to MOVE
            c = self.get_card(PIND_EYRIE,action - AID_DECREE_MOVE,'hand')
            self.turn_log.change_plr_hand_size(PIND_EYRIE,-1)
            current_player.add_to_decree(c, DECREE_MOVE)
            self.eyrie_cards_added += 1
            if c.suit == SUIT_BIRD:
                self.eyrie_bird_added = 1
            if self.eyrie_cards_added == 2:
                self.phase_steps = 3
        elif action >= AID_DECREE_BATTLE and action <= AID_DECREE_BATTLE + 41: # add card to BATTLE
            c = self.get_card(PIND_EYRIE,action - AID_DECREE_BATTLE,'hand')
            self.turn_log.change_plr_hand_size(PIND_EYRIE,-1)
            current_player.add_to_decree(c, DECREE_BATTLE)
            self.eyrie_cards_added += 1
            if c.suit == SUIT_BIRD:
                self.eyrie_bird_added = 1
            if self.eyrie_cards_added == 2:
                self.phase_steps = 3
        elif action >= AID_DECREE_BUILD and action <= AID_DECREE_BUILD + 41: # add card to BUILD
            c = self.get_card(PIND_EYRIE,action - AID_DECREE_BUILD,'hand')
            self.turn_log.change_plr_hand_size(PIND_EYRIE,-1)
            current_player.add_to_decree(c, DECREE_BUILD)
            self.eyrie_cards_added += 1
            if c.suit == SUIT_BIRD:
                self.eyrie_bird_added = 1
            if self.eyrie_cards_added == 2:
                self.phase_steps = 3

        elif action >= AID_CARD_STAND_DELIVER and action <= AID_CARD_STAND_DELIVER + 2:
            self.activate_stand_and_deliver(PIND_EYRIE,action - AID_CARD_STAND_DELIVER)
            self.persistent_used_this_turn.add(CID_STAND_AND_DELIVER)
        elif action == AID_CARD_ROYAL_CLAIM:
            self.activate_royal_claim(PIND_EYRIE)
            self.persistent_used_this_turn.add(CID_ROYAL_CLAIM)
        elif action >= AID_BUILD1 and action <= AID_BUILD1 + 11: # choose new roost location with no roosts
            current_player.place_roost()
            self.board.place_building(PIND_EYRIE, BIND_ROOST, action - AID_BUILD1)
            self.turn_log.change_clr_building(action - AID_BUILD1,PIND_EYRIE,BIND_ROOST,1)
            n_warriors = min(3,current_player.warrior_storage)
            self.board.place_warriors(PIND_EYRIE, n_warriors, action - AID_BUILD1)
            self.turn_log.change_clr_warriors(action - AID_BUILD1,PIND_EYRIE,n_warriors)
            current_player.change_num_warriors(-n_warriors)
            self.turn_log.change_plr_warrior_supply(PIND_EYRIE,-n_warriors)
            self.phase_steps = 4
    
    def eyrie_daylight(self,action:int,current_player:Eyrie):
        "Performs the given daylight action for the Eyrie."
        if action >= AID_CRAFT_CARD and action <= AID_CRAFT_CARD + 40: # craft a card
            self.craft_card(PIND_EYRIE,action - AID_CRAFT_CARD)
        elif action == AID_GENERIC_SKIP:
            logger.debug("- Chose to Skip")
            if self.phase_steps == 2: # we are choosing not to use cards when given a last choice
                if any(any(x) for x in self.remaining_decree.values()):
                    self.phase_steps = 3
                else:
                    self.phase_steps = 4
            else: # skipping using command warren / skipping crafting cards
                self.phase_steps += 1

        elif action >= AID_BATTLE_MARQUISE and action <= AID_BATTLE_MARQUISE + 11:
            self.battle = Battle(PIND_EYRIE,PIND_MARQUISE,action - AID_BATTLE_MARQUISE)
            self.turn_log.record_battle(action - AID_BATTLE_MARQUISE,PIND_EYRIE,PIND_MARQUISE)
            if self.phase_steps == 0:
                logger.debug("Command Warren Activated:")
                self.phase_steps = 1
                self.persistent_used_this_turn.add(CID_COMMAND_WARREN)
            else:
                logger.debug(">> Decree: BATTLE")
                self.reduce_decree_count(DECREE_BATTLE, self.board.clearings[action - AID_BATTLE_MARQUISE].suit)
        elif action >= AID_BATTLE_ALLIANCE and action <= AID_BATTLE_ALLIANCE + 11:
            self.battle = Battle(PIND_EYRIE,PIND_ALLIANCE,action - AID_BATTLE_ALLIANCE)
            self.turn_log.record_battle(action - AID_BATTLE_ALLIANCE,PIND_EYRIE,PIND_ALLIANCE)
            if self.phase_steps == 0:
                logger.debug("Command Warren Activated:")
                self.phase_steps = 1
                self.persistent_used_this_turn.add(CID_COMMAND_WARREN)
            else:
                logger.debug(">> Decree: BATTLE")
                self.reduce_decree_count(DECREE_BATTLE, self.board.clearings[action - AID_BATTLE_ALLIANCE].suit)
        elif action >= AID_MOVE and action <= AID_MOVE + 3599:
            logger.debug(">> Decree: MOVE")
            start,foo = divmod(action - AID_MOVE,300)
            end,amount = divmod(foo,25)
            self.board.move_warriors(PIND_EYRIE,amount + 1,start,end)
            self.turn_log.change_clr_warriors(start,PIND_EYRIE,-amount-1)
            self.turn_log.change_clr_warriors(end,PIND_EYRIE,amount+1)

            dest = self.board.clearings[end]
            if dest.is_sympathetic():
                self.outrage_offender = PIND_EYRIE
                self.outrage_suits.append(dest.suit)

            self.reduce_decree_count(DECREE_MOVE, self.board.clearings[start].suit)
        elif action >= AID_CHOOSE_CLEARING and action <= AID_CHOOSE_CLEARING + 11:
            # choosing where to recruit (assuming we have the warriors to place ALL (even charismatic))
            logger.debug(">> Decree: RECRUIT")
            amount = 2 if (current_player.chosen_leader_index == LEADER_CHARISMATIC) else 1
            if amount > current_player.warrior_storage:
                # trying to recruit with charismatic, but only 1 warrior in store
                logger.debug("\tCannot recruit 2 with Charismatic, so Eyrie will turmoil!")
                amount = 1
                turmoil = True
            else:
                turmoil = False
            current_player.change_num_warriors(-amount)
            self.turn_log.change_plr_warrior_supply(PIND_EYRIE,-amount)
            self.board.place_warriors(PIND_EYRIE,amount,action - AID_CHOOSE_CLEARING)
            self.turn_log.change_clr_warriors(action - AID_CHOOSE_CLEARING,PIND_EYRIE,amount)
            self.reduce_decree_count(DECREE_RECRUIT, self.board.clearings[action - AID_CHOOSE_CLEARING].suit)
            if turmoil:
                self.phase_steps = 3
        elif action >= AID_BUILD1 and action <= AID_BUILD1 + 11:
            # building a Roost
            logger.debug(">> Decree: BUILD")
            self.board.place_building(PIND_EYRIE,BIND_ROOST,action - AID_BUILD1)
            self.turn_log.change_clr_building(action - AID_BUILD1,PIND_EYRIE,BIND_ROOST,1)
            current_player.place_roost()
            self.reduce_decree_count(DECREE_BUILD, self.board.clearings[action - AID_BUILD1].suit)
        elif self.phase_steps == 3:
            # we are turmoiling and choosing a new leader
            current_player.choose_new_leader(action - AID_CHOOSE_LEADER)
            self.phase_steps = 4

        elif action >= AID_TAKE_DOM and action <= AID_TAKE_DOM + 167:
            suit, target_id = divmod(action - AID_TAKE_DOM,42)
            logger.debug(f"\t> Spending a card to take the available {ID_TO_SUIT[suit]} Dominance card...")
            self.discard_from_hand(PIND_EYRIE, target_id)
            self.available_dominances[suit] = 0
            dom_card = self.available_dom_card_objs[suit]
            self.available_dom_card_objs[suit] = None
            current_player.hand.append(dom_card)
            self.turn_log.change_plr_hand_size(PIND_EYRIE,1)
            self.turn_log.change_plr_cards_gained(PIND_EYRIE,dom_card.id)
        elif action >= AID_ACTIVATE_DOM and action <= AID_ACTIVATE_DOM + 3:
            suit = action - AID_ACTIVATE_DOM
            target_id = suit + 38
            logger.debug(f">>> The Eyrie activates the {ID_TO_SUIT[suit]} Dominance Card! <<<")
            dom_card = self.get_card(PIND_EYRIE,target_id,'hand')
            self.turn_log.change_plr_hand_size(PIND_EYRIE,-1)
            self.active_dominances[PIND_EYRIE] = dom_card
            self.victory_points[PIND_EYRIE] = -1
            self.adjust_reward_for_dom_activation(PIND_EYRIE)

        elif action >= AID_CARD_CODEBREAKERS and action <= AID_CARD_CODEBREAKERS + 2:
            self.persistent_used_this_turn.add(CID_CODEBREAKERS)
            self.activate_codebreakers(PIND_EYRIE,action - AID_CARD_CODEBREAKERS)
        elif action >= AID_CARD_TAX_COLLECTOR and action <= AID_CARD_TAX_COLLECTOR + 11: # activate tax collector
            self.persistent_used_this_turn.add(CID_TAX_COLLECTOR)
            self.activate_tax_collector(PIND_EYRIE,action - AID_CARD_TAX_COLLECTOR)
        elif action >= AID_CRAFT_ROYAL_CLAIM and action <= AID_CRAFT_ROYAL_CLAIM + 14: # craft Royal Claim
            self.craft_royal_claim(PIND_EYRIE,action)
    
    def eyrie_evening(self,action:int):
        "Performs the given action for the Eyrie in Evening."
        if action >= AID_MOVE and action <= AID_MOVE + 3599: # Cobbler
            logger.debug("Cobbler Activated:")
            start,foo = divmod(action - AID_MOVE,300)
            end,amount = divmod(foo,25)
            self.board.move_warriors(PIND_EYRIE,amount + 1,start,end)
            self.turn_log.change_clr_warriors(start,PIND_EYRIE,-amount-1)
            self.turn_log.change_clr_warriors(end,PIND_EYRIE,amount+1)

            dest = self.board.clearings[end]
            if dest.is_sympathetic():
                self.outrage_offender = PIND_EYRIE
                self.outrage_suits.append(dest.suit)

            self.persistent_used_this_turn.add(CID_COBBLER)
            self.phase_steps = 1
        elif action == AID_GENERIC_SKIP: # choose not to use cobbler
            logger.debug("- Chose to Skip")
            self.phase_steps = 1
        elif action >= AID_DISCARD_CARD and action <= AID_DISCARD_CARD + 41: # Discard excess card
            self.discard_from_hand(PIND_EYRIE, action - AID_DISCARD_CARD)


    def alliance_setup(self,current_player:Alliance):
        "Performs the setup for the Alliance (draws 3 supporters)."
        for i in range(3):
            c_drawn = self.deck.draw(1)[0]
            current_player.add_to_supporters(c_drawn)
            logger.debug(f"\tThe Alliance draws: {c_drawn.name} and adds it to their supporters")
    
    def alliance_birdsong(self,action:int,current_player:Alliance):
        "Performs the action during the Alliance's birdsong / changes the turn stage."
        if action >= AID_CARD_BBB and action <= AID_CARD_BBB + 2:
            self.activate_better_burrow(PIND_ALLIANCE,action - AID_CARD_BBB)
            self.persistent_used_this_turn.add(CID_BBB)
            self.phase_steps = 1
        elif action == AID_GENERIC_SKIP: # don't use BBB OR Skip current action
            logger.debug("- Chose to Skip")
            if self.phase_steps == 0:
                self.phase_steps = 1
            elif self.phase_steps == 1:
                self.phase_steps = 3
            elif self.phase_steps == 3:
                self.phase_steps = 5
            elif self.phase_steps == 5:
                self.phase_steps = 6

        elif action >= AID_REVOLT and action <= AID_REVOLT + 11: # Start a Revolt
            c = self.board.clearings[action - AID_REVOLT]
            self.phase_steps = 2
            self.remaining_supporter_cost = 2
            self.required_supporter_suit = c.suit
            self.alliance_action_clearing = action - AID_REVOLT
            logger.debug(f"> The Alliance REVOLT in Clearing {c.id} <")
        elif action >= AID_SPREAD_SYMPATHY and action <= AID_SPREAD_SYMPATHY + 11: # spread sympathy
            c = self.board.clearings[action - AID_SPREAD_SYMPATHY]
            self.phase_steps = 4
            tokens_left = current_player.get_num_tokens_in_store(TIND_SYMPATHY)
            next_cost = Alliance.sympathy_costs[10 - tokens_left]
            self.remaining_supporter_cost = next_cost + int(c.has_martial_law())
            self.required_supporter_suit = c.suit
            self.alliance_action_clearing = action - AID_SPREAD_SYMPATHY
            logger.debug(f"- The Alliance spread sympathy to Clearing {c.id} -")
        elif action >= AID_SPEND_SUPPORTER and action <= AID_SPEND_SUPPORTER + 41: # choose supporter to spend
            cid = action - AID_SPEND_SUPPORTER
            self.discard_from_supporters(current_player,cid)
            self.remaining_supporter_cost -= 1
            if self.remaining_supporter_cost > 0:
                return
            # cost spent
            if self.phase_steps == 2:
                # activate revolt
                self.revolt_helper(current_player,self.alliance_action_clearing)
            elif self.phase_steps == 4:
                # place sympathy token/score
                self.board.place_token(PIND_ALLIANCE,TIND_SYMPATHY,self.alliance_action_clearing)
                self.turn_log.change_clr_tokens(self.alliance_action_clearing,PIND_ALLIANCE,TIND_SYMPATHY,1)
                pts = current_player.spread_sympathy_helper()
                self.change_score(PIND_ALLIANCE,pts)
            self.alliance_action_clearing = None
            self.required_supporter_suit = None

        elif action >= AID_CARD_STAND_DELIVER and action <= AID_CARD_STAND_DELIVER + 2:
            self.activate_stand_and_deliver(PIND_ALLIANCE,action - AID_CARD_STAND_DELIVER)
            self.persistent_used_this_turn.add(CID_STAND_AND_DELIVER)
        elif action == AID_CARD_ROYAL_CLAIM:
            self.activate_royal_claim(PIND_ALLIANCE)
            self.persistent_used_this_turn.add(CID_ROYAL_CLAIM)
    
    def alliance_daylight(self,action:int,current_player:Alliance):
        "Performs the given daylight action for the Alliance."
        if action >= AID_CRAFT_CARD and action <= AID_CRAFT_CARD + 40: # craft a card
            self.craft_card(PIND_ALLIANCE,action - AID_CRAFT_CARD)
        elif action == AID_GENERIC_SKIP:
            logger.debug("- Chose to Skip")
            self.phase_steps += 1

        elif action >= AID_MOBILIZE and action <= AID_MOBILIZE + 41:
            c_to_add = self.get_card(PIND_ALLIANCE,action - AID_MOBILIZE,"hand")
            self.turn_log.change_plr_hand_size(PIND_ALLIANCE,-1)
            logger.debug(f"\t> Mobilize: {c_to_add.name} added to supporters")
            self.add_to_supporters_check(current_player,c_to_add)
        elif action >= AID_TRAIN and action <= AID_TRAIN + 41:
            self.discard_from_hand(PIND_ALLIANCE,action - AID_TRAIN)
            current_player.num_officers += 1
            current_player.change_num_warriors(-1)
            self.turn_log.change_plr_warrior_supply(PIND_ALLIANCE,-1)
            logger.debug(f"> Train: Added an Officer to box ({current_player.num_officers} Total)")
        
        elif action >= AID_CARD_CODEBREAKERS and action <= AID_CARD_CODEBREAKERS + 2:
            self.persistent_used_this_turn.add(CID_CODEBREAKERS)
            self.activate_codebreakers(PIND_ALLIANCE,action - AID_CARD_CODEBREAKERS)
        elif action >= AID_CARD_TAX_COLLECTOR and action <= AID_CARD_TAX_COLLECTOR + 11: # activate tax collector
            self.persistent_used_this_turn.add(CID_TAX_COLLECTOR)
            self.activate_tax_collector(PIND_ALLIANCE,action - AID_CARD_TAX_COLLECTOR)
        elif action >= AID_BATTLE_EYRIE and action <= AID_BATTLE_EYRIE + 11:
            self.battle = Battle(PIND_ALLIANCE,PIND_EYRIE,action - AID_BATTLE_EYRIE)
            self.turn_log.record_battle(action - AID_BATTLE_EYRIE,PIND_ALLIANCE,PIND_EYRIE)
            logger.debug("Command Warren Activated:")
            self.phase_steps = 1
            self.persistent_used_this_turn.add(CID_COMMAND_WARREN)
        elif action >= AID_BATTLE_MARQUISE and action <= AID_BATTLE_MARQUISE + 11:
            self.battle = Battle(PIND_ALLIANCE,PIND_MARQUISE,action - AID_BATTLE_MARQUISE)
            self.turn_log.record_battle(action - AID_BATTLE_MARQUISE,PIND_ALLIANCE,PIND_MARQUISE)
            logger.debug("Command Warren Activated:")
            self.phase_steps = 1
            self.persistent_used_this_turn.add(CID_COMMAND_WARREN)
        
        elif action >= AID_TAKE_DOM and action <= AID_TAKE_DOM + 167:
            suit, target_id = divmod(action - AID_TAKE_DOM,42)
            logger.debug(f"\t> Spending a card to take the available {ID_TO_SUIT[suit]} Dominance card...")
            self.discard_from_hand(PIND_ALLIANCE, target_id)
            self.available_dominances[suit] = 0
            dom_card = self.available_dom_card_objs[suit]
            self.available_dom_card_objs[suit] = None
            current_player.hand.append(dom_card)
            self.turn_log.change_plr_hand_size(PIND_ALLIANCE,1)
            self.turn_log.change_plr_cards_gained(PIND_ALLIANCE,dom_card.id)
        elif action >= AID_ACTIVATE_DOM and action <= AID_ACTIVATE_DOM + 3:
            suit = action - AID_ACTIVATE_DOM
            target_id = suit + 38
            logger.debug(f">>> The Alliance activates the {ID_TO_SUIT[suit]} Dominance Card! <<<")
            dom_card = self.get_card(PIND_ALLIANCE,target_id,'hand')
            self.turn_log.change_plr_hand_size(PIND_ALLIANCE,-1)
            self.active_dominances[PIND_ALLIANCE] = dom_card
            self.victory_points[PIND_ALLIANCE] = -1
            self.adjust_reward_for_dom_activation(PIND_ALLIANCE)

        elif action >= AID_CRAFT_ROYAL_CLAIM and action <= AID_CRAFT_ROYAL_CLAIM + 14: # craft Royal Claim
            self.craft_royal_claim(PIND_ALLIANCE,action)
    
    def alliance_evening(self,action:int,current_player:Alliance):
        "Performs the given action for the Alliance in Evening."
        if action >= AID_MOVE and action <= AID_MOVE + 3599: # Move action or Cobbler?
            if self.phase_steps == 0:
                logger.debug("Cobbler Activated:")
                self.phase_steps = 1
                self.persistent_used_this_turn.add(CID_COBBLER)
            elif self.phase_steps == 1:
                logger.debug("> Operation: Move")
                self.evening_actions_left -= 1
            start,foo = divmod(action - AID_MOVE,300)
            end,amount = divmod(foo,25)
            self.board.move_warriors(PIND_ALLIANCE,amount + 1,start,end)
            self.turn_log.change_clr_warriors(start,PIND_ALLIANCE,-amount-1)
            self.turn_log.change_clr_warriors(end,PIND_ALLIANCE,amount+1)

        elif action >= AID_BATTLE_EYRIE and action <= AID_BATTLE_EYRIE + 11:
            self.battle = Battle(PIND_ALLIANCE,PIND_EYRIE,action - AID_BATTLE_EYRIE)
            self.turn_log.record_battle(action - AID_BATTLE_EYRIE,PIND_ALLIANCE,PIND_EYRIE)
            logger.debug("> Operation: Battle")
            self.evening_actions_left -= 1
        elif action >= AID_BATTLE_MARQUISE and action <= AID_BATTLE_MARQUISE + 11:
            self.battle = Battle(PIND_ALLIANCE,PIND_MARQUISE,action - AID_BATTLE_MARQUISE)
            self.turn_log.record_battle(action - AID_BATTLE_MARQUISE,PIND_ALLIANCE,PIND_MARQUISE)
            logger.debug("> Operation: Battle")
            self.evening_actions_left -= 1

        elif action >= AID_RECRUIT_ALLIANCE and action <= AID_RECRUIT_ALLIANCE + 11:
            logger.debug("> Operation: Recruit")
            self.evening_actions_left -= 1
            current_player.change_num_warriors(-1)
            self.turn_log.change_plr_warrior_supply(PIND_ALLIANCE,-1)
            self.board.place_warriors(PIND_ALLIANCE,1,action - AID_RECRUIT_ALLIANCE)
            self.turn_log.change_clr_warriors(action - AID_RECRUIT_ALLIANCE,PIND_ALLIANCE,1)
        elif action >= AID_ORGANIZE and action <= AID_ORGANIZE + 11:
            logger.debug("> Operation: Organize")
            self.evening_actions_left -= 1
            cid = action - AID_ORGANIZE

            self.board.place_warriors(PIND_ALLIANCE,-1,cid)
            self.turn_log.change_clr_warriors(cid,PIND_ALLIANCE,-1)
            current_player.change_num_warriors(1)
            self.turn_log.change_plr_warrior_supply(PIND_ALLIANCE,1)

            self.board.place_token(PIND_ALLIANCE,TIND_SYMPATHY,cid)
            self.turn_log.change_clr_tokens(cid,PIND_ALLIANCE,TIND_SYMPATHY,1)
            pts = current_player.spread_sympathy_helper()
            self.change_score(PIND_ALLIANCE,pts)
        elif action == AID_GENERIC_SKIP: # choose not to use cobbler / any more military operations
            logger.debug("- Chose to Skip")
            if self.phase_steps == 2:
                return
            self.phase_steps += 1
        elif action >= AID_DISCARD_CARD and action <= AID_DISCARD_CARD + 41: # Discard excess card
            self.discard_from_hand(PIND_ALLIANCE, action - AID_DISCARD_CARD)


    def vagabond_setup(self,action:int,vplayer:Vagabond):
        "Performs the corresponding Vagabond setup action."
        s = self.phase_steps
        if s == 0: # choosing which character to play as
            if action == AID_CHOOSE_VB_CLASS: # Thief
                logger.debug("Chosen Class: Thief")
                logger.debug("\tStarting Items: Boot-Torch-Tea-Sword")
                vplayer.chosen_character = CHAR_THIEF
                vplayer.add_item(ITEM_BOOT,0,0)
                vplayer.add_item(ITEM_TORCH,0,0)
                vplayer.add_item(ITEM_TEA,0,0)
                vplayer.add_item(ITEM_SWORD,0,0)
            elif action == (AID_CHOOSE_VB_CLASS + 1): # Tinker
                logger.debug("Chosen Class: Tinker")
                logger.debug("\tStarting Items: Boot-Torch-Bag-Hammer")
                vplayer.chosen_character = CHAR_TINKER
                vplayer.add_item(ITEM_BOOT,0,0)
                vplayer.add_item(ITEM_TORCH,0,0)
                vplayer.add_item(ITEM_BAG,0,0)
                vplayer.add_item(ITEM_HAMMER,0,0)
            elif action == (AID_CHOOSE_VB_CLASS + 2): # Ranger
                logger.debug("Chosen Class: Ranger")
                logger.debug("\tStarting Items: Boot-Torch-Crossbow-Sword")
                vplayer.chosen_character = CHAR_RANGER
                vplayer.add_item(ITEM_BOOT,0,0)
                vplayer.add_item(ITEM_TORCH,0,0)
                vplayer.add_item(ITEM_CROSSBOW,0,0)
                vplayer.add_item(ITEM_SWORD,0,0)
        elif s == 1: # choosing which forest to start in
            chosen_forest = action - AID_STARTING_FOREST
            vplayer.location = chosen_forest + 12
            self.board.forests[chosen_forest].vagabond_present = 1
            logger.debug(f"Vagabond starts in Forest {chosen_forest}")
            # finish setup
            for i in range(3):
                self.draw_new_quest()
            ruin_items = [ITEM_BAG,ITEM_BOOT,ITEM_HAMMER,ITEM_SWORD]
            random.shuffle(ruin_items)
            for clearing in self.ruin_items.keys():
                item = ruin_items.pop()
                self.ruin_items[clearing] = item
                logger.debug(f"\t{ID_TO_ITEM[item]} placed in Ruin in clearing {clearing}")
        self.phase_steps += 1

    def vagabond_birdsong(self,action:int,vplayer:Vagabond):
        "Performs the action during the Vagabond's birdsong / changes the turn stage."
        if action >= AID_CARD_BBB and action <= AID_CARD_BBB + N_PLAYERS-1:
            self.activate_better_burrow(PIND_VAGABOND,action - AID_CARD_BBB)
            self.persistent_used_this_turn.add(CID_BBB)
            self.phase_steps = 1
        elif action == AID_GENERIC_SKIP: # don't use BBB OR Don't add second card to decree
            logger.debug("- Chose to Skip")
            self.phase_steps += 1

        elif action >= AID_REFRESH_DAM and action <= AID_REFRESH_DAM + 7:
            vplayer.refresh_item(action - AID_REFRESH_DAM, 1)
            self.refreshes_left -= 1
        elif action >= AID_REFRESH_UNDAM and action <= AID_REFRESH_UNDAM + 7:
            vplayer.refresh_item(action - AID_REFRESH_UNDAM, 0)
            self.refreshes_left -= 1

        elif action >= AID_VB_MOVE and action <= AID_VB_MOVE + 360:
            start,end = divmod(action - AID_VB_MOVE,19)
            self.board.move_vagabond(start,end)

            start_text = f"Forest {start-12}" if start > 11 else f"Clearing {start}"
            end_text = f"Forest {end-12}" if end > 11 else f"Clearing {end}"
            logger.debug(f"> Vagabond sneakily Slips from {start_text} to {end_text}")
            self.phase_steps = 3

        elif action >= AID_CARD_STAND_DELIVER and action <= AID_CARD_STAND_DELIVER + N_PLAYERS-1:
            self.activate_stand_and_deliver(PIND_VAGABOND,action - AID_CARD_STAND_DELIVER)
            self.persistent_used_this_turn.add(CID_STAND_AND_DELIVER)
        # elif action == AID_CARD_ROYAL_CLAIM:
        #     self.activate_royal_claim(PIND_EYRIE)
        #     self.persistent_used_this_turn.add(CID_ROYAL_CLAIM)
        
    
    def vagabond_daylight(self,action:int,vplayer:Vagabond):
        "Performs the given daylight action for the Vagabond."
        if action >= AID_CRAFT_CARD and action <= AID_CRAFT_CARD + 40: # craft a card
            self.craft_card(PIND_VAGABOND,action - AID_CRAFT_CARD)
        elif action == AID_GENERIC_SKIP:
            logger.debug("- Chose to Skip")
            if self.phase_steps == 0:
                # skipping using command warren
                self.phase_steps = 1
            else:
                # skipping rest of daylight actions
                self.phase_steps = 5

        elif action >= AID_BATTLE_MARQUISE and action <= AID_BATTLE_MARQUISE + 11:
            self.battle = Battle(PIND_EYRIE,PIND_MARQUISE,action - AID_BATTLE_MARQUISE)
            self.turn_log.record_battle(action - AID_BATTLE_MARQUISE,PIND_EYRIE,PIND_MARQUISE)
            if self.phase_steps == 0:
                logger.debug("Command Warren Activated:")
                self.phase_steps = 1
                self.persistent_used_this_turn.add(CID_COMMAND_WARREN)
            else:
                logger.debug(">> Decree: BATTLE")
                self.reduce_decree_count(DECREE_BATTLE, self.board.clearings[action - AID_BATTLE_MARQUISE].suit)
        elif action >= AID_BATTLE_ALLIANCE and action <= AID_BATTLE_ALLIANCE + 11:
            self.battle = Battle(PIND_EYRIE,PIND_ALLIANCE,action - AID_BATTLE_ALLIANCE)
            self.turn_log.record_battle(action - AID_BATTLE_ALLIANCE,PIND_EYRIE,PIND_ALLIANCE)
            if self.phase_steps == 0:
                logger.debug("Command Warren Activated:")
                self.phase_steps = 1
                self.persistent_used_this_turn.add(CID_COMMAND_WARREN)
            else:
                logger.debug(">> Decree: BATTLE")
                self.reduce_decree_count(DECREE_BATTLE, self.board.clearings[action - AID_BATTLE_ALLIANCE].suit)
        
        elif action >= AID_VB_MOVE and action <= AID_VB_MOVE + 360:
            start,end = divmod(action - AID_VB_MOVE,19)
            self.board.move_vagabond(start,end)
            vplayer.exhaust_item(ITEM_BOOT)

            hostile_ids = {i for i,rel in vplayer.relationships.items() if rel == 0}
            end_c = self.board.clearings[end]
            # check if moving to hostile clearing and if we are able to
            if any(end_c.get_num_warriors(p_index) > 0 for p_index in hostile_ids):
                logger.debug("\t>> Moving into a hostile clearing! Must exhaust another boot...")
                vplayer.exhaust_item(ITEM_BOOT)

            start_text = f"Forest {start-12}" if start > 11 else f"Clearing {start}"
            end_text = f"Forest {end-12}" if end > 11 else f"Clearing {end}"
            logger.debug(f"> Vagabond Moves from {start_text} to {end_text}")
        
        elif action >= AID_START_AIDING and action <= AID_START_AIDING + 125:
            target_id,card_id = divmod(action - AID_START_AIDING,42)
            self.aid_target = target_id
            logger.debug(f"> The Vagabond Aids the {ID_TO_PLAYER[target_id]}")
            
            aid_card = self.get_card(PIND_VAGABOND,card_id,"hand")
            self.players[target_id].hand.append(aid_card)
            logger.debug(f"\t{aid_card.name} given (secretly)")

        elif action == AID_EXPLORE:
            clearing = self.board.clearings[vplayer.location]
            logger.debug(f"> The Vagabond Explores the Ruins in Clearing {vplayer.location}")
            vplayer.exhaust_item(ITEM_TORCH)

            item = self.ruin_items[vplayer.location]
            vplayer.add_item(item,0,0)
            logger.debug(f"\tThe Vagabond found a {ID_TO_ITEM[item]}!")
            self.change_score(PIND_VAGABOND,1)
            clearing.num_building_slots += 1
            clearing.num_ruins -= 1

        elif action >= AID_STRIKE and action <= AID_STRIKE + 12:
            chosen_action = action - AID_STRIKE
            warrior_killed = None
            clearing:Clearing = self.board.clearings[vplayer.location]
            if chosen_action == 0: # kill Marquise warrior
                clearing.change_num_warriors(PIND_MARQUISE,-1)
                mplayer:Marquise = self.players[PIND_MARQUISE]
                mplayer.change_num_warriors(1)
                if mplayer.has_suit_in_hand(clearing.suit) and self.keep_is_up():
                    self.field_hospitals.append((1,clearing.suit))
                warrior_killed = PIND_MARQUISE
                target = "Marquise Warrior"
            elif chosen_action in {1,2,3}: # marquise building
                clearing.remove_building(PIND_MARQUISE,chosen_action - 1)
                self.players[PIND_MARQUISE].change_num_buildings(chosen_action - 1, 1)
                target = ID_TO_MBUILD[chosen_action - 1]
            elif chosen_action in {4,5}: # marquise token
                clearing.remove_token(PIND_MARQUISE,chosen_action - 4)
                self.players[PIND_MARQUISE].change_num_tokens(chosen_action - 4, 1)
                target = ID_TO_MTOKEN[chosen_action - 4]

            elif chosen_action == 6: # Eyrie warrior
                clearing.change_num_warriors(PIND_EYRIE,-1)
                self.players[PIND_EYRIE].change_num_warriors(1)
                warrior_killed = PIND_EYRIE
                target = "Eyrie Warrior"
            elif chosen_action == 7: # Roost
                clearing.remove_building(PIND_EYRIE,BIND_ROOST)
                self.players[PIND_EYRIE].change_num_buildings(BIND_ROOST, 1)
                target = "Roost"

            elif chosen_action == 8: # Alliance Warrior
                clearing.change_num_warriors(PIND_ALLIANCE,-1)
                self.players[PIND_ALLIANCE].change_num_warriors(1)
                warrior_killed = PIND_ALLIANCE
                target = "Alliance Warrior"
            elif chosen_action in {9,10,11}: # Alliance Base
                clearing.remove_building(PIND_ALLIANCE,chosen_action - 9)
                self.players[PIND_ALLIANCE].change_num_buildings(chosen_action - 9, 1)
                target = ID_TO_ABUILD[chosen_action - 9]
            elif chosen_action == 12: # Sympathy Token
                clearing.remove_token(PIND_ALLIANCE,TIND_SYMPATHY)
                self.players[PIND_ALLIANCE].change_num_tokens(TIND_SYMPATHY, 1)
                target = "Sympathy Token"
                self.outrage_offender = PIND_VAGABOND
                self.outrage_suits.append(clearing.suit)
            
            logger.debug(f"Vagabond strikes down {target} in Clearing {vplayer.location}")
            vplayer.exhaust_item(ITEM_CROSSBOW)
            if warrior_killed is not None:
                if vplayer.relationships[warrior_killed] != 0:
                    logger.debug(f"\t> The {ID_TO_PLAYER[warrior_killed]} are now HOSTILE toward the Vagabond!")
                    vplayer.relationships[warrior_killed] = 0
            else:
                self.change_score(PIND_VAGABOND,1)
    
        elif action >= AID_COMPLETE_QUEST and action <= AID_COMPLETE_QUEST + 29:
            logger.debug("> The Vagabond goes on a Quest!")
            qid,cards = divmod(action - AID_COMPLETE_QUEST,2)
            self.complete_quest(vplayer,qid)
            
            if cards == 1:
                logger.debug("The Vagabond chooses to draw 2 cards:")
                self.draw_cards(PIND_VAGABOND,2)
            else:
                logger.debug("The Vagabond chooses to score points:")
                qsuit = self.board.clearings[vplayer.location].suit
                self.change_score(PIND_VAGABOND,len(vplayer.completed_quests[qsuit]))

        elif action >= AID_THIEF_ABILITY and action <= AID_THIEF_ABILITY + 2:
            target_index = action - AID_THIEF_ABILITY
            logger.debug(f"> Thief activates Steal on {ID_TO_PLAYER[target_index]}...")
            vplayer.exhaust_item(ITEM_TORCH)

            target_p_hand = self.players[target_index].hand
            chosen_i = random.randint(0,len(target_p_hand) - 1)
            chosen_card = target_p_hand.pop(chosen_i)
            self.turn_log.change_plr_hand_size(target_index,-1)
            logger.debug(f"\tCard Stolen: {chosen_card.name}")

            vplayer.hand.append(chosen_card)
            self.turn_log.change_plr_hand_size(PIND_VAGABOND,1)
        
        elif action >= AID_TINKER_ABILITY and action <= AID_TINKER_ABILITY + 41:
            target_id = action - AID_TINKER_ABILITY
            for i,card in enumerate(self.discard_pile):
                if card.id == target_id:
                    target_card = self.discard_pile.pop(i)
                    break

            if self.discard_array[i][0] == 1:
                self.discard_array[i][0] = 0
            elif self.discard_array[i][1] == 1:
                self.discard_array[i][1] = 0
                self.discard_array[i][0] = 1
            elif self.discard_array[i][2] == 1:
                self.discard_array[i][2] = 0
                self.discard_array[i][1] = 1

            vplayer.exhaust_item(ITEM_TORCH)
            logger.debug(f"> Tinker activates Day Labor, taking {target_card.name} from the Discard Pile")
            vplayer.hand.append(target_card)
            self.turn_log.change_plr_hand_size(PIND_VAGABOND,1)

        elif self.phase_steps == 3:
            # we are turmoiling and choosing a new leader
            current_player.choose_new_leader(action - AID_CHOOSE_LEADER)
            self.phase_steps = 4

        elif action >= AID_TAKE_DOM and action <= AID_TAKE_DOM + 167:
            suit, target_id = divmod(action - AID_TAKE_DOM,42)
            logger.debug(f"\t> Spending a card to take the available {ID_TO_SUIT[suit]} Dominance card...")
            self.discard_from_hand(PIND_EYRIE, target_id)
            self.available_dominances[suit] = 0
            dom_card = self.available_dom_card_objs[suit]
            self.available_dom_card_objs[suit] = None
            current_player.hand.append(dom_card)
            self.turn_log.change_plr_hand_size(PIND_EYRIE,1)
            self.turn_log.change_plr_cards_gained(PIND_EYRIE,dom_card.id)
        elif action >= AID_ACTIVATE_DOM and action <= AID_ACTIVATE_DOM + 3:
            suit = action - AID_ACTIVATE_DOM
            target_id = suit + 38
            logger.debug(f">>> The Eyrie activates the {ID_TO_SUIT[suit]} Dominance Card! <<<")
            dom_card = self.get_card(PIND_EYRIE,target_id,'hand')
            self.turn_log.change_plr_hand_size(PIND_EYRIE,-1)
            self.active_dominances[PIND_EYRIE] = dom_card
            self.victory_points[PIND_EYRIE] = -1
            self.adjust_reward_for_dom_activation(PIND_EYRIE)

        elif action >= AID_CARD_CODEBREAKERS and action <= AID_CARD_CODEBREAKERS + N_PLAYERS - 1:
            self.persistent_used_this_turn.add(CID_CODEBREAKERS)
            self.activate_codebreakers(PIND_VAGABOND,action - AID_CARD_CODEBREAKERS)
        # elif action >= AID_CARD_TAX_COLLECTOR and action <= AID_CARD_TAX_COLLECTOR + 11: # activate tax collector
        #     self.persistent_used_this_turn.add(CID_TAX_COLLECTOR)
        #     self.activate_tax_collector(PIND_EYRIE,action - AID_CARD_TAX_COLLECTOR)
        # elif action >= AID_CRAFT_ROYAL_CLAIM and action <= AID_CRAFT_ROYAL_CLAIM + 14: # craft Royal Claim
        #     self.craft_royal_claim(PIND_EYRIE,action)
    
    def vagabond_evening(self,action:int):
        "Performs the given action for the Eyrie in Evening."
        if action >= AID_MOVE and action <= AID_MOVE + 3599: # Cobbler
            logger.debug("Cobbler Activated:")
            start,foo = divmod(action - AID_MOVE,300)
            end,amount = divmod(foo,25)
            self.board.move_warriors(PIND_EYRIE,amount + 1,start,end)
            self.turn_log.change_clr_warriors(start,PIND_EYRIE,-amount-1)
            self.turn_log.change_clr_warriors(end,PIND_EYRIE,amount+1)

            dest = self.board.clearings[end]
            if dest.is_sympathetic():
                self.outrage_offender = PIND_EYRIE
                self.outrage_suits.append(dest.suit)

            self.persistent_used_this_turn.add(CID_COBBLER)
            self.phase_steps = 1
        elif action == AID_GENERIC_SKIP: # choose not to use cobbler
            logger.debug("- Chose to Skip")
            self.phase_steps = 1
        elif action >= AID_DISCARD_CARD and action <= AID_DISCARD_CARD + 41: # Discard excess card
            self.discard_from_hand(PIND_EYRIE, action - AID_DISCARD_CARD)


    def render(self):
        if len(self.field_hospitals) > 0:
            print(" -- Resolving Field Hospitals --")
        if self.battle.stage != Battle.STAGE_DONE:
            print(self.board.clearings[self.battle.clearing_id])
            print(self.battle)
        else:
            print(self.players[self.current_player])
            print(self.deck)


#obs_sparse = [i if o == 1 else (i,o) for i,o in enumerate(self.observation) if o != 0]
if __name__ == "__main__":
    env = RootGame(CHOSEN_MAP,STANDARD_DECK_COMP)

    done = False
    action_count = 0
    np.set_printoptions(threshold=np.inf)
    env.reset()

    # total_rewards = np.zeros(N_PLAYERS)
    # # while action_count < 100:
    # while not done:
    #     legal_actions = env.legal_actions()
    #     logger.debug(f"> Action {action_count} - Player: {ID_TO_PLAYER[env.current_player]}")
    #     logger.info(f"Legal Actions: {legal_actions}")
    #     # print(f"Player: {ID_TO_PLAYER[env.current_player]}")
    #     # print(f"> Action {action_count} - Legal Actions: {legal_actions}")

    #     # action = -1
    #     # while action not in legal_actions:
    #     #     action = int(input("Choose a valid action: "))
    #     action = random.choice(legal_actions)
    #     # print(f"\tAction Chosen: {action}")
    #     logger.info(f"\t> Action Chosen: {action}")
    #     obs,reward,done = env.step(action)

    #     logger.debug(f"- Reward for this action: {reward}")
    #     total_rewards += reward
    #     logger.debug(f"\t> New reward total: {total_rewards}")

    #     # logger.debug(f"Observation length: {len(obs)}")
    #     # if action_count % 10 == 0:
    #     #     logger.debug(f"{obs}")
    #     # if env.battle.stage != Battle.STAGE_DONE:
    #     #     for i,sq in enumerate(obs.reshape((139,5,5))):
    #     #         logger.debug(f"- Observation Square {i}:\n{sq}\n")
    #     # if done:
    #     #     env.render()

    #     action_count += 1

    # obs = env.get_observation()
    # logger.debug(f"Length: {len(obs)}\n{obs}")
    # for i,sq in enumerate(obs.reshape((139,5,5))):
    #     logger.debug(f"- Observation Square {i}:\n{sq}\n")

    glens = []
    # obs = env.get_observation()
    # logger.debug(f"Length: {len(obs)}\n{obs}")
    for _ in range(25):
        done = False
        action_count = 0
        total_rewards = np.zeros(N_PLAYERS)
        while not done:
            legal_actions = env.legal_actions()
            logger.info(f"Player: {ID_TO_PLAYER[env.current_player]}")
            logger.info(f"> Action {action_count} - Legal Actions: {legal_actions}")
            # print(f"Player: {ID_TO_PLAYER[env.current_player]}")
            # print(f"> Action {action_count} - Legal Actions: {legal_actions}")

            # action = -1
            # while action not in legal_actions:
            #     action = int(input("Choose a valid action: "))
            action = random.choice(legal_actions)
            # print(f"\tAction Chosen: {action}")
            logger.info(f"\t> Action Chosen: {action}")
            obs,reward,done = env.step(action)
            
            logger.debug(f"- Reward for this action: {reward}")
            total_rewards += reward
            logger.debug(f"\t> New reward total: {total_rewards}")
            
            action_count += 1
        glens.append(action_count)
        print(f"{total_rewards} from {action_count} actions")
        env.reset()

    print(f"\nGames: {glens}")
    print(f"\nLongest Length: {max(glens)}")
    print(f"Shortest Length: {min(glens)}")
    print(f"Average Length: {sum(glens)/25}")