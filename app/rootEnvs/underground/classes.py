import copy
import random
import numpy as np
from typing import Tuple
import logging
# from stable_baselines import logger
# from .actionConstants import *
from actionConstants import *

Recipe = Tuple[int]

logging.basicConfig(filename='file.log',format="%(asctime)s|%(levelname)s|%(name)s|%(message)s",filemode='w')
logger = logging.getLogger("classes")
logger.setLevel(logging.DEBUG)

### Named Constants
SUIT_MOUSE = 0
SUIT_RABBIT = 1
SUIT_FOX = 2
SUIT_BIRD = 3
ID_TO_SUIT = {
    SUIT_MOUSE: "Mouse",
    SUIT_RABBIT: "Rabbit",
    SUIT_FOX: "Fox",
    SUIT_BIRD: "Bird"
}

ITEM_HAMMER = 0
ITEM_SWORD = 1
ITEM_BOOT = 2
ITEM_TEA = 3
ITEM_COINS = 4
ITEM_BAG = 5
ITEM_CROSSBOW = 6
ITEM_TORCH = 7
ITEM_NONE = 8
ID_TO_ITEM = {
    ITEM_HAMMER: "Hammer",
    ITEM_SWORD: "Sword",
    ITEM_BOOT: "Boot",
    ITEM_TEA: "Tea",
    ITEM_COINS: "Coins",
    ITEM_BAG: "Bag",
    ITEM_CROSSBOW: "Crossbow",
    ITEM_TORCH: "Torch",
    ITEM_NONE: "None"
}

N_PLAYERS = 4
TURN_MEMORY = 2
WIN_SCALAR = 0.023
POINT_WIN_REWARD = 10
DOM_WIN_REWARD = 15
# WIN_SCALAR = 0.011
# POINT_WIN_REWARD = 50
# DOM_WIN_REWARD = 60
GAME_SCALAR = WIN_SCALAR
MAX_ACTIONS = 1250

PIND_MARQUISE = 0
PIND_EYRIE = 1
PIND_DUCHY = 2
PIND_CORVID = 3
ID_TO_PLAYER = {
    PIND_MARQUISE: "Marquise de Cat",
    PIND_EYRIE: "Eyrie Dynasties",
    PIND_DUCHY: "Underground Duchy",
    PIND_CORVID: "Corvid Conspiracy",
    -1: "None"
}

# MARQUISE
BIND_SAWMILL = 0
BIND_WORKSHOP = 1
BIND_RECRUITER = 2
TIND_KEEP = 0
TIND_WOOD = 1
ID_TO_MBUILD = {
    BIND_SAWMILL: "Sawmill",
    BIND_WORKSHOP: "Workshop",
    BIND_RECRUITER: "Recruiter"
}
ID_TO_MTOKEN = {
    TIND_KEEP: "THE KEEP",
    TIND_WOOD: "Wood"
}

# EYRIE
BIND_ROOST = 0
LEADER_BUILDER = 0
LEADER_CHARISMATIC = 1
LEADER_COMMANDER = 2
LEADER_DESPOT = 3
DECREE_RECRUIT = 0
DECREE_MOVE = 1
DECREE_BATTLE = 2
DECREE_BUILD = 3
ID_TO_LEADER = {
    LEADER_BUILDER: "Builder",
    LEADER_CHARISMATIC: "Charismatic",
    LEADER_COMMANDER: "Commander",
    LEADER_DESPOT: "Despot",
    None: "None"
}
ID_TO_DECREE = {
    DECREE_RECRUIT: "Recruit",
    DECREE_MOVE: "Move",
    DECREE_BATTLE: "Battle",
    DECREE_BUILD: "Build"
}

# DUCHY
BIND_CITADEL = 0
BIND_MARKET = 1
TIND_TUNNEL = 0

MTYPE_SQUIRE = 0
MTYPE_NOBLE = 1
MTYPE_LORD = 2

MIND_FOREMOLE = 0
MIND_CAPTAIN = 1
MIND_MARSHAL = 2
MIND_BRIGADIER = 3
MIND_BANKER = 4
MIND_MAYOR = 5
MIND_DUCHESS_OF_MUD = 6
MIND_BARON_OF_DIRT = 7
MIND_EARL_OF_STONE = 8

# CORVIDS
TIND_BOMB = 0
TIND_SNARE = 1
TIND_EXTORTION = 2
TIND_RAID = 3

class Clearing:
    def __init__(self,id:int,suit:int,num_building_slots:int,num_ruins:int,opposite_corner_id:int,adj_clearing_ids:set,adj_forest_ids:set,river_adj_ids:set) -> None:
        self.id = id
        self.suit = suit
        self.num_building_slots = num_building_slots
        self.num_ruins = num_ruins
        self.opposite_corner_id = opposite_corner_id
        self.vagabond_present = 0
        self.warriors = {i:0 for i in range(N_PLAYERS)}
        self.tokens = {i:[] for i in range(N_PLAYERS)}
        self.buildings = {i:[] for i in range(N_PLAYERS)}
        self.adjacent_clearing_ids = adj_clearing_ids
        self.adjacent_forest_ids = adj_forest_ids
        self.river_adjacent_ids = river_adj_ids
    
    def get_obs_array(self):
        "Returns an array describing the current state of this clearing."
        # since this information is very important, I'll try to magnify the
        # size of each info piece and scale them more drastically for
        # unusually high amounts of a thing in one clearing
        ret = np.zeros(3)
        # ret[0] = min(self.get_num_warriors(PIND_MARQUISE) / 16, 1)
        ret[0] = self.get_num_warriors(PIND_MARQUISE) / 25
        # ret[1] = min(self.get_num_tokens(PIND_MARQUISE,TIND_WOOD) / 4, 1)
        ret[1] = self.get_num_tokens(PIND_MARQUISE,TIND_WOOD) / 8
        if self.get_num_tokens(PIND_MARQUISE,TIND_KEEP) > 0:
            ret[2] = 1

        foo = np.zeros(3)
        foo[0] = self.get_num_buildings(PIND_MARQUISE,BIND_SAWMILL) / 3
        foo[1] = self.get_num_buildings(PIND_MARQUISE,BIND_WORKSHOP) / 3
        foo[2] = self.get_num_buildings(PIND_MARQUISE,BIND_RECRUITER) / 3
        ret = np.append(ret,foo)

        foo = np.zeros(2)
        # foo[0] = min(self.get_num_warriors(PIND_EYRIE) / 12, 1)
        foo[0] = self.get_num_warriors(PIND_EYRIE) / 20
        if self.get_num_buildings(PIND_EYRIE,BIND_ROOST) > 0:
            foo[1] = 1
        ret = np.append(ret,foo)

        foo = np.zeros(2)
        # foo[0] = min(self.get_num_warriors(PIND_ALLIANCE) / 5, 1)
        foo[0] = self.get_num_warriors(PIND_ALLIANCE) / 10
        if self.get_num_tokens(PIND_ALLIANCE,TIND_SYMPATHY) > 0:
            foo[1] = 1
        ret = np.append(ret,foo)

        foo = np.zeros(9)
        if self.get_num_buildings(PIND_ALLIANCE,BIND_MOUSE_BASE) > 0:
            foo[0] = 1
        elif self.get_num_buildings(PIND_ALLIANCE,BIND_RABBIT_BASE) > 0:
            foo[1] = 1
        elif self.get_num_buildings(PIND_ALLIANCE,BIND_FOX_BASE) > 0:
            foo[2] = 1

        foo[3] = self.vagabond_present
        if self.num_ruins > 0:
            foo[4] = 1
        foo[5] = self.get_num_empty_slots() / 3

        ruler = self.get_ruler()
        if ruler != -1:
            foo[ruler + 6] = 1

        return np.append(ret,foo)
    
    def get_num_empty_slots(self) -> int:
        "Returns the number of empty slots available to build in for the clearing."
        return self.num_building_slots - sum(len(x) for x in self.buildings.values())
    
    def get_ruling_power(self, faction_index:int) -> int:
        "Returns the total ruling power of the given faction: # of Warriors + # of buildings."
        return self.warriors[faction_index] + len(self.buildings[faction_index])
    
    def has_presence(self, faction_index:int) -> bool:
        """
        Returns True if any pieces of the given faction,
        including tokens, warriors, or buildings, are located in
        this clearing, and False otherwise.

        This means that that the given faction could be attacked.
        """
        return bool(self.get_ruling_power(faction_index) or len(self.tokens[faction_index]))

    def get_ruler(self) -> int:
        """
        Returns the player index of the player that rules the current clearing.
        If nobody returns the current clearing, returns -1.
        """
        num_marquise = self.get_ruling_power(PIND_MARQUISE)
        num_eyrie = self.get_ruling_power(PIND_EYRIE)
        num_alliance = self.get_ruling_power(PIND_ALLIANCE)

        if (num_alliance == 0) and (num_eyrie == 0) and (num_marquise == 0):
            return -1
        # a faction has some piece in it (max rule power > 0)
        max_power = max(num_marquise,num_eyrie,num_alliance)
        # the Eyrie rule tied clearings they have a piece in with "Lords of the Forest"
        if num_eyrie == max_power:
            return PIND_EYRIE
        # The Eyrie have < max rule power, so it's either the alliance or marquise
        if num_marquise > num_alliance:
            return PIND_MARQUISE
        if num_alliance > num_marquise:
            return PIND_ALLIANCE
        # there is a tie between marquise/alliance, so nobody rules
        return -1
    
    def is_ruler(self,faction_index:int) -> bool:
        "Returns True if the given faction IS the ruler of this clearing, False otherwise."
        return self.get_ruler() == faction_index
    
    ### BUILDING METHODS
    def get_num_buildings(self,faction_index:int,building_index:int = -1) -> int:
        """
        Returns the number of buildings of the given type for the given faction in the clearing.

        If no building index is specified, returns the total number of buildings
        for the given faction, regardless of their type.
        """
        return len(self.buildings[faction_index]) if (building_index == -1) else self.buildings[faction_index].count(building_index)
    
    def place_building(self,faction_index:int,building_index:int) -> None:
        "Place the given building in this clearing. Assumes the move is legal, performing no checks."
        # logger.debug(f"\t\tBuilding {building_index} added for {ID_TO_PLAYER[faction_index]} in clearing {self.id}")
        self.buildings[faction_index].append(building_index)
    
    def remove_building(self,faction_index:int,building_index:int) -> None:
        "Removes the given building in this clearing, assuming it exists. Does not handle points."
        # logger.debug(f"\t\tBuilding {building_index} removed for {ID_TO_PLAYER[faction_index]} in clearing {self.id}")
        self.buildings[faction_index].remove(building_index)

    ### TOKEN METHODS
    def get_num_tokens(self,faction_index:int,token_index:int = -1) -> int:
        """Returns the number of tokens of the given type for the given faction in the clearing.

        If no token index is specified, returns the total number of tokens
        for the given faction, regardless of their type.
        """
        return len(self.tokens[faction_index]) if (token_index == -1) else self.tokens[faction_index].count(token_index)

    def place_token(self,faction_index:int,token_index:int) -> None:
        "Place the given token in this clearing. Assumes the move is legal, performing no checks."
        # logger.debug(f"\t\tToken {token_index} added for {ID_TO_PLAYER[faction_index]} in clearing {self.id}")
        self.tokens[faction_index].append(token_index)
    
    def remove_token(self,faction_index:int,token_index:int) -> None:
        "Removes the given token in this clearing, assuming it exists. Does not handle points."
        # logger.debug(f"\t\tToken {token_index} removed for {ID_TO_PLAYER[faction_index]} in clearing {self.id}")
        self.tokens[faction_index].remove(token_index)
    
    ### WARRIOR METHODS
    def get_num_warriors(self,faction_index:int) -> int:
        "Returns the number of warriors of the given faction in the clearing."
        return self.warriors[faction_index]

    def change_num_warriors(self,faction_index:int,change:int) -> None:
        """
        Adds the specified number of warriors of the specified faction to the clearing.
        Use a negative number to remove warriors. Does NOT change faction supply counts.
        """
        # logger.debug(f"\t\tWarriors changed by {change} for {ID_TO_PLAYER[faction_index]} in clearing {self.id}")
        self.warriors[faction_index] += change
    

    def can_start_battle(self,attacker_index:int,defender_index:int) -> bool:
        """
        Returns True if one faction can attack the other as directed
        in this particular clearing, and False otherwise.
        """
        return bool( self.warriors[attacker_index] and self.has_presence(defender_index) )

    def can_place(self,faction_index:int) -> bool:
        """
        Returns True if the faction can place a piece in this clearing. This is mainly affected
        by the Marquise's Keep, which blocks other Factions from placing pieces in its clearing.
        """
        return (TIND_KEEP not in self.tokens[PIND_MARQUISE]) or (faction_index == PIND_MARQUISE)
    
    def is_adjacent_to(self,other_index:int):
        "Returns True only if this clearing is connected to the clearing with the other index given."
        return (other_index in self.adjacent_clearing_ids)
    
    def favor_helper(self,safe_faction_index:int):
        """
        Helps with the process of resolving a favor card. Removes any
        and all warriors, buildings, and tokens of all factions that are
        not the safe faction given.

        Returns a 2-tuple: the total number of points that
        should be scored from removing tokens/buildings in this clearing,
        and the number of Marquise warriors removed (for Field Hospitals).
        """
        ans = marqwar = 0
        for faction_i in {j for j in range(N_PLAYERS) if j != safe_faction_index}:
            ans += self.get_num_buildings(faction_i) + self.get_num_tokens(faction_i)
            if faction_i == PIND_MARQUISE:
                marqwar = self.get_num_warriors(faction_i)
                
        return ans,marqwar


class Forest:
    def __init__(self,id:int,adj_clearing_ids:set,adj_forest_ids:set) -> None:
        self.id = id
        self.adjacent_clearing_ids = adj_clearing_ids
        self.adjacent_forest_ids = adj_forest_ids
        self.vagabond_present = 0


class Board:
    def __init__(self, board_comp:tuple) -> None:
        self.board_comp = board_comp
        self.reset()
    
    def __str__(self) -> str:
        s = "Current Board:\n"
        for c in self.clearings:
            s += str(c) + "\n"
        return s
    
    def get_obs_array(self):
        "Returns an array of the current board's state."
        ret = np.asarray([c.get_obs_array() for c in self.clearings])

        # foo = np.zeros(7)
        # for i in range(7):
        #     foo[i] = self.forests[i].vagabond_present

        # return np.append(ret,foo)
        return ret
    
    def randomize_suits(self):
        "Randomizes the suits of each clearing in self.clearings, NOT the board_comp given on creation."
        logger.debug(">--- RANDOMIZING MAP SUITS ---<")
        suits = [SUIT_MOUSE] * 4 + [SUIT_RABBIT] * 4 + [SUIT_FOX] * 4
        for c in self.clearings:
            s = random.choice(suits)
            suits.remove(s)
            c.suit = s
            logger.debug(f"\tClearing {c.id} suit set to {ID_TO_SUIT[s]}")

    def reset(self):
        "Resets the map to the cleared starting state."
        self.clearings: list[Clearing] = copy.deepcopy(self.board_comp[0])
        self.forests: list[Forest] = copy.deepcopy(self.board_comp[1])
        # self.randomize_suits()
        
    def get_total_building_counts(self,faction_index:int,building_index:int = -1):
        """
        Finds the number of buildings of the given faction in each clearing.
        If no building type is specified, returns the total number
        of buildings belonging to the given faction in each clearing.

        Returns a list of integers, with each index matching the corresponding clearing index.
        """
        return [x.get_num_buildings(faction_index,building_index) for x in self.clearings]
    
    def get_clearing_building_counts(self,faction_index:int,clearing_index:int,building_index:int = -1):
        """
        Returns the number of buildings of the given faction of the given type in the GIVEN clearing.
        If no building type is specified, returns the total number
        of buildings belonging to the given faction in the ONE clearing.
        """
        return self.clearings[clearing_index].get_num_buildings(faction_index,building_index)

    def get_total_token_counts(self,faction_index:int,token_index:int = -1):
        """
        Finds the number of tokens of the given faction in each clearing.
        If no token type is specified, returns the total number
        of tokens belonging to the given faction in each clearing.

        Returns a list of integers, with each index matching the corresponding clearing index.
        """
        return [x.get_num_tokens(faction_index,token_index) for x in self.clearings]

    def get_clearing_token_counts(self,faction_index:int,clearing_index:int,token_index:int = -1):
        """
        Returns the number of buildings of the given faction of the given type in the GIVEN clearing.
        If no building type is specified, returns the total number
        of buildings belonging to the given faction in the ONE clearing.
        """
        return self.clearings[clearing_index].get_num_tokens(faction_index,token_index)
    
    def get_empty_building_slot_counts(self):
        """
        Finds the number of empty building slots in each clearing on
        the board. Returns a list of integers, one for each corresponding clearing.
        """
        return [x.get_num_empty_slots() for x in self.clearings]

    def get_rulers(self):
        """
        Finds the current ruling faction of each clearing. Returns a list of integers,
        each corresponding to the faction ID of the ruler in the given clearing, or
        a -1 if there is no ruler of the clearing.
        """
        return [x.get_ruler() for x in self.clearings]
    
    def get_possible_battles(self,attacker_index:int,defender_index:int):
        """
        Returns a list of all of the places where the given faction can start
        a battle with the target faction.

        Returns a list of booleans, one for each clearing.
        """
        return [x.can_start_battle(attacker_index,defender_index) for x in self.clearings]
    
    def get_num_warriors(self,faction_index:int):
        "Returns a list of the number of warriors of the given faction in each clearing."
        return [x.get_num_warriors(faction_index) for x in self.clearings]
    
    def get_crafting_power(self,faction_index:int):
        """
        Returns the crafting power of the given faction. This is
        represented by a list of integers:
        - [Mouse Power / Rabbit Power / Fox Power]
        """
        power = [0,0,0]
        if faction_index == PIND_MARQUISE:
            item_counts = self.get_total_building_counts(PIND_MARQUISE,BIND_WORKSHOP)
        elif faction_index == PIND_EYRIE:
            item_counts = self.get_total_building_counts(PIND_EYRIE,BIND_ROOST)
        else:
            item_counts = self.get_total_token_counts(PIND_ALLIANCE,TIND_SYMPATHY)

        for clearing_i,num in enumerate(item_counts):
            if num:
                c = self.clearings[clearing_i]
                power[c.suit] += num
        return power
    
    def get_legal_move_actions(self,faction_index:int,starting_suits:set):
        """
        Finds every possible distinct legal move that the given faction can currently
        make on the board. The start clearing of each move must have its suit in starting_suits.

        Returns a list of integers: the AID's of each distinct move possible.
        """
        ans = []
        for i,start_clearing in enumerate(self.clearings):
            n_warriors = start_clearing.get_num_warriors(faction_index)
            if (n_warriors == 0) or (start_clearing.suit not in starting_suits):
                continue
            valid_dest_ids = []
            if start_clearing.is_ruler(faction_index):
                valid_dest_ids += list(start_clearing.adjacent_clearing_ids)
            else:
                for dest_id in start_clearing.adjacent_clearing_ids:
                    if self.clearings[dest_id].is_ruler(faction_index):
                        valid_dest_ids.append(dest_id)
            
            ans += [(i*12 + j + GEN_MOVE_CLEARINGS) for j in valid_dest_ids]
        return ans
    
    def move_warriors(self,faction_index:int,amount:int,start_index:int,end_index:int):
        """
        Subtracts warriors of a faction from one clearing, and adds them to another.
        Performs no other checks / assumes the move will be legal.
        """
        logger.debug(f"\t\tMoving {amount} warriors of {ID_TO_PLAYER[faction_index]} from {start_index} to {end_index}")
        start_c,end_c = self.clearings[start_index],self.clearings[end_index]
        
        start_c.change_num_warriors(faction_index,-amount)
        end_c.change_num_warriors(faction_index,amount)

    def move_vagabond(self,start_location:int,end_location:int):
        "Moves the vagabond pawn from one clearing/forest to another."
        if start_location <= 11:
            self.clearings[start_location].vagabond_present = 0
        else:
            self.forests[start_location - 12].vagabond_present = 0
        if end_location <= 11:
            self.clearings[end_location].vagabond_present = 1
        else:
            self.forests[end_location - 12].vagabond_present = 1

    def place_warriors(self,faction_index:int,amount:int,clearing_index:int):
        "Adds the given number of warriors of the faction to the clearing, assuming it is legal to do so."
        logger.debug(f"\t\tPlacing {amount} {ID_TO_PLAYER[faction_index]} warriors in clearing {clearing_index}")
        self.clearings[clearing_index].change_num_warriors(faction_index,amount)
    
    def place_building(self,faction_index:int,building_index:int,clearing_index:int):
        "Adds one of the given building type to the given clearing, assuming it's legal to do so."
        if faction_index == PIND_MARQUISE:
            build = ID_TO_MBUILD[building_index]
        elif faction_index == PIND_EYRIE:
            build = "Roost"
        else:
            build = ID_TO_ABUILD[building_index]
        logger.debug(f"\t\t{build} placed in clearing {clearing_index}")
        self.clearings[clearing_index].place_building(faction_index,building_index)

    def place_token(self,faction_index:int,token_index:int,clearing_index:int):
        "Adds one of the given building type to the given clearing, assuming it's legal to do so."
        if faction_index == PIND_MARQUISE:
            tok = ID_TO_MTOKEN[token_index]
        else:
            tok = ID_TO_ATOKEN[token_index]
        logger.debug(f"\t\t{tok} token placed in clearing {clearing_index}")
        self.clearings[clearing_index].place_token(faction_index,token_index)
    
    def resolve_favor(self,safe_faction_index:int,clearing_indexes:list):
        """
        Resolves the effects of the safe faction crafting a favor card. The
        Effect is carried out in each clearing in the given list of indexes.

        Returns a 2-tuple: the total number of points scored by the
        activating player for removing buildings and tokens, and
        the list of Marquise warriors removed per clearing / the suit (for Field Hospitals)
        """
        ans = 0
        field_hospitals = []
        for i in clearing_indexes:
            pts,marqwars = self.clearings[i].favor_helper(safe_faction_index)
            ans += pts
            if marqwars > 0:
                field_hospitals.append( (marqwars,self.clearings[i].suit) )
        return ans,field_hospitals
    
    def get_wood_available(self):
        """
        For the Marquise, returns a list of integers, one for each clearing.
        The integer for clearing i is the amount of wood tokens available
        to use to Build in clearing i, either from that clearing or using
        any number of connected clearings ruled by the cats.

        If the Marquise do NOT rule clearing i, the amount of wood
        in that clearing is given as -1.
        """
        # logger.debug("GWA Function")
        ans = [-1 for i in range(12)]
        # find out which clearings the Marquise rule - which is where wood actually counts
        foo = [(x == PIND_MARQUISE) for x in self.get_rulers()]
        clearings_left_to_assign = {i for i,ruled in enumerate(foo) if ruled}

        # make groups of connected clearings ruled by the Marquise
        while clearings_left_to_assign:
            i = clearings_left_to_assign.pop()
            # logger.debug(f"\tStarting new group at clearing {i}")
            new_group = {i}
            total = self.clearings[i].get_num_tokens(PIND_MARQUISE,TIND_WOOD)
            c_to_add_to_group = {j for j in self.clearings[i].adjacent_clearing_ids if j in clearings_left_to_assign}
            while c_to_add_to_group:
                j = c_to_add_to_group.pop()
                clearings_left_to_assign.remove(j)
                new_group.add(j)
                total += self.clearings[j].get_num_tokens(PIND_MARQUISE,TIND_WOOD)
                # logger.debug(f"\tAdded clearing {j} to the group. New total: {total}")
                c_to_add_to_group.update( {k for k in self.clearings[j].adjacent_clearing_ids if k in (clearings_left_to_assign - c_to_add_to_group)} )

            for i in new_group:
                ans[i] = total
        # logger.debug(f"GWA Answer: {ans}")
        return ans
    
    def get_wood_to_build_in(self, clearing_index:int):
        """
        For the Marquise, returns a list of integers, one for each clearing.
        Assuming that we are building in the given input clearing index, the
        integer in index i of the returned list is the number of wood tokens
        available to legally take out of clearing i for contributing to the build.

        Thus, a clearing that has 1+ wood tokens, but is NOT connected by rule to
        the building clearing will show up as 0, as it has no usable wood for this build.
        """
        ans = [0 for i in range(12)]
        # logger.debug("GWB Function")
        # it is assumed that the given clearing is ruled by the Marquise (required to build)
        clearings_checked = set()
        clearings_to_check = {clearing_index}
        # logger.debug(f"\tStarting check at clearing {clearing_index}")
        while clearings_to_check:
            i = clearings_to_check.pop()
            clearings_checked.add(i)
            # logger.debug(f"\t  Adding wood in clearing {i}...")
            ans[i] = self.clearings[i].get_num_tokens(PIND_MARQUISE,TIND_WOOD)
            for j in self.clearings[i].adjacent_clearing_ids:
                if (j not in clearings_checked) and (self.clearings[j].get_ruler() == PIND_MARQUISE):
                    # logger.debug(f"\t\tClearing {j} needs to be added...")
                    clearings_to_check.add(j)
        # logger.debug(f"GWB Answer: {ans}")
        return ans
    
    def get_num_sympathetic(self,suit:int):
        "Returns the number of sympathetic clearings currently on the board with the given suit."
        return sum([bool(c.suit == suit and c.is_sympathetic()) for c in self.clearings])
    
    def get_slip_actions(self,vb_location:int):
        """
        Given a location, returns a list of AIDs for all of the valid
        slip actions possible from that location. This is a single, free VB
        movement option that seems to always be valid (ignores rule/relationships).
        """
        ans = []
        if vb_location <= 11: # in clearing
            clearing = self.clearings[vb_location]
            for end_i in clearing.adjacent_clearing_ids:
                ans.append(VB_MOVE + end_i)
            for end_i in clearing.adjacent_forest_ids:
                ans.append(VB_MOVE + (end_i + 12))
        else: # in forest
            forest = self.forests[vb_location - 12]
            for end_i in forest.adjacent_clearing_ids:
                ans.append(VB_MOVE + end_i)
            for end_i in forest.adjacent_forest_ids:
                ans.append(VB_MOVE + (end_i + 12))
        return ans


class Card:
    def __init__(self,id:int,suit:int,name:str,recipe:Recipe,is_ambush:bool,is_dominance:bool,is_persistent:bool,item:int,points:int) -> None:
        self.id = id
        self.suit = suit
        self.name = name
        self.crafting_recipe = recipe
        self.is_ambush = is_ambush
        self.is_dominance = is_dominance
        self.is_persistent = is_persistent
        self.crafting_item = item
        self.points = points
    
    def __str__(self) -> str:
        return f"{self.name} ({ID_TO_SUIT[self.suit]}) (ID {self.id}) ({self.points} Points) (Recipe {self.crafting_recipe})"

class Deck:
    def __init__(self, deck_comp:list):
        self.deck_comp = deck_comp
        self.reset()
    
    def __str__(self) -> str:
        return f" - Deck - {self.size()} Cards\n"
    
    def shuffle(self):
        random.shuffle(self.cards)
    
    def reset(self):
        "Remakes the deck from the starting composition and then shuffles it."
        self.cards = []
        for card,amount in self.deck_comp:
            addition = [card for i in range(amount)]
            self.cards += addition
        self.shuffle()

    def draw(self, n):
        """
        Attempts to draw n cards from the deck by popping from the 'cards' list.
        Returns a list of the card objects drawn.

        If the deck runs out, simply returns all of the cards it could draw.
        """
        drawn = []
        for x in range(n):
            try:
                drawn.append(self.cards.pop())
            except:
                pass
        return drawn
    
    def add(self, cards:list):
        "Adds the given cards in the input to the deck and then shuffles the deck."
        for card in cards:
            self.cards.append(card)
        self.shuffle()
                
    def size(self):
        "Returns the current number of cards in the deck."
        return len(self.cards)


class Player:
    def __init__(self,id:int) -> None:
        self.id = id
        self.warrior_storage = 0
        self.buildings = {}
        self.tokens = {}
        self.crafted_items = {i:0 for i in range(7)}
        self.persistent_cards = []
        self.hand = []
    
    def __str__(self) -> str:
        ret = f"""Player ID {self.id}
    Warriors: {self.warrior_storage} - Buildings: {self.buildings} - Tokens: {self.tokens}
    Crafted Items: {self.crafted_items} - Crafted Cards: {[i.name for i in self.persistent_cards]}
    Hand:\n"""
        for card in self.hand:
            ret += f"\t- {str(card)}\n"
        return ret

    def get_num_buildings_on_track(self, building_index:int) -> int:
        "Returns the number of buildings of the given type left on this player's track."
        return self.buildings[building_index]
    
    def get_num_tokens_in_store(self, token_index:int) -> int:
        "Returns the number of tokens of the given type left in this player's store."
        return self.tokens[token_index]

    def change_num_warriors(self, change:int) -> None:
        """
        Changes the number of warriors in this faction's supply by adding 'change'.
        Use a negative number to remove warriors.
        """
        # logger.debug(f"\t\t{ID_TO_PLAYER[self.id]} warrior storage changed by {change}")
        self.warrior_storage += change
    
    def change_num_buildings(self, building_index:int, change:int) -> None:
        """
        Changes the number of buildings of the given type in this faction's supply by adding 'change'.
        Use a negative number to remove buildings of the given type.
        """
        # logger.debug(f"\t\t{ID_TO_PLAYER[self.id]} buildings ID {building_index} changed by {change}")
        self.buildings[building_index] += change
    
    def change_num_tokens(self, token_index:int, change:int) -> None:
        """
        Changes the number of tokens of the given type in this faction's supply by adding 'change'.
        Use a negative number to remove tokens of the given type.
        """
        # logger.debug(f"\t\t{ID_TO_PLAYER[self.id]} tokens ID {token_index} changed by {change}")
        self.tokens[token_index] += change
    
    def change_num_items(self, item_index:int, change:int) -> None:
        """
        Changes the number of items of the given type in this faction's supply by adding 'change'.
        Use a negative number to remove items of the given type.
        """
        # logger.debug(f"\t\t{ID_TO_PLAYER[self.id]} items ID {item_index} changed by {change}")
        self.crafted_items[item_index] += change
    
    def has_suit_in_hand(self, suit_id:int):
        """
        Returns True only if any card in the player's hand is
        either of the exact given suit OR is a BIRD card.
        """
        return any((c.suit in {suit_id, SUIT_BIRD}) for c in self.hand)

    def get_ambush_actions(self, clearing_suit: int):
        ans = set()
        valid_suits = {SUIT_BIRD,clearing_suit}
        for card in self.hand:
            if card.is_ambush and card.suit in valid_suits:
                ans.add(GEN_USE_AMBUSH+4)
                ans.add(card.suit + GEN_USE_AMBUSH)
        return list(ans)
    
    def get_attacker_card_actions(self):
        "Returns a list of all valid attacking AID's this player can do with their current persistent cards."
        ans = set()
        for card in self.persistent_cards:
            if card.id == CID_ARMORERS:
                ans.add(GEN_EFFECTS_NONE)
                ans.add(GEN_EFFECTS_ARMORERS)
            elif card.id == CID_BRUTAL_TACTICS:
                ans.add(GEN_EFFECTS_NONE)
                ans.add(GEN_EFFECTS_BRUTTACT)
        if len(ans) == 3:
            ans.add(GEN_EFFECTS_ARM_BT)
        return list(ans)

    def get_defender_card_actions(self):
        "Returns a list of all valid defending AID's this player can do with their current persistent cards."
        ans = set()
        for card in self.persistent_cards:
            if card.id == CID_ARMORERS:
                ans.add(GEN_EFFECTS_NONE)
                ans.add(GEN_EFFECTS_ARMORERS)
            elif card.id == CID_SAPPERS:
                ans.add(GEN_EFFECTS_NONE)
                ans.add(GEN_EFFECTS_SAPPERS)
        if len(ans) == 3:
            ans.add(GEN_EFFECTS_ARMSAP)
        return list(ans)
    
    def has_card_id_in_hand(self,id:int):
        "Returns True only if this player has a card with the given ID in their hand."
        return any(c.id == id for c in self.hand)
    

class Marquise(Player):
    building_costs = [0,1,2,3,3,4]
    point_tracks = {}
    point_tracks[BIND_SAWMILL] = [0,1,2,3,4,5]
    point_tracks[BIND_WORKSHOP] = [0,2,2,3,4,5]
    point_tracks[BIND_RECRUITER] = [0,1,2,3,3,4]

    def __init__(self, id: int,) -> None:
        super().__init__(id)
        self.warrior_storage = 25
        for i in range(3):
            self.buildings[i] = 6
        self.tokens[TIND_KEEP] = 1
        self.tokens[TIND_WOOD] = 8
        self.keep_clearing_id = -1
    
    def __str__(self) -> str:
        return "--- Marquise de Cat ---\n" + super().__str__() + f"\nKeep placed in clearing {self.keep_clearing_id}"

    def get_obs_array(self):
        foo = np.zeros((3,6))
        for i,a in self.buildings.items():
            if a > 0:
                foo[i][a - 1] = 1
        ret = np.append((self.warrior_storage / 25),foo)
        
        foo = np.zeros(9)
        if self.tokens[TIND_WOOD] > 0:
            foo[self.get_num_tokens_in_store(TIND_WOOD) - 1] = 1
        if self.tokens[TIND_KEEP] > 0:
            foo[8] = 1
        ret = np.append(ret,foo)

        # cards in hand = 0,1,2,3, 4 or more
        hand_size = min(5, len(self.hand)) / 5
        ret = np.append(ret,hand_size)
        
        foo = np.zeros(11)
        foo.put([CID_TO_PERS_INDEX[c.id] for c in self.persistent_cards], 1)
        ret = np.append(ret,foo)
        
        foo = np.zeros(7)
        for i,a in self.crafted_items.items():
            foo[i] = a / 2
        return np.append(ret,foo)

    def get_num_cards_to_draw(self) -> int:
        "Returns the number of cards to draw at the end of the turn (In Evening)."
        recruiters_left = self.get_num_buildings_on_track(BIND_RECRUITER)
        if recruiters_left > 3:
            return 1
        elif recruiters_left > 1:
            return 2
        return 3

    def update_from_building_placed(self, building_index:int) -> tuple:
        """
        Updates the player board as if the given building was placed:
        - Removes 1 building from the corresponding track
        - Wood token amounts should be handled outside when they are
        properly removed from the board

        Returns the number of wood to use and Victory Points scored.
        """
        i = 6 - self.get_num_buildings_on_track(building_index)
        self.change_num_buildings(building_index,-1)

        return Marquise.building_costs[i], Marquise.point_tracks[building_index][i]
    

class Eyrie(Player):
    roost_points = [0,0,1,2,3,4,4,5]
    leader_starting_viziers = { # (Recruit, Move, Battle, Build)
        LEADER_BUILDER:     (1,1,0,0),
        LEADER_CHARISMATIC: (1,0,1,0),
        LEADER_COMMANDER:   (0,1,1,0),
        LEADER_DESPOT:      (0,1,0,1)
    }

    def __init__(self, id: int) -> None:
        super().__init__(id)
        self.warrior_storage = 20
        self.buildings[BIND_ROOST] = 7
        self.available_leaders = [0,1,2,3]
        self.deposed_leaders = []
        self.chosen_leader_index = None
        self.decree = {i:[] for i in range(4)}
        self.viziers = [Card(CID_LOYAL_VIZIER,SUIT_BIRD,"Loyal Vizier",(0,0,0,0),False,False,False,ITEM_NONE,0) for i in range(2)]
    
    def __str__(self) -> str:
        ret = "--- Eyrie Dynasties ---\n" + super().__str__() + f"\nCurrent Leader: {ID_TO_LEADER[self.chosen_leader_index]} - Leaders Available: {[ID_TO_LEADER[i] for i in self.available_leaders]}\n"
        ret += "Decree:\n"
        for i,lst in self.decree.items():
            ret += f"\t{ID_TO_DECREE[i]}: {[(card.name,ID_TO_SUIT[card.suit]) for card in lst]}\n"
        return ret
    
    def get_obs_array(self):
        foo = np.zeros(7)
        if self.buildings[BIND_ROOST] > 0:
            foo[self.buildings[BIND_ROOST] - 1] = 1
        ret = np.append((self.warrior_storage / 20),foo)

        hand_size = min(5, len(self.hand)) / 5
        ret = np.append(ret,hand_size)
        
        foo = np.zeros(11)
        foo.put([CID_TO_PERS_INDEX[c.id] for c in self.persistent_cards], 1)
        ret = np.append(ret,foo)
        
        foo = np.zeros(7)
        for i,a in self.crafted_items.items():
            foo[i] = a / 2
        ret = np.append(ret,foo)

        foo = np.zeros(8)
        foo.put(self.available_leaders, 1)
        if self.chosen_leader_index is not None:
            foo[self.chosen_leader_index + 4] = 1
        ret = np.append(ret,foo)

        foo = np.zeros((4,43)) # how many of each card is in each decree slot
        bar = np.zeros((4,4,5)) # suit counts for each decree item
        for dec_i in range(4):
            for c in self.decree[dec_i]:
                # count one more of this card
                foo[dec_i][c.id] += 1/3
                # count this card's suit
                if sum(bar[dec_i][c.suit]) == 0:
                    bar[dec_i][c.suit][0] = 1
                else:
                    for suit_seen in range(4):
                        if bar[dec_i][c.suit][suit_seen] == 1:
                            bar[dec_i][c.suit][suit_seen] = 0
                            bar[dec_i][c.suit][suit_seen + 1] = 1
                            break
        # logger.debug(f"decree suit counts: {bar}")
        foo = np.append(foo,bar)    
                        
        return np.append(ret,foo)

    def get_num_cards_to_draw(self) -> int:
        "Returns the number of cards to draw at the end of the turn (In Evening)."
        roosts_left = self.get_num_buildings_on_track(BIND_ROOST)
        if roosts_left > 4:
            return 1
        elif roosts_left > 1:
            return 2
        return 3
    
    def get_points_to_score(self) -> int:
        "Returns the number of points that should be scored in the evening phase."
        x = self.get_num_buildings_on_track(BIND_ROOST)
        return self.roost_points[7 - x]

    def place_roost(self) -> None:
        "Removes 1 roost from the track, as if it was placed on the board."
        self.change_num_buildings(BIND_ROOST,-1)
    
    def add_to_decree(self,card_to_add:Card,decree_index:int):
        "Adds the given card object to the decree."
        logger.debug(f"\t\tCard {card_to_add.name} added to decree at {ID_TO_DECREE[decree_index]}")
        self.decree[decree_index].append(card_to_add)
    
    def choose_new_leader(self, leader_index:int) -> None:
        """
        Sets the given leader as the new leader of the Eyries. Assumes that the two
        Loyal Vizier Cards are in the factions list of viziers, and will attempt to
        place them in the corresponding Decree columns for the given leader.
        """
        logger.debug(f"\tNew Leader Chosen: {ID_TO_LEADER[leader_index]}")
        self.chosen_leader_index = leader_index
        self.available_leaders.remove(leader_index)

        # Place the starting viziers for the new leader
        for i, place_vizier in enumerate(Eyrie.leader_starting_viziers[leader_index]):
            if place_vizier:
                self.add_to_decree(self.viziers.pop(),i)
    
    def turmoil_helper(self):
        """
        Resolves several actions as if Turmoil has just occured:
        - Takes out the Loyal Viziers from the Decree
        - Completely empties the decree
        - Removes the current leader and resets the available leaders if needed

        Returns a tuple containing two items:
        - list: contains all of the card objects in the decree that should be discarded
        - int: the number of bird cards that were in the
        decree in total, which is the number of points that the Eyrie should lose
        """
        num_bird_cards = 0
        cards_to_discard = []

        # we want to remove the two viziers, but will
        # simultaneously count the number of bird cards
        # in the entire decree
        for card_list in self.decree.values():
            num_birds_in_this_slot = sum(x.suit == SUIT_BIRD for x in card_list)
            # this slot only matters now if any birds are in it
            if num_birds_in_this_slot:
                num_bird_cards += num_birds_in_this_slot
                for i,card in enumerate(card_list):
                    if card.id == CID_LOYAL_VIZIER:
                        # This is a loyal vizier, which we need to keep for the Eyrie's board
                        # We can safely pop the card because at most one will appear in each of these lists
                        self.viziers.append(card_list.pop(i))
                        break
            # add the non-vizier cards to the list to discard
            cards_to_discard += card_list
        
        # reset the decree
        self.decree = {i:[] for i in range(4)}

        # depose the current leader
        self.deposed_leaders.append(self.chosen_leader_index)
        self.chosen_leader_index = None
        if len(self.deposed_leaders) == 4:
            # reset available leaders if all 4 have been deposed
            self.available_leaders = [0,1,2,3]
            self.deposed_leaders = []

        return cards_to_discard, num_bird_cards
    

class MinisterCard():
    def __init__(self,id:int,name:str,minister_type:int) -> None:
        self.id = id
        self.name = name
        self.minister_type = minister_type

        if minister_type == MTYPE_SQUIRE:
            self.sway_cost = 2
        elif minister_type == MTYPE_NOBLE:
            self.sway_cost = 3
        elif minister_type == MTYPE_LORD:
            self.sway_cost = 4

class Duchy(Player):

    def __init__(self, id: int) -> None:
        super().__init__(id)
        self.warrior_storage = 20
        self.buildings[BIND_CITADEL] = 3
        self.buildings[BIND_MARKET] = 3
        self.tokens[TIND_TUNNEL] = 3

        self.crowns = {MTYPE_SQUIRE: 3, MTYPE_NOBLE: 3, MTYPE_LORD: 3}
        self.unswayed_ministers = [
            MinisterCard(MIND_FOREMOLE,"Foremole",MTYPE_SQUIRE),
            MinisterCard(MIND_CAPTAIN,"Captain",MTYPE_SQUIRE),
            MinisterCard(MIND_MARSHAL,"Marshal",MTYPE_SQUIRE),
            MinisterCard(MIND_BRIGADIER,"Brigadier",MTYPE_NOBLE),
            MinisterCard(MIND_BANKER,"Banker",MTYPE_NOBLE),
            MinisterCard(MIND_MAYOR,"Mayor",MTYPE_NOBLE),
            MinisterCard(MIND_DUCHESS_OF_MUD,"Duchess of Mud",MTYPE_LORD),
            MinisterCard(MIND_BARON_OF_DIRT,"Baron of Dirt",MTYPE_LORD),
            MinisterCard(MIND_EARL_OF_STONE,"Earl of Stone",MTYPE_LORD)
        ]
        self.swayed_ministers = []


class Corvid(Player):

    def __init__(self, id: int) -> None:
        super().__init__(id)
        self.warrior_storage = 15
        for tid in range(4):
            self.tokens[tid] = 2


class Battle:
    "Keeps track of info about the current active battle."
    # a choice must be made about using an ambush card
    STAGE_DEF_AMBUSH = 0
    STAGE_ATT_AMBUSH = 1
    # choosing what extra effects/cards to activate
    STAGE_ATT_EFFECTS = 2
    STAGE_DEF_EFFECTS = 3
    # choosing the order in which their pieces will be damaged
    STAGE_ATT_ORDER = 4
    STAGE_DEF_ORDER = 5
    # the cats are given a choice whether to activate field hospitals or not
    STAGE_FIELD_HOSPITALS = 6
    # waiting for the dice roll
    STAGE_DICE_ROLL = 7
    # battle is done
    STAGE_DONE = 8
    # Vagabond is picking their ally
    STAGE_VB_CHOOSE_ALLY = 9

    def __init__(self,att_id:int,def_id:int,clearing_id:int) -> None:
        self.attacker_id = att_id
        self.defender_id = def_id
        self.clearing_id = clearing_id
        self.stage = None
        self.att_rolled_hits = None
        self.att_extra_hits = 0
        self.def_rolled_hits = None
        self.def_extra_hits = 0
        self.att_hits_to_deal = 0
        self.def_hits_to_deal = 0
        self.att_ambush_id = None
        self.def_ambush_id = None
        self.att_cardboard_removed = False
        self.def_cardboard_removed = False

        self.vagabond_battle_ally = None
        self.vagabond_ally_hits_taken = 0
        self.vagabond_hits_taken = 0
    
    def __str__(self) -> str:
        ret = f"--- BATTLE: {ID_TO_PLAYER[self.attacker_id]} attacking {ID_TO_PLAYER[self.defender_id]} in Clearing {self.clearing_id} ---\n"
        ret += f"Roll: {(self.att_rolled_hits,self.def_rolled_hits)}"
        return ret
    
    def get_obs_array(self):
        ret = np.zeros(10)
        if self.stage is not None:
            ret[self.stage] = 1
        if self.stage == self.STAGE_DONE:
            return np.append(ret,np.zeros(33))
        
        foo = np.zeros(11)
        if self.attacker_id >= 0:
            foo[self.attacker_id] = 1
        if self.defender_id >= 0:
            foo[self.defender_id + 4] = 1
        if self.vagabond_battle_ally is not None:
            foo[self.vagabond_battle_ally + 8] = 1
        ret = np.append(ret,foo)

        foo = np.zeros(12)
        if self.clearing_id >= 0:
            foo[self.clearing_id] = 1
        ret = np.append(ret,foo)

        foo = np.zeros(6)
        if self.att_rolled_hits is not None:
            foo[0] = self.att_rolled_hits / 3
        if self.def_rolled_hits is not None:
            foo[1] = self.def_rolled_hits / 3
        foo[2] = self.att_extra_hits / 4
        foo[3] = self.def_extra_hits / 4
        if self.att_ambush_id is not None:
            foo[4] = (self.att_ambush_id + 1) / 4
        if self.def_ambush_id is not None:
            foo[5] = (self.def_ambush_id + 1) / 4
        ret = np.append(ret,foo)

        foo = np.zeros(4)
        foo[0] = min(4, self.att_hits_to_deal) / 4
        foo[1] = min(4, self.def_hits_to_deal) / 4
        foo[2] = min(4, self.vagabond_ally_hits_taken) / 4
        foo[3] = min(4, self.vagabond_hits_taken) / 4
        
        return np.append(ret,foo)


# (Card info, Amount in deck)
# Recipe amounts are (Mouse, Bunny, Fox, Wild)
EP_DECK_COMP = [
    # (id,   Suit,        Name,                    Recipe,    is_ambush, is_dom, is_persistent    item,          points), Amount
    (Card(0, SUIT_FOX,    "Anvil",                 (0,0,1,0),   False,     False,   False,      ITEM_HAMMER,      2),      1),
    (Card(1, SUIT_BIRD,   "Arms Trader",           (0,0,2,0),   False,     False,   False,      ITEM_SWORD,       2),      1),
    (Card(2, SUIT_RABBIT,  "A Visit to Friends",   (0,1,0,0),   False,     False,   False,      ITEM_BOOT,        1),      1),
    (Card(3, SUIT_RABBIT,  "Bake Sale",            (0,2,0,0),   False,     False,   False,      ITEM_COINS,       3),      1),
    (Card(4,SUIT_BIRD,   "Birdy Bindle",          (1,0,0,0),   False,     False,   False,      ITEM_BAG,         1),      1),
    (Card(5,SUIT_BIRD,   "Crossbow (Bird)",       (0,0,1,0),   False,     False,   False,      ITEM_CROSSBOW,    1),      1),
    (Card(6,SUIT_MOUSE,  "Crossbow (Mouse)",      (0,0,1,0),   False,     False,   False,      ITEM_CROSSBOW,    1),      1),
    (Card(7, SUIT_FOX,    "Foxfolk Steel",        (0,0,2,0),   False,     False,   False,      ITEM_SWORD,       2),      1),
    (Card(8, SUIT_FOX,    "Gently Used Knapsack", (1,0,0,0),   False,     False,   False,      ITEM_BAG,         1),      1),
    (Card(9, SUIT_MOUSE,   "Investments",         (0,2,0,0),   False,     False,   False,      ITEM_COINS,       3),      1),
    (Card(10, SUIT_MOUSE,    "Mouse-in-a-Sack",    (1,0,0,0),   False,     False,   False,      ITEM_BAG,         1),      1),
    (Card(11, SUIT_FOX,   "Protection Racket",     (0,2,0,0),   False,     False,   False,      ITEM_COINS,       3),      1),
    (Card(12, SUIT_RABBIT,  "Root Tea (Rabbit)",   (1,0,0,0),   False,     False,   False,      ITEM_TEA,         2),      1),
    (Card(13, SUIT_FOX,  "Root Tea (Fox)",         (1,0,0,0),   False,     False,   False,      ITEM_TEA,         2),      1),
    (Card(14, SUIT_MOUSE,  "Root Tea (Mouse)",     (1,0,0,0),   False,     False,   False,      ITEM_TEA,         2),      1),
    (Card(15, SUIT_RABBIT, "Smuggler's Trail",     (1,0,0,0),   False,     False,   False,      ITEM_BAG,         1),      1),
    (Card(16, SUIT_MOUSE,   "Sword",               (0,0,2,0),   False,     False,   False,      ITEM_SWORD,       2),      1),
    (Card(17, SUIT_FOX,   "Travel Gear (Fox)",     (0,1,0,0),   False,     False,   False,      ITEM_BOOT,        1),      1),
    (Card(18, SUIT_MOUSE,   "Travel Gear (Mouse)", (0,1,0,0),   False,     False,   False,      ITEM_BOOT,        1),      1),
    (Card(19, SUIT_BIRD,   "Woodland Runners",     (0,1,0,0),   False,     False,   False,      ITEM_BOOT,        1),      1),
    # (id,   Suit,        Name,                    Recipe,    is_ambush, is_dom, is_persistent    item,          points), Amount
    (Card(20, SUIT_BIRD,   "Eyrie Emigre",          (0,0,2,0),   False,     False,   True,       ITEM_NONE,        0),      1),
    (Card(21, SUIT_BIRD,   "Soup Kitchens",         (1,1,1,0),   False,     False,   True,       ITEM_NONE,        0),      1),
    (Card(22, SUIT_FOX,   "Informants",             (0,0,2,0),   False,     False,   True,       ITEM_NONE,        0),      2),
    (Card(23, SUIT_FOX,   "False Orders",           (0,0,1,0),   False,     False,   True,       ITEM_NONE,        0),      2),
    (Card(24, SUIT_RABBIT, "Swap Meet",             (0,1,0,0),   False,     False,   True,       ITEM_NONE,        0),      2),
    (Card(25, SUIT_RABBIT, "Coffin Makers",         (0,2,0,0),   False,     False,   True,       ITEM_NONE,        0),      1),
    (Card(26, SUIT_RABBIT, "Tunnels",               (0,1,0,0),   False,     False,   True,       ITEM_NONE,        0),      2),
    (Card(27, SUIT_RABBIT, "Charm Offensive",       (0,1,0,0),   False,     False,   True,       ITEM_NONE,        0),      1),
    (Card(28, SUIT_MOUSE, "Murine Broker",          (2,0,0,0),   False,     False,   True,       ITEM_NONE,        0),      1),
    (Card(29, SUIT_MOUSE, "Master Engravers",       (2,0,0,0),   False,     False,   True,       ITEM_NONE,        0),      1),
    (Card(30, SUIT_MOUSE, "League of Adventurous Mice", (1,0,0,0),   False,     False,   True,       ITEM_NONE,        0),      2),
    (Card(31, SUIT_MOUSE, "Mouse Partisans",        (1,0,0,0),   False,     False,   True,       ITEM_NONE,        0),      1),
    (Card(32, SUIT_RABBIT, "Rabbit Partisans",      (0,1,0,0),   False,     False,   True,       ITEM_NONE,        0),      1),
    (Card(33, SUIT_FOX, "Fox Partisans",            (0,0,1,0),   False,     False,   True,       ITEM_NONE,        0),      1),
   
    (Card(34, SUIT_FOX, "Propaganda Bureau",        (0,0,0,3),   False,     False,   True,       ITEM_NONE,        0),      1),
    (Card(35, SUIT_BIRD, "Saboteurs",               (0,0,0,1),   False,     False,   True,       ITEM_NONE,        0),      3),
    (Card(36, SUIT_BIRD, "Corvid Planners",         (0,0,0,2),   False,     False,   True,       ITEM_NONE,        0),      1),
    (Card(37, SUIT_BIRD, "Boat Builders",           (0,0,0,2),   False,     False,   True,       ITEM_NONE,        0),      1),

    (Card(38, SUIT_MOUSE,  "Ambush! (Mouse)",       (0,0,0,0),   True,      False,   False,      ITEM_NONE,        0),      1),
    (Card(39, SUIT_RABBIT,  "Ambush! (Rabbit)",     (0,0,0,0),   True,      False,   False,      ITEM_NONE,        0),      1),
    (Card(40, SUIT_FOX,    "Ambush! (Fox)",         (0,0,0,0),   True,      False,   False,      ITEM_NONE,        0),      1),
    (Card(41, SUIT_BIRD,   "Ambush! (Bird)",        (0,0,0,0),   True,      False,   False,      ITEM_NONE,        0),      2),
    (Card(42,SUIT_MOUSE,  "Mouse Dominance",       (0,0,0,0),   False,     True,    False,      ITEM_NONE,        0),      1),
    (Card(43,SUIT_RABBIT,  "Rabbit Dominance",      (0,0,0,0),   False,     True,    False,      ITEM_NONE,        0),      1),
    (Card(44,SUIT_FOX,    "Fox Dominance",         (0,0,0,0),   False,     True,    False,      ITEM_NONE,        0),      1),
    (Card(45,SUIT_BIRD,   "Bird Dominance",        (0,0,0,0),   False,     True,    False,      ITEM_NONE,        0),      1)
]

CID_AMBUSH_BIRD = 41

CID_SABOTEURS = 35
CID_FALSE_ORDERS = 23
CID_SWAP_MEET = 24
CID_EYRIE_EMIGRE = 20
CID_PROP_BUREAU = 34
CID_LOAM = 30
CID_CHARM_OFFENSIVE = 27
CID_INFORMANTS = 22
CID_MOUSE_PARTISANS = 31
CID_RABBIT_PARTISANS = 32
CID_FOX_PARTISANS = 33
CID_BOAT_BUILDERS = 37
CID_CORVID_PLANNERS = 36
CID_SOUP_KITCHENS = 21
CID_COFFIN_MAKERS = 25
CID_TUNNELS = 26
CID_MURINE_BROKER = 28
CID_MASTER_ENGRAVERS = 29
CID_LOYAL_VIZIER = len(EP_DECK_COMP)

CID_TO_PERS_INDEX = {
    CID_SABOTEURS: 0,
    CID_FALSE_ORDERS: 1,
    CID_SWAP_MEET: 2,
    CID_EYRIE_EMIGRE: 3,
    CID_PROP_BUREAU: 4,
    CID_LOAM: 5,
    CID_CHARM_OFFENSIVE: 6,
    CID_INFORMANTS: 7,
    CID_MOUSE_PARTISANS: 8,
    CID_RABBIT_PARTISANS: 9,
    CID_FOX_PARTISANS: 10,
    CID_BOAT_BUILDERS: 11,
    CID_CORVID_PLANNERS: 12,
    CID_SOUP_KITCHENS: 13,
    CID_COFFIN_MAKERS: 14,
    CID_TUNNELS: 15,
    CID_MURINE_BROKER: 16,
    CID_MASTER_ENGRAVERS: 17
}

ACTION_TO_BIRD_ID = {
    MC_SPEND_BIRD_CARD: CID_AMBUSH_BIRD,
    MC_SPEND_BIRD_CARD + 1: CID_SABOTEURS,
    MC_SPEND_BIRD_CARD + 2: 1, # arms trader
    MC_SPEND_BIRD_CARD + 3: 4, # birdy bindle
    MC_SPEND_BIRD_CARD + 4: CID_EYRIE_EMIGRE,
    MC_SPEND_BIRD_CARD + 5: 5, # bird crossbow
    MC_SPEND_BIRD_CARD + 6: CID_BOAT_BUILDERS,
    MC_SPEND_BIRD_CARD + 7: 19, # woodland runners
    MC_SPEND_BIRD_CARD + 8: CID_CORVID_PLANNERS,
    MC_SPEND_BIRD_CARD + 9: 45, # bird dominance
    MC_SPEND_BIRD_CARD + 10: CID_SOUP_KITCHENS
}
BIRD_ID_TO_ACTION = {i:a for (a,i) in ACTION_TO_BIRD_ID.items()}

MAP_AUTUMN = ([ # board clearings
    #        id, suit,         num_building_slots, num_ruins, opposite_corner_id, set of adj clearings / forests
    Clearing(0,  SUIT_FOX,     1,                 0,         11,                  {1,3,4}, {0,1}),
    Clearing(1,  SUIT_RABBIT,  2,                 0,         -1,                  {0,2}, {0}),
    Clearing(2,  SUIT_MOUSE,   2,                 0,         8,                   {1,3,7}, {0,2}),
    Clearing(3,  SUIT_RABBIT,  1,                 1,         -1,                  {0,2,5}, {0,1,2}),
    Clearing(4,  SUIT_MOUSE,   2,                 0,         -1,                  {0,5,8}, {1,3}),
    Clearing(5,  SUIT_FOX,     1,                 1,         -1,                  {3,4,6,8,10}, {1,2,3,4,5}),
    Clearing(6,  SUIT_MOUSE,   2,                 1,         -1,                  {5,7,11}, {2,5,6}),
    Clearing(7,  SUIT_FOX,     1,                 1,         -1,                  {2,6,11}, {2,6}),
    Clearing(8,  SUIT_RABBIT,  1,                 0,         2,                   {4,5,9}, {3,4}),
    Clearing(9,  SUIT_FOX,     2,                 0,         -1,                  {8,10}, {4}),
    Clearing(10, SUIT_MOUSE,   2,                 0,         -1,                  {5,9,11}, {4,5}),
    Clearing(11, SUIT_RABBIT,  1,                 0,         0,                   {6,7,10}, {5,6})
], [ # board forests
    Forest(0,{0,1,2,3},{1,2}),
    Forest(1,{0,3,4,5},{0,2,3}),
    Forest(2,{2,3,5,6,7},{0,1,5,6}),
    Forest(3,{4,5,8},{1,4}),
    Forest(4,{5,8,9,10},{3,5}),
    Forest(5,{5,6,10,11},{2,4,6}),
    Forest(6,{6,7,11},{2,5})
])

MAP_LAKE = ([ # Clearings
    #        id, suit,         num_building_slots, num_ruins, opposite_corner_id, set of adj clearings / forests / river connected clearings
    Clearing(0,  SUIT_FOX,     1,                 0,         11,                  {1,4,5}, {0,1}, set()),
    Clearing(1,  SUIT_MOUSE,   1,                 0,         -1,                  {0,2,5,6}, {0}, set()),
    Clearing(2,  SUIT_FOX,     2,                 0,         -1,                  {1,3,6}, {0}, set()),
    Clearing(3,  SUIT_RABBIT,  1,                 0,         8,                   {2,7}, {0}, set()),
    Clearing(4,  SUIT_RABBIT,  1,                 0,         -1,                  {0,5,8}, {0}, set()),
    Clearing(5,  SUIT_MOUSE,   2,                 1,         -1,                  {0,1,4}, {0}, {6,9,11}),
    Clearing(6,  SUIT_RABBIT,  2,                 1,         -1,                  {1,2,7}, {0}, {5,9,11}),
    Clearing(7,  SUIT_MOUSE,   2,                 1,         -1,                  {3,6,11}, {0}, set()),
    Clearing(8,  SUIT_MOUSE,   1,                 0,         3,                   {4,9,10}, {0}, set()),
    Clearing(9,  SUIT_RABBIT,  2,                 1,         -1,                  {8,10}, {0}, {5,6,11}),
    Clearing(10, SUIT_FOX,     1,                 0,         -1,                  {8,9,11}, {0}, set()),
    Clearing(11, SUIT_FOX,     2,                 0,         0,                   {7,10}, {0}, {5,6,9})
], [ # Forests
    Forest(0,{0,1,2,3},{1,2}),
    # Forest(0,{0,1,2,3},{1,2}),
    # Forest(0,{0,1,2,3},{1,2}),
    # Forest(0,{0,1,2,3},{1,2}),
    # Forest(0,{0,1,2,3},{1,2}),
    # Forest(0,{0,1,2,3},{1,2}),
    # Forest(0,{0,1,2,3},{1,2}),
    # Forest(0,{0,1,2,3},{1,2}),
    # Forest(0,{0,1,2,3},{1,2})
])

# MAP_WINTER = [
#     #        id, suit,         num_building_slots, num_ruins, opposite_corner_id, set of adj clearings
#     Clearing(0,  SUIT_FOX,     1,                 0,         11,                  {1,4,5}),
#     Clearing(1,  SUIT_RABBIT,  2,                 0,         -1,                  {0,2}),
#     Clearing(2,  SUIT_MOUSE,   2,                 0,         -1,                  {1,3}),
#     Clearing(3,  SUIT_RABBIT,  1,                 0,         8,                   {2,6,7}),
#     Clearing(4,  SUIT_MOUSE,   1,                 0,         -1,                  {0,8}),
#     Clearing(5,  SUIT_FOX,     2,                 1,         -1,                  {0,8,9}),
#     Clearing(6,  SUIT_MOUSE,   2,                 1,         -1,                  {3,10,11}),
#     Clearing(7,  SUIT_FOX,     1,                 0,         -1,                  {3,11}),
#     Clearing(8,  SUIT_RABBIT,  2,                 0,         3,                   {4,5,9}),
#     Clearing(9,  SUIT_FOX,     1,                 1,         -1,                  {5,8,10}),
#     Clearing(10, SUIT_MOUSE,   1,                 1,         -1,                  {6,9,11}),
#     Clearing(11, SUIT_RABBIT,  2,                 0,         0,                   {6,7,10})
# ]

CHOSEN_MAP = MAP_LAKE

CLEARING_SUITS = {
    SUIT_FOX: [c.id for c in CHOSEN_MAP[0] if c.suit == SUIT_FOX],
    SUIT_MOUSE: [c.id for c in CHOSEN_MAP[0] if c.suit == SUIT_MOUSE],
    SUIT_RABBIT: [c.id for c in CHOSEN_MAP[0] if c.suit == SUIT_RABBIT]
}

QUEST_DECK_COMP = [
    (QuestCard(0,SUIT_MOUSE,"Expel Bandits",{ITEM_SWORD:2}), 1),
    (QuestCard(1,SUIT_MOUSE,"Guard Duty",{ITEM_SWORD:1,ITEM_TORCH:1}), 1),
    (QuestCard(2,SUIT_MOUSE,"Fend off a Bear",{ITEM_CROSSBOW:1,ITEM_TORCH:1}), 1),
    (QuestCard(3,SUIT_MOUSE,"Escort",{ITEM_BOOT:2}), 1),
    (QuestCard(4,SUIT_MOUSE,"Logistics Help",{ITEM_BOOT:1,ITEM_BAG:1}), 1),
    (QuestCard(5,SUIT_RABBIT,"Guard Duty",{ITEM_SWORD:1,ITEM_TORCH:1}), 1),
    (QuestCard(6,SUIT_RABBIT,"Errand",{ITEM_TEA:1,ITEM_BOOT:1}), 1),
    (QuestCard(7,SUIT_RABBIT,"Give a Speech",{ITEM_TEA:1,ITEM_TORCH:1}), 1),
    (QuestCard(8,SUIT_RABBIT,"Fend off a Bear",{ITEM_CROSSBOW:1,ITEM_TORCH:1}), 1),
    (QuestCard(9,SUIT_RABBIT,"Expel Bandits",{ITEM_SWORD:2}), 1),
    (QuestCard(10,SUIT_FOX,"Fundraising",{ITEM_TEA:1,ITEM_COINS:1}), 1),
    (QuestCard(11,SUIT_FOX,"Errand",{ITEM_TEA:1,ITEM_BOOT:1}), 1),
    (QuestCard(12,SUIT_FOX,"Logistics Help",{ITEM_BOOT:1,ITEM_BAG:1}), 1),
    (QuestCard(13,SUIT_FOX,"Repair a Shed",{ITEM_HAMMER:1,ITEM_TORCH:1}), 1),
    (QuestCard(14,SUIT_FOX,"Give a Speech",{ITEM_TEA:1,ITEM_TORCH:1}), 1)
]