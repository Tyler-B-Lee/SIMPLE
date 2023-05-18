import random

class Card():
    def __init__(self, id, order, name):
        self.id = id
        self.order = order
        self.name = name

class ResourceCard(Card):
    def __init__(self, id, order, name, material='', amount=1):
        super(ResourceCard, self).__init__(id, order, name)
        self.type = 'resource'
        self.material = material
        self.amount = amount
        
    @property
    def symbol(self):
        return f"(id: {self.id}) {self.material} - {self.amount}"


class CraftCard(Card):
    def __init__(self, id, order, name, material, tool, points, recipe):
        super(CraftCard, self).__init__(id, order, name)
        self.type = 'craft'
        self.material = material
        self.tool = tool
        self.used = False
        self.recipe = recipe
        self.points = points
    
    def flip(self):
        self.used = True
    
    @property
    def symbol(self):
        return f"(id: {self.id}) {self.material} {self.tool} ({self.points} pts) - {'Used' if self.used else 'Usable'}"

class Creeper(Card):
    def __init__(self, id, order, name):
        super(Creeper, self).__init__(id, order, name)
        self.type = 'creeper'

    @property
    def symbol(self):
        return f'(id: {self.id}) CREEPER'

class TNT(Card):
    def __init__(self, id, order, name):
        super(TNT, self).__init__(id, order, name)
        self.type = 'tnt'

    @property
    def symbol(self):
        return f'(id: {self.id}) TNT'
        

class Player():
    def __init__(self, id):
        self.id = id
        self.resources = Position()
        self.tools = Position()
        self.reserve = None
        self.actions = 2
    
    @property
    def score(self):
        tool_points = [x.points for x in self.tools.cards]
        return sum(tool_points)
    
    def num_usable(self, tool_type):
        """Returns the number of usable 'tool_type' the player has. If 'tool_type' is the literal
        string 'any', then counts the total number of usable tools of any type the player has."""
        ans = 0
        if tool_type == 'any':
            for t in self.tools.cards:
                if (t.used == False):
                    ans += 1
        else:
            for t in self.tools.cards:
                if (t.tool == tool_type) and (t.used == False):
                    ans += 1
        return ans
    
    def use_tool(self, tool_type):
        """Goes through the player's list of tools and 'flips' the first one of the given type
        that is unused. This assumes that an unused tool of the given type can be used."""
        for t in self.tools.cards:
            if (t.tool == tool_type) and (t.used == False):
                t.flip()
                break
    
    def crafting_helper(self, tool_mat: str, card_to_use: ResourceCard, used_ids: set, mats_owed: dict):
        id_set_copy = used_ids.copy()
        id_set_copy.add(card_to_use.id)
        mats_copy = mats_owed.copy()
        # print(f'Trying to use {card_to_use.name}, id={card_to_use.id}')

        card_mat = card_to_use.material
        if mats_copy[tool_mat] > 0:
            if card_mat == 'wild':
                card_mat = tool_mat
        else:
            if card_mat == 'wild':
                card_mat = 'wood'
        mats_copy[card_mat] -= card_to_use.amount
        # print(f'    Remaining recipe: {mats_copy}')

        if all([x < 1 for x in mats_copy.values()]):
            # print('     ----- Possible crafting solution found!')
            return True

        if mats_copy[tool_mat] > 0:
            # print(f'    Still need {tool_mat}')
            mat_needed = tool_mat
        else:
            # print('     Still need wood')
            mat_needed = 'wood'
            
        possible = False
        for card in self.resources.cards:
            if (card.id not in id_set_copy) and ((card.material == mat_needed) or (card.material == 'wild')):
                possible = self.crafting_helper(tool_mat, card, id_set_copy, mats_copy)
                # print(f'   --- Possible = {possible}')
                if possible:
                    break
        return possible

       
class Deck():
    def __init__(self, contents):
        self.contents = contents
        self.create()
    
    def shuffle(self):
        """Shuffle this deck."""
        random.shuffle(self.cards)

    def draw(self, n) -> list:
        """Returns a list containing the top n cards from this deck, popped from the end.
        So the first card in the returned list was the first one drawn from the top, and so on."""
        drawn = []
        for x in range(n):
            if self.size() > 0:
                drawn.append(self.cards.pop())
            else:
                break
        return drawn
    
    def add(self, cards):
        "Appends each card in 'cards' (an iterable) one at a time to this pile's list of cards."
        for card in cards:
            self.cards.append(card)

    def create(self):
        self.cards = []

        card_id = 0
        for order, x in enumerate(self.contents):
            x['info']['order'] = order
            for i in range(x['count']):
                x['info']['id'] = card_id
                self.add([x['card'](**x['info'])])
                card_id += 1
                
        self.shuffle()
                
    def size(self):
        return len(self.cards)
        
                
class Pile():
    def __init__(self):
        self.cards = []  
    
    def add(self, cards: list):
        "Appends each card in 'cards' (an iterable) one at a time to this pile's list of cards."
        for card in cards:
            self.cards.append(card)
    
    def size(self):
        return len(self.cards)
    
    def draw_top(self) -> Card:
        "Returns the card object of the card popped from the end of this Pile's list."
        card = self.cards.pop()
        return card
    
    def peek(self) -> Card:
        "Returns the pointer for the top card object of this pile (at index -1)."
        return self.cards[-1]
    
class Position():
    def __init__(self):
        self.cards = []  
    
    def add(self, cards):
        "Appends each card in 'cards' (an iterable) one at a time to this pile's list of cards."
        for card in cards:
            self.cards.append(card)
    
    def size(self):
        return len(self.cards)

    def pick(self, name) -> Card:
        """Searches the position's cards for the first card it finds whose 'name' value
        matches the input 'name' value, pops that card from the list, and returns that card.
        If no card with that name is found, returns None."""
        for i, c in enumerate(self.cards):
            if c.name == name:
                self.cards.pop(i)
                return c