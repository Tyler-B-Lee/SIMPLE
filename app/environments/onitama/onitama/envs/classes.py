# self.contents = [
#             {'card': Card, 'info': {'name': 'Boar', 'priority': 'r', 'moves': {(-1,0),(0,1),(1,0)}}}  #0 
#             , {'card': Card, 'info': {'name': 'Dragon', 'priority': 'r', 'moves': {(-2,1),(-1,-1),(1,-1),(2,1)}}}  #1
#             , {'card': Card, 'info': {'name': 'Mantis', 'priority': 'r', 'moves': {(-1,1),(0,-1),(1,1)}}}  #2
#             , {'card': Card, 'info': {'name': 'Elephant', 'priority': 'r', 'moves': {(-1,0),(-1,1),(1,0),(1,1)}}}  #3
#             , {'card': Card, 'info': {'name': 'Crane', 'priority': 'b', 'moves': {(-1,-1),(0,1),(1,-1)}}}  #4
#             , {'card': Card, 'info': {'name': 'Monkey', 'priority': 'b', 'moves': {(-1,-1),(-1,1),(1,-1),(1,1)}}}  #5
#             , {'card': Card, 'info': {'name': 'Tiger', 'priority': 'b', 'moves': {(0,2),(0,-1)}}}  #6
#             , {'card': Card, 'info': {'name': 'Crab', 'priority': 'b', 'moves': {(-2,0),(0,1),(2,0)}}}  #7
#             , {'card': Card, 'info': {'name': 'Goose', 'priority': 'b', 'moves': {(-1,0),(-1,1),(1,0),(1,-1)}}}  #8
#             , {'card': Card, 'info': {'name': 'Eel', 'priority': 'b', 'moves': {(-1,-1),(-1,1),(1,0)}}}  #9
#             , {'card': Card, 'info': {'name': 'Horse', 'priority': 'r', 'moves': {(-1,0),(0,1),(0,-1)}}}  #10
#             , {'card': Card, 'info': {'name': 'Frog', 'priority': 'r', 'moves': {(-2,0),(-1,1),(1,-1)}}}  #11
#             , {'card': Card, 'info': {'name': 'Rooster', 'priority': 'r', 'moves': {(-1,0),(-1,-1),(1,0),(1,1)}}}  #12
#             , {'card': Card, 'info': {'name': 'Cobra', 'priority': 'r', 'moves': {(-1,0),(1,1),(1,-1)}}}  #13
#             , {'card': Card, 'info': {'name': 'Ox', 'priority': 'b', 'moves': {(0,1),(0,-1),(1,0)}}}  #14
#             , {'card': Card, 'info': {'name': 'Rabbit', 'priority': 'b', 'moves': {(-1,-1),(1,1),(2,0)}}}  #15
#         ]

import random

class Player():
    def __init__(self, color):
        self.color = color
        self.active_pieces = []
        self.card_a = None
        self.card_b = None
        self.symbol = "Red" if (self.color == 'r') else "Blue"


class Piece():
    def __init__(self, id: int, color: str, row, col) -> None:
        self.id = id
        self.is_master = True if (self.id == 2 or self.id == 7) else False
        self.color = color
        self.row = row
        self.col = col
        self.set_symbol()
    
    def move(self, row, col):
        self.row = row
        self.col = col
    
    def set_symbol(self):
        if self.color == 'r':
            if self.is_master:
                self.symbol = ' 🔥 R '
            else:
                self.symbol = f' 🔺r{self.id} '
        elif self.is_master:
            self.symbol = ' 💎 B '
        else:
            self.symbol = f' 🔷b{self.id} '


class Card():
    def __init__(self, id=int, name=str, priority=str, moves=list):
        self.id = id
        self.name = name
        self.priority = priority
        self.moves = moves

        self.symbol = f"{self.name} (ID {self.id}): {self.moves}"


class Deck():
    def __init__(self, contents):
        self.contents = contents
        self.create()
    
    def shuffle(self):
        random.shuffle(self.cards)

    def draw(self):
        """Pops the top card of the deck and returns it."""
        return self.cards.pop()
    
    def pick(self, n: str):
        "Given the name of a card, draws and returns that card from the deck. Returns None if not found."
        n = n.capitalize()
        for i, c in enumerate(self.cards):
            if (n in c.name):
                self.cards.pop(i)
                return c
    
    def add(self, cards):
        for card in cards:
            self.cards.append(card)

    def create(self):
        self.cards = []

        card_id = 0
        for x in self.contents:
            x['info']['id'] = card_id
            self.add([x['card'](**x['info'])])
            card_id += 1
                
        self.shuffle()