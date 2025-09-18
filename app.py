import gradio as gr
import random
import json
import os
import pickle
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import numpy as np
from huggingface_hub import HfApi, hf_hub_download, upload_file

# Card and Game Engine Classes
class Card:
    def __init__(self, suit: str, rank: str):
        self.suit = suit
        self.rank = rank
        self.value = self._get_value()
    
    def _get_value(self) -> int:
        if self.rank in ['J', 'Q', 'K']:
            return 10
        elif self.rank == 'A':
            return 11  # Will be adjusted in Hand class
        else:
            return int(self.rank)
    
    def __str__(self):
        return f"{self.rank}{self.suit[0]}"

class Deck:
    def __init__(self, num_decks: int = 6):
        self.num_decks = num_decks
        self.cards = []
        self.card_counter = None  # Will be set by GameState
        self.reset()
    
    def reset(self):
        suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
        ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
        
        self.cards = []
        for _ in range(self.num_decks):
            for suit in suits:
                for rank in ranks:
                    self.cards.append(Card(suit, rank))
        
        random.shuffle(self.cards)
        
        # Reset card counter when deck is shuffled
        if self.card_counter:
            self.card_counter.reset()
    
    def deal_card(self) -> Card:
        if len(self.cards) < 20:  # Reshuffle when deck gets low
            self.reset()
        
        card = self.cards.pop()
        
        # Update card counter
        if self.card_counter:
            self.card_counter.count_card(card)
        
        return card

class Hand:
    def __init__(self):
        self.cards: List[Card] = []
        self.bet = 0
        self.is_doubled = False
        self.is_split = False
        self.is_surrendered = False
        self.is_busted = False
        self.is_blackjack = False
        self.is_split_hand = False  # Indicates this is a hand created from splitting
    
    def add_card(self, card: Card):
        self.cards.append(card)
        self._check_blackjack()
        self._check_bust()
    
    def get_value(self) -> int:
        total = 0
        aces = 0
        
        for card in self.cards:
            if card.rank == 'A':
                aces += 1
                total += 11
            else:
                total += card.value
        
        # Adjust for aces
        while total > 21 and aces > 0:
            total -= 10
            aces -= 1
        
        return total
    
    def is_soft(self) -> bool:
        """Check if hand has a soft ace (ace counted as 11)"""
        total = sum(card.value for card in self.cards if card.rank != 'A')
        aces = sum(1 for card in self.cards if card.rank == 'A')
        
        if aces == 0:
            return False
        
        # If we can count at least one ace as 11 without busting, it's soft
        return total + aces + 10 <= 21
    
    def _check_blackjack(self):
        # Blackjack only counts on original 2-card hands, not split hands
        if len(self.cards) == 2 and self.get_value() == 21 and not self.is_split_hand:
            self.is_blackjack = True
    
    def _check_bust(self):
        if self.get_value() > 21:
            self.is_busted = True
    
    def can_split(self) -> bool:
        return (len(self.cards) == 2 and 
                self.cards[0].rank == self.cards[1].rank and 
                not self.is_split)
    
    def can_double(self) -> bool:
        return len(self.cards) == 2 and not self.is_doubled
    
    def can_surrender(self) -> bool:
        return len(self.cards) == 2 and not self.is_doubled and not self.is_split_hand
    
    def get_display(self, hide_first: bool = False) -> str:
        if hide_first and len(self.cards) > 0:
            cards_str = "XX " + " ".join(str(card) for card in self.cards[1:])
            return f"{cards_str} (Hidden)"
        else:
            cards_str = " ".join(str(card) for card in self.cards)
            value = self.get_value()
            status = ""
            if self.is_surrendered:
                status = " (SURRENDERED)"
            elif self.is_blackjack:
                status = " (BLACKJACK!)"
            elif self.is_busted:
                status = " (BUST!)"
            elif self.is_soft() and value <= 21:
                status = " (Soft)"
            
            bet_info = f" [Bet: ${self.bet}]" if self.bet > 0 else ""
            return f"{cards_str} = {value}{status}{bet_info}"

# Dealer Class
class Dealer:
    def __init__(self, deck: Deck):
        self.deck = deck
        self.hand = Hand()
    
    def should_hit(self) -> bool:
        """Dealer hits on soft 17"""
        value = self.hand.get_value()
        if value < 17:
            return True
        elif value == 17 and self.hand.is_soft():
            return True
        else:
            return False
    
    def play_hand(self):
        """Play dealer's hand according to rules"""
        while self.should_hit():
            card = self.deck.deal_card()
            self.hand.add_card(card)
    
    def new_hand(self):
        self.hand = Hand()

# Player Class
@dataclass
class PlayerState:
    bank: int = 1000
    hands: List[Hand] = None
    current_bet: int = 0
    current_hand_index: int = 0
    
    def __post_init__(self):
        if self.hands is None:
            self.hands = [Hand()]
    
    def get_current_hand(self) -> Hand:
        return self.hands[self.current_hand_index]
    
    def has_more_hands(self) -> bool:
        return self.current_hand_index < len(self.hands) - 1
    
    def next_hand(self):
        if self.has_more_hands():
            self.current_hand_index += 1
    
    def reset_hands(self):
        self.hands = [Hand()]
        self.current_hand_index = 0

# Card Counting System
class CardCounter:
    def __init__(self):
        self.running_count = 0
        self.cards_dealt = 0
        self.decks = 6
    
    def count_card(self, card: Card):
        """Update count based on Hi-Lo counting system"""
        if card.rank in ['2', '3', '4', '5', '6']:
            self.running_count += 1
        elif card.rank in ['10', 'J', 'Q', 'K', 'A']:
            self.running_count -= 1
        # 7, 8, 9 are neutral (0)
        
        self.cards_dealt += 1
    
    def get_true_count(self) -> float:
        """Calculate true count (running count / decks remaining)"""
        decks_remaining = self.decks - (self.cards_dealt / 52)
        if decks_remaining <= 0:
            decks_remaining = 0.5  # Minimum to avoid division by zero
        return self.running_count / decks_remaining
    
    def reset(self):
        """Reset count for new shoe"""
        self.running_count = 0
        self.cards_dealt = 0
    
    def get_display(self) -> str:
        """Get count display string"""
        true_count = self.get_true_count()
        return f"Count: {self.running_count:+} | True: {true_count:+.1f}"

# Basic Strategy Bot
class BasicStrategyBot:
    def __init__(self):
        self.bank = 1000
        self.games_played = 0
        self.wins = 0
        self.losses = 0
        
        # Surrender strategy (first priority when available)
        self.surrender_strategy = {
            # Player total: {dealer_upcard: should_surrender}
            16: {9: True, 10: True, 11: True},  # Surrender 16 vs 9, 10, A
            15: {10: True, 11: True},           # Surrender 15 vs 10, A
        }
        
        # Basic Strategy Charts
        # Hard hands (no ace or ace counted as 1)
        self.hard_strategy = {
            # Player total: {dealer_upcard: action}
            5: {2:'H', 3:'H', 4:'H', 5:'H', 6:'H', 7:'H', 8:'H', 9:'H', 10:'H', 11:'H'},
            6: {2:'H', 3:'H', 4:'H', 5:'H', 6:'H', 7:'H', 8:'H', 9:'H', 10:'H', 11:'H'},
            7: {2:'H', 3:'H', 4:'H', 5:'H', 6:'H', 7:'H', 8:'H', 9:'H', 10:'H', 11:'H'},
            8: {2:'H', 3:'H', 4:'H', 5:'H', 6:'H', 7:'H', 8:'H', 9:'H', 10:'H', 11:'H'},
            9: {2:'H', 3:'D', 4:'D', 5:'D', 6:'D', 7:'H', 8:'H', 9:'H', 10:'H', 11:'H'},
            10: {2:'D', 3:'D', 4:'D', 5:'D', 6:'D', 7:'D', 8:'D', 9:'D', 10:'H', 11:'H'},
            11: {2:'D', 3:'D', 4:'D', 5:'D', 6:'D', 7:'D', 8:'D', 9:'D', 10:'D', 11:'D'},
            12: {2:'H', 3:'H', 4:'S', 5:'S', 6:'S', 7:'H', 8:'H', 9:'H', 10:'H', 11:'H'},
            13: {2:'S', 3:'S', 4:'S', 5:'S', 6:'S', 7:'H', 8:'H', 9:'H', 10:'H', 11:'H'},
            14: {2:'S', 3:'S', 4:'S', 5:'S', 6:'S', 7:'H', 8:'H', 9:'H', 10:'H', 11:'H'},
            15: {2:'S', 3:'S', 4:'S', 5:'S', 6:'S', 7:'H', 8:'H', 9:'H', 10:'H', 11:'H'},
            16: {2:'S', 3:'S', 4:'S', 5:'S', 6:'S', 7:'H', 8:'H', 9:'H', 10:'H', 11:'H'},
            17: {2:'S', 3:'S', 4:'S', 5:'S', 6:'S', 7:'S', 8:'S', 9:'S', 10:'S', 11:'S'},
            18: {2:'S', 3:'S', 4:'S', 5:'S', 6:'S', 7:'S', 8:'S', 9:'S', 10:'S', 11:'S'},
            19: {2:'S', 3:'S', 4:'S', 5:'S', 6:'S', 7:'S', 8:'S', 9:'S', 10:'S', 11:'S'},
            20: {2:'S', 3:'S', 4:'S', 5:'S', 6:'S', 7:'S', 8:'S', 9:'S', 10:'S', 11:'S'},
            21: {2:'S', 3:'S', 4:'S', 5:'S', 6:'S', 7:'S', 8:'S', 9:'S', 10:'S', 11:'S'},
        }
        
        # Soft hands (ace counted as 11)
        self.soft_strategy = {
            # Ace + card value: {dealer_upcard: action}
            2: {2:'H', 3:'H', 4:'H', 5:'D', 6:'D', 7:'H', 8:'H', 9:'H', 10:'H', 11:'H'},  # A,2
            3: {2:'H', 3:'H', 4:'H', 5:'D', 6:'D', 7:'H', 8:'H', 9:'H', 10:'H', 11:'H'},  # A,3
            4: {2:'H', 3:'H', 4:'D', 5:'D', 6:'D', 7:'H', 8:'H', 9:'H', 10:'H', 11:'H'},  # A,4
            5: {2:'H', 3:'H', 4:'D', 5:'D', 6:'D', 7:'H', 8:'H', 9:'H', 10:'H', 11:'H'},  # A,5
            6: {2:'H', 3:'D', 4:'D', 5:'D', 6:'D', 7:'H', 8:'H', 9:'H', 10:'H', 11:'H'},  # A,6
            7: {2:'S', 3:'D', 4:'D', 5:'D', 6:'D', 7:'S', 8:'S', 9:'H', 10:'H', 11:'H'},  # A,7
            8: {2:'S', 3:'S', 4:'S', 5:'S', 6:'S', 7:'S', 8:'S', 9:'S', 10:'S', 11:'S'},  # A,8
            9: {2:'S', 3:'S', 4:'S', 5:'S', 6:'S', 7:'S', 8:'S', 9:'S', 10:'S', 11:'S'},  # A,9
        }
        
        # Pair splitting strategy
        self.pair_strategy = {
            # Pair value: {dealer_upcard: action} - P=split, else follow hard/soft strategy
            2: {2:'P', 3:'P', 4:'P', 5:'P', 6:'P', 7:'P', 8:'H', 9:'H', 10:'H', 11:'H'},
            3: {2:'P', 3:'P', 4:'P', 5:'P', 6:'P', 7:'P', 8:'H', 9:'H', 10:'H', 11:'H'},
            4: {2:'H', 3:'H', 4:'H', 5:'P', 6:'P', 7:'H', 8:'H', 9:'H', 10:'H', 11:'H'},
            5: {2:'D', 3:'D', 4:'D', 5:'D', 6:'D', 7:'D', 8:'D', 9:'D', 10:'H', 11:'H'},
            6: {2:'P', 3:'P', 4:'P', 5:'P', 6:'P', 7:'H', 8:'H', 9:'H', 10:'H', 11:'H'},
            7: {2:'P', 3:'P', 4:'P', 5:'P', 6:'P', 7:'P', 8:'H', 9:'H', 10:'H', 11:'H'},
            8: {2:'P', 3:'P', 4:'P', 5:'P', 6:'P', 7:'P', 8:'P', 9:'P', 10:'P', 11:'P'},
            9: {2:'P', 3:'P', 4:'P', 5:'P', 6:'P', 7:'S', 8:'P', 9:'P', 10:'S', 11:'S'},
            10: {2:'S', 3:'S', 4:'S', 5:'S', 6:'S', 7:'S', 8:'S', 9:'S', 10:'S', 11:'S'},
            11: {2:'P', 3:'P', 4:'P', 5:'P', 6:'P', 7:'P', 8:'P', 9:'P', 10:'P', 11:'P'},
        }
    
    def get_action(self, hand: Hand, dealer_upcard: Card) -> str:
        """Get optimal action based on basic strategy"""
        dealer_value = dealer_upcard.value if dealer_upcard.value <= 10 else 10
        if dealer_upcard.rank == 'A':
            dealer_value = 11
        
        # Check for surrender first (only on initial two-card hands)
        if hand.can_surrender():
            hand_value = hand.get_value()
            if hand_value in self.surrender_strategy:
                if self.surrender_strategy[hand_value].get(dealer_value, False):
                    return 'surrender'
        
        # Check for pair splitting 
        if hand.can_split():
            pair_value = hand.cards[0].value if hand.cards[0].value <= 10 else 10
            if hand.cards[0].rank == 'A':
                pair_value = 11
            
            action = self.pair_strategy.get(pair_value, {}).get(dealer_value, 'H')
            if action == 'P' and self.bank >= hand.bet:
                return 'split'
        
        # Check if hand is soft (ace counted as 11)
        if hand.is_soft():
            # Find the non-ace card value
            other_card_value = 0
            for card in hand.cards:
                if card.rank != 'A':
                    other_card_value += card.value if card.value <= 10 else 10
            
            if other_card_value <= 9:  # A,2 through A,9
                action = self.soft_strategy.get(other_card_value, {}).get(dealer_value, 'H')
            else:  # A,10 = 21, stand
                action = 'S'
        else:
            # Hard hand
            hand_value = hand.get_value()
            action = self.hard_strategy.get(hand_value, {}).get(dealer_value, 'H')
        
        # Convert action codes to strings
        if action == 'H':
            return 'hit'
        elif action == 'S':
            return 'stand'
        elif action == 'D':
            if hand.can_double() and self.bank >= hand.bet:
                return 'double'
            else:
                return 'hit'  # If can't double, hit
        
        return 'stand'  # Default fallback
    
    def get_stats(self) -> str:
        if self.games_played == 0:
            return "Strategy Bot: No games played yet"
        
        win_rate = (self.wins / self.games_played) * 100
        return f"Strategy Bot: {self.games_played} games, {win_rate:.1f}% win rate, Bank: ${self.bank}"

# AI Bot with Learning
class BlackjackBot:
    def __init__(self):
        self.bank = 1000
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.epsilon = 0.1  # Exploration rate
        self.games_played = 0
        self.wins = 0
        self.losses = 0
        
        # Hugging Face Hub configuration
        self.repo_id = "kjmetzler/BlackJack"  # Your space name
        self.model_filename = "bot_model.pkl"
        
        # Try to get HF token from environment (for authenticated uploads)
        self.hf_token = os.environ.get("HF_TOKEN")
        
        # Load saved model if it exists
        self.load_model()
    
    def get_state_key(self, hand: Hand, dealer_upcard: Card) -> str:
        """Convert game state to string key for Q-table"""
        player_value = hand.get_value()
        is_soft = hand.is_soft()
        can_double = hand.can_double()
        can_split = hand.can_split()
        
        return f"{player_value}_{is_soft}_{dealer_upcard.value}_{can_double}_{can_split}"
    
    def get_action(self, hand: Hand, dealer_upcard: Card) -> str:
        """Choose action using epsilon-greedy strategy"""
        state_key = self.get_state_key(hand, dealer_upcard)
        
        # Get available actions
        actions = ['hit', 'stand']
        if hand.can_double() and self.bank >= hand.bet:
            actions.append('double')
        if hand.can_split() and self.bank >= hand.bet:
            actions.append('split')
        if hand.can_surrender():
            actions.append('surrender')
        
        # Epsilon-greedy exploration
        if random.random() < self.epsilon:
            return random.choice(actions)
        
        # Choose best known action
        q_values = self.q_table[state_key]
        if not q_values:
            return 'stand'  # Default action
        
        # Only consider available actions
        available_q_values = {action: q_values[action] for action in actions if action in q_values}
        if not available_q_values:
            return random.choice(actions)
        
        return max(available_q_values.items(), key=lambda x: x[1])[0]
    
    def update_q_value(self, state_key: str, action: str, reward: float, next_state_key: str = None):
        """Update Q-value using Q-learning"""
        current_q = self.q_table[state_key][action]
        
        if next_state_key:
            max_next_q = max(self.q_table[next_state_key].values()) if self.q_table[next_state_key] else 0
            new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        else:
            new_q = current_q + self.learning_rate * (reward - current_q)
        
        self.q_table[state_key][action] = new_q
    
    def learn_from_game(self, states_actions: List[Tuple[str, str]], final_reward: float):
        """Learn from completed game"""
        for i, (state_key, action) in enumerate(reversed(states_actions)):
            # Discount reward based on how far from end the action was
            discounted_reward = final_reward * (self.discount_factor ** i)
            self.update_q_value(state_key, action, discounted_reward)
    
    def save_model(self):
        """Save Q-table and stats to Hugging Face Hub"""
        model_data = {
            'q_table': dict(self.q_table),
            'games_played': self.games_played,
            'wins': self.wins,
            'losses': self.losses,
            'bank': self.bank
        }
        
        try:
            # Save locally first
            with open(self.model_filename, 'wb') as f:
                pickle.dump(model_data, f)
            
            # Upload to Hugging Face Hub
            try:
                api = HfApi(token=self.hf_token)
                api.upload_file(
                    path_or_fileobj=self.model_filename,
                    path_in_repo=self.model_filename,
                    repo_id=self.repo_id,
                    repo_type="space",
                    commit_message=f"Update bot model - {self.games_played} games played"
                )
                print(f"Model saved to Hub: {self.games_played} games played")
            except Exception as hub_error:
                print(f"Warning: Could not save to Hub (will use local file): {hub_error}")
                
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def load_model(self):
        """Load Q-table and stats from Hugging Face Hub or local file"""
        try:
            # First, try to download from Hugging Face Hub
            try:
                model_path = hf_hub_download(
                    repo_id=self.repo_id,
                    filename=self.model_filename,
                    repo_type="space",
                    cache_dir="./cache",
                    token=self.hf_token
                )
                print("Loading model from Hugging Face Hub...")
                
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                self._loaded_from_hub = True
                    
            except Exception as hub_error:
                print(f"Could not load from Hub, trying local file: {hub_error}")
                
                # Fallback to local file
                if os.path.exists(self.model_filename):
                    with open(self.model_filename, 'rb') as f:
                        model_data = pickle.load(f)
                    print("Loaded model from local file")
                    self._loaded_from_hub = False
                else:
                    print("No saved model found, starting fresh")
                    return
            
            # Load the model data
            self.q_table = defaultdict(lambda: defaultdict(float))
            for state, actions in model_data.get('q_table', {}).items():
                for action, value in actions.items():
                    self.q_table[state][action] = value
                    
            self.games_played = model_data.get('games_played', 0)
            self.wins = model_data.get('wins', 0)
            self.losses = model_data.get('losses', 0)
            self.bank = model_data.get('bank', 1000)
            
            print(f"Model loaded successfully: {self.games_played} games played")
            
        except Exception as e:
            print(f"Error loading model, starting fresh: {e}")
    
    def get_stats(self) -> str:
        if self.games_played == 0:
            return "Learning Bot: No games played yet"
        
        win_rate = (self.wins / self.games_played) * 100
        model_source = "🌐 Hub" if hasattr(self, '_loaded_from_hub') else "💾 Local"
        return f"Learning Bot: {self.games_played} games, {win_rate:.1f}% win rate, Bank: ${self.bank} ({model_source})"

# Player Statistics Tracking
class PlayerStats:
    def __init__(self):
        self.games_played = 0
        self.wins = 0
        self.losses = 0
    
    def get_stats(self) -> str:
        if self.games_played == 0:
            return "Player: No games played yet"
        
        win_rate = (self.wins / self.games_played) * 100
        return f"Player: {self.games_played} games, {win_rate:.1f}% win rate"
    def __init__(self):
        self.bank = 1000
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.epsilon = 0.1  # Exploration rate
        self.games_played = 0
        self.wins = 0
        self.losses = 0
        
        # Hugging Face Hub configuration
        self.repo_id = "kjmetzler/BlackJack"  # Your space name
        self.model_filename = "bot_model.pkl"
        
        # Try to get HF token from environment (for authenticated uploads)
        self.hf_token = os.environ.get("HF_TOKEN")
        
        # Load saved model if it exists
        self.load_model()
    
    def get_state_key(self, hand: Hand, dealer_upcard: Card) -> str:
        """Convert game state to string key for Q-table"""
        player_value = hand.get_value()
        is_soft = hand.is_soft()
        can_double = hand.can_double()
        can_split = hand.can_split()
        
        return f"{player_value}_{is_soft}_{dealer_upcard.value}_{can_double}_{can_split}"
    
    def get_action(self, hand: Hand, dealer_upcard: Card) -> str:
        """Choose action using epsilon-greedy strategy"""
        state_key = self.get_state_key(hand, dealer_upcard)
        
        # Get available actions
        actions = ['hit', 'stand']
        if hand.can_double() and self.bank >= hand.bet:
            actions.append('double')
        if hand.can_split() and self.bank >= hand.bet:
            actions.append('split')
        if hand.can_surrender():
            actions.append('surrender')
        
        # Epsilon-greedy exploration
        if random.random() < self.epsilon:
            return random.choice(actions)
        
        # Choose best known action
        q_values = self.q_table[state_key]
        if not q_values:
            return 'stand'  # Default action
        
        # Only consider available actions
        available_q_values = {action: q_values[action] for action in actions if action in q_values}
        if not available_q_values:
            return random.choice(actions)
        
        return max(available_q_values.items(), key=lambda x: x[1])[0]
    
    def update_q_value(self, state_key: str, action: str, reward: float, next_state_key: str = None):
        """Update Q-value using Q-learning"""
        current_q = self.q_table[state_key][action]
        
        if next_state_key:
            max_next_q = max(self.q_table[next_state_key].values()) if self.q_table[next_state_key] else 0
            new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        else:
            new_q = current_q + self.learning_rate * (reward - current_q)
        
        self.q_table[state_key][action] = new_q
    
    def learn_from_game(self, states_actions: List[Tuple[str, str]], final_reward: float):
        """Learn from completed game"""
        for i, (state_key, action) in enumerate(reversed(states_actions)):
            # Discount reward based on how far from end the action was
            discounted_reward = final_reward * (self.discount_factor ** i)
            self.update_q_value(state_key, action, discounted_reward)
    
    def save_model(self):
        """Save Q-table and stats to Hugging Face Hub"""
        model_data = {
            'q_table': dict(self.q_table),
            'games_played': self.games_played,
            'wins': self.wins,
            'losses': self.losses,
            'bank': self.bank
        }
        
        try:
            # Save locally first
            with open(self.model_filename, 'wb') as f:
                pickle.dump(model_data, f)
            
            # Upload to Hugging Face Hub
            try:
                api = HfApi(token=self.hf_token)
                api.upload_file(
                    path_or_fileobj=self.model_filename,
                    path_in_repo=self.model_filename,
                    repo_id=self.repo_id,
                    repo_type="space",
                    commit_message=f"Update bot model - {self.games_played} games played"
                )
                print(f"Model saved to Hub: {self.games_played} games played")
            except Exception as hub_error:
                print(f"Warning: Could not save to Hub (will use local file): {hub_error}")
                
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def load_model(self):
        """Load Q-table and stats from Hugging Face Hub or local file"""
        try:
            # First, try to download from Hugging Face Hub
            try:
                model_path = hf_hub_download(
                    repo_id=self.repo_id,
                    filename=self.model_filename,
                    repo_type="space",
                    cache_dir="./cache",
                    token=self.hf_token
                )
                print("Loading model from Hugging Face Hub...")
                
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                self._loaded_from_hub = True
                    
            except Exception as hub_error:
                print(f"Could not load from Hub, trying local file: {hub_error}")
                
                # Fallback to local file
                if os.path.exists(self.model_filename):
                    with open(self.model_filename, 'rb') as f:
                        model_data = pickle.load(f)
                    print("Loaded model from local file")
                    self._loaded_from_hub = False
                else:
                    print("No saved model found, starting fresh")
                    return
            
            # Load the model data
            self.q_table = defaultdict(lambda: defaultdict(float))
            for state, actions in model_data.get('q_table', {}).items():
                for action, value in actions.items():
                    self.q_table[state][action] = value
                    
            self.games_played = model_data.get('games_played', 0)
            self.wins = model_data.get('wins', 0)
            self.losses = model_data.get('losses', 0)
            self.bank = model_data.get('bank', 1000)
            
            print(f"Model loaded successfully: {self.games_played} games played")
            
        except Exception as e:
            print(f"Error loading model, starting fresh: {e}")
    
    def get_stats(self) -> str:
        if self.games_played == 0:
            return "Bot Stats: No games played yet"
        
        win_rate = (self.wins / self.games_played) * 100
        model_source = "🌐 Hub" if hasattr(self, '_loaded_from_hub') else "💾 Local"
        return f"Bot Stats: {self.games_played} games, {win_rate:.1f}% win rate, Bank: ${self.bank} ({model_source})"

# Game State Management
class GameState:
    def __init__(self):
        self.deck = Deck(6)
        self.dealer = Dealer(self.deck)
        self.player = PlayerState()
        self.player_stats = PlayerStats()
        self.bot = BlackjackBot()
        self.bot_hand = Hand()  # Learning bot gets its own hand from the start
        self.strategy_bot = BasicStrategyBot()
        self.strategy_bot_hand = Hand()  # Strategy bot gets its own hand
        self.card_counter = CardCounter()
        
        # Connect card counter to deck
        self.deck.card_counter = self.card_counter
        
        self.game_phase = "betting"  # betting, playing, dealer, results
        self.message = "Place your bets!"
        self.bot_states_actions = []  # For learning
        self.bot_last_action = ""  # Track bot's current move
        self.strategy_bot_last_action = ""  # Track strategy bot's current move
        self.bot_is_thinking = False
    
    def reset_banks(self):
        self.player.bank = 1000
        self.bot.bank = 1000
        self.strategy_bot.bank = 1000
        return "Banks reset to $1000!"

# Global game state
game_state = GameState()

# Game Logic Functions
def start_new_game():
    """Start a new round of blackjack"""
    game_state.dealer.new_hand()
    game_state.player.reset_hands()
    game_state.bot_hand = Hand()
    game_state.strategy_bot_hand = Hand()
    game_state.game_phase = "betting"
    game_state.message = "Place your bets!"
    game_state.bot_states_actions = []
    game_state.bot_last_action = ""
    game_state.strategy_bot_last_action = ""
    game_state.bot_is_thinking = False
    
    return update_display()

def place_bet(bet_amount: int):
    """Place bet for player and both bots"""
    if game_state.game_phase != "betting":
        return update_display()
    
    if (bet_amount > game_state.player.bank or 
        bet_amount > game_state.bot.bank or 
        bet_amount > game_state.strategy_bot.bank):
        game_state.message = "Insufficient funds!"
        return update_display()
    
    if bet_amount <= 0:
        game_state.message = "Bet must be greater than 0!"
        return update_display()
    
    # Place bets
    game_state.player.get_current_hand().bet = bet_amount
    game_state.player.bank -= bet_amount
    game_state.bot_hand.bet = bet_amount
    game_state.bot.bank -= bet_amount
    game_state.strategy_bot_hand.bet = bet_amount
    game_state.strategy_bot.bank -= bet_amount
    
    # Deal initial cards (2 to each player, 2 to dealer)
    # First card to each player
    game_state.player.get_current_hand().add_card(game_state.deck.deal_card())
    game_state.bot_hand.add_card(game_state.deck.deal_card())
    game_state.strategy_bot_hand.add_card(game_state.deck.deal_card())
    game_state.dealer.hand.add_card(game_state.deck.deal_card())
    
    # Second card to each player  
    game_state.player.get_current_hand().add_card(game_state.deck.deal_card())
    game_state.bot_hand.add_card(game_state.deck.deal_card())
    game_state.strategy_bot_hand.add_card(game_state.deck.deal_card())
    game_state.dealer.hand.add_card(game_state.deck.deal_card())
    
    game_state.game_phase = "playing"
    game_state.message = f"Cards dealt! Make your move! (Bet: ${bet_amount})"
    
    return update_display()

def player_hit():
    """Player chooses to hit"""
    if game_state.game_phase != "playing" or game_state.bot_is_thinking:
        return update_display()
    
    current_hand = game_state.player.get_current_hand()
    current_hand.add_card(game_state.deck.deal_card())
    
    if current_hand.is_busted:
        game_state.message = f"Hand {game_state.player.current_hand_index + 1} busted!"
        return handle_next_player_hand()
    else:
        game_state.message = f"Card dealt to hand {game_state.player.current_hand_index + 1}! Choose your next move."
    
    return update_display()

def player_stand():
    """Player chooses to stand"""
    if game_state.game_phase != "playing" or game_state.bot_is_thinking:
        return update_display()
    
    game_state.message = f"Hand {game_state.player.current_hand_index + 1} stands!"
    return handle_next_player_hand()

def player_double():
    """Player chooses to double down"""
    if game_state.game_phase != "playing" or game_state.bot_is_thinking:
        return update_display()
    
    current_hand = game_state.player.get_current_hand()
    
    if not current_hand.can_double():
        game_state.message = "Cannot double down!"
        return update_display()
    
    if current_hand.bet > game_state.player.bank:
        game_state.message = "Insufficient funds to double down!"
        return update_display()
    
    # Double the bet
    game_state.player.bank -= current_hand.bet
    current_hand.bet *= 2
    current_hand.is_doubled = True
    
    # Deal one more card
    current_hand.add_card(game_state.deck.deal_card())
    
    if current_hand.is_busted:
        game_state.message = f"Hand {game_state.player.current_hand_index + 1} doubled and busted!"
    else:
        game_state.message = f"Hand {game_state.player.current_hand_index + 1} doubled down!"
    
    return handle_next_player_hand()

def player_split():
    """Player chooses to split"""
    if game_state.game_phase != "playing" or game_state.bot_is_thinking:
        return update_display()
    
    current_hand = game_state.player.get_current_hand()
    
    if not current_hand.can_split():
        game_state.message = "Cannot split!"
        return update_display()
    
    if current_hand.bet > game_state.player.bank:
        game_state.message = "Insufficient funds to split!"
        return update_display()
    
    # Create new hand from split
    new_hand = Hand()
    new_hand.bet = current_hand.bet
    new_hand.is_split_hand = True
    current_hand.is_split_hand = True
    
    # Move second card to new hand
    new_hand.add_card(current_hand.cards.pop())
    
    # Deal new cards to both hands
    current_hand.add_card(game_state.deck.deal_card())
    new_hand.add_card(game_state.deck.deal_card())
    
    # Add new hand to player
    game_state.player.hands.insert(game_state.player.current_hand_index + 1, new_hand)
    game_state.player.bank -= current_hand.bet
    
    current_hand.is_split = True
    game_state.message = f"Hand split! Playing hand {game_state.player.current_hand_index + 1} of {len(game_state.player.hands)}."
    
    return update_display()

def player_surrender():
    """Player chooses to surrender"""
    if game_state.game_phase != "playing" or game_state.bot_is_thinking:
        return update_display()
    
    current_hand = game_state.player.get_current_hand()
    
    if not current_hand.can_surrender():
        game_state.message = "Cannot surrender!"
        return update_display()
    
    current_hand.is_surrendered = True
    game_state.player.bank += current_hand.bet // 2  # Get half bet back
    
    game_state.message = f"Hand {game_state.player.current_hand_index + 1} surrendered!"
    return handle_next_player_hand()

def handle_next_player_hand():
    """Handle moving to next hand or finishing player turn"""
    if game_state.player.has_more_hands():
        game_state.player.next_hand()
        game_state.message = f"Playing hand {game_state.player.current_hand_index + 1} of {len(game_state.player.hands)}."
        return update_display()
    else:
        # Player finished, now complete the rest of the game
        complete_game_round()
        return update_display()

def complete_game_round():
    """Complete both bots play, dealer play, and results calculation"""
    # Learning Bot plays first
    states_actions = []
    game_state.bot_last_action = "THINKING..."
    
    # Learning bot makes decisions
    while (not game_state.bot_hand.is_busted and 
           not game_state.bot_hand.is_blackjack and 
           not game_state.bot_hand.is_surrendered):
        
        state_key = game_state.bot.get_state_key(game_state.bot_hand, game_state.dealer.hand.cards[0])
        action = game_state.bot.get_action(game_state.bot_hand, game_state.dealer.hand.cards[0])
        
        states_actions.append((state_key, action))
        game_state.bot_last_action = action.upper()
        
        if action == "hit":
            game_state.bot_hand.add_card(game_state.deck.deal_card())
        elif action == "stand":
            break
        elif action == "double" and game_state.bot_hand.can_double() and game_state.bot.bank >= game_state.bot_hand.bet:
            game_state.bot.bank -= game_state.bot_hand.bet
            game_state.bot_hand.bet *= 2
            game_state.bot_hand.is_doubled = True
            game_state.bot_hand.add_card(game_state.deck.deal_card())
            break
        elif action == "surrender" and game_state.bot_hand.can_surrender():
            game_state.bot_hand.is_surrendered = True
            game_state.bot.bank += game_state.bot_hand.bet // 2
            break
        else:
            break
    
    game_state.bot_states_actions = states_actions
    
    # Strategy Bot plays
    game_state.strategy_bot_last_action = "THINKING..."
    
    # Strategy bot makes decisions
    while (not game_state.strategy_bot_hand.is_busted and 
           not game_state.strategy_bot_hand.is_blackjack and 
           not game_state.strategy_bot_hand.is_surrendered):
        
        action = game_state.strategy_bot.get_action(game_state.strategy_bot_hand, game_state.dealer.hand.cards[0])
        game_state.strategy_bot_last_action = action.upper()
        
        if action == "hit":
            game_state.strategy_bot_hand.add_card(game_state.deck.deal_card())
        elif action == "stand":
            break
        elif action == "double" and game_state.strategy_bot_hand.can_double() and game_state.strategy_bot.bank >= game_state.strategy_bot_hand.bet:
            game_state.strategy_bot.bank -= game_state.strategy_bot_hand.bet
            game_state.strategy_bot_hand.bet *= 2
            game_state.strategy_bot_hand.is_doubled = True
            game_state.strategy_bot_hand.add_card(game_state.deck.deal_card())
            break
        elif action == "split" and game_state.strategy_bot_hand.can_split() and game_state.strategy_bot.bank >= game_state.strategy_bot_hand.bet:
            # For simplicity, strategy bot will just hit after splitting
            game_state.strategy_bot_hand.add_card(game_state.deck.deal_card())
            break
        else:
            break
    
    # Dealer plays
    while game_state.dealer.should_hit():
        card = game_state.deck.deal_card()
        game_state.dealer.hand.add_card(card)
    
    # Calculate results
    dealer_value = game_state.dealer.hand.get_value()
    
    # Calculate results for player hands
    player_results = []
    player_won = False
    for i, hand in enumerate(game_state.player.hands):
        result = get_hand_result(hand, dealer_value)
        winnings = calculate_winnings(hand, result)
        game_state.player.bank += winnings
        net_winnings = winnings - hand.bet
        player_results.append(f"Hand {i+1}: {result} (${net_winnings:+})")
        if result == "won":
            player_won = True
    
    # Update player stats
    game_state.player_stats.games_played += 1
    if player_won or any("pushed" in result for result in player_results):
        if player_won:
            game_state.player_stats.wins += 1
    else:
        game_state.player_stats.losses += 1
    
    # Calculate learning bot results
    bot_result = get_hand_result(game_state.bot_hand, dealer_value)
    bot_winnings = calculate_winnings(game_state.bot_hand, bot_result)
    game_state.bot.bank += bot_winnings
    
    # Update learning bot stats and learning
    game_state.bot.games_played += 1
    if bot_result == "won":
        reward = 1.0
        game_state.bot.wins += 1
    elif bot_result == "lost":
        reward = -1.0
        game_state.bot.losses += 1
    else:  # push
        reward = 0.0
    
    game_state.bot.learn_from_game(game_state.bot_states_actions, reward)
    game_state.bot.save_model()
    
    # Calculate strategy bot results
    strategy_bot_result = get_hand_result(game_state.strategy_bot_hand, dealer_value)
    strategy_bot_winnings = calculate_winnings(game_state.strategy_bot_hand, strategy_bot_result)
    game_state.strategy_bot.bank += strategy_bot_winnings
    
    # Update strategy bot stats
    game_state.strategy_bot.games_played += 1
    if strategy_bot_result == "won":
        game_state.strategy_bot.wins += 1
    elif strategy_bot_result == "lost":
        game_state.strategy_bot.losses += 1
    
    # Update message
    player_summary = " | ".join(player_results)
    bot_net = bot_winnings - game_state.bot_hand.bet
    strategy_bot_net = strategy_bot_winnings - game_state.strategy_bot_hand.bet
    
    game_state.message = (f"RESULTS - Player: {player_summary} | "
                         f"Learning Bot: {bot_result} (${bot_net:+}) | "
                         f"Strategy Bot: {strategy_bot_result} (${strategy_bot_net:+}) | "
                         f"Dealer: {dealer_value}")
    
    game_state.game_phase = "results"
    game_state.bot_last_action = ""
    game_state.strategy_bot_last_action = ""

def get_hand_result(hand: Hand, dealer_value: int) -> str:
    """Determine if hand wins, loses, or pushes"""
    if hand.is_surrendered:
        return "surrendered"
    
    hand_value = hand.get_value()
    dealer_busted = dealer_value > 21
    
    if hand.is_busted:
        return "lost"
    elif dealer_busted:
        return "won"
    elif hand_value > dealer_value:
        return "won"
    elif hand_value < dealer_value:
        return "lost"
    else:
        return "pushed"

def calculate_winnings(hand: Hand, result: str) -> int:
    """Calculate winnings based on result"""
    if result == "won":
        if hand.is_blackjack:
            return int(hand.bet * 2.5)  # Blackjack pays 3:2
        else:
            return hand.bet * 2  # Normal win pays 1:1
    elif result == "pushed":
        return hand.bet  # Return original bet
    elif result == "surrendered":
        return hand.bet // 2  # Return half bet
    else:
        return 0  # Lose bet

def reset_banks():
    """Reset player and both bot banks"""
    game_state.player.bank = 1000
    game_state.bot.bank = 1000
    game_state.strategy_bot.bank = 1000
    game_state.message = "All banks reset to $1000!"
    return update_display()

def update_display():
    """Update all display components"""
    # Dealer display
    dealer_display = game_state.dealer.hand.get_display(
        hide_first=(game_state.game_phase == "playing")
    )
    
    # Player display (handle multiple hands)
    if len(game_state.player.hands) == 1:
        player_display = game_state.player.hands[0].get_display()
    else:
        hand_displays = []
        for i, hand in enumerate(game_state.player.hands):
            marker = "👉 " if i == game_state.player.current_hand_index and game_state.game_phase == "playing" else ""
            hand_displays.append(f"{marker}Hand {i+1}: {hand.get_display()}")
        player_display = "\n".join(hand_displays)
    
    # Learning Bot display with action indicator
    bot_action_text = f" (Last: {game_state.bot_last_action})" if game_state.bot_last_action else ""
    thinking_text = " (Thinking...)" if game_state.bot_is_thinking else ""
    bot_display = game_state.bot_hand.get_display() + bot_action_text + thinking_text
    
    # Strategy Bot display with action indicator
    strategy_bot_action_text = f" (Last: {game_state.strategy_bot_last_action})" if game_state.strategy_bot_last_action else ""
    strategy_bot_display = game_state.strategy_bot_hand.get_display() + strategy_bot_action_text
    
    # Banks
    player_bank = f"Player Bank: ${game_state.player.bank}"
    bot_bank = f"Learning Bot Bank: ${game_state.bot.bank}"
    strategy_bot_bank = f"Strategy Bot Bank: ${game_state.strategy_bot.bank}"
    
    # Performance stats (both player and bots)
    player_stats = game_state.player_stats.get_stats()
    bot_stats = game_state.bot.get_stats()
    strategy_bot_stats = game_state.strategy_bot.get_stats()
    performance_display = f"{player_stats}\n{bot_stats}\n{strategy_bot_stats}"
    
    # Card count display
    count_display = game_state.card_counter.get_display()
    
    # Game controls visibility
    show_bet_controls = game_state.game_phase == "betting"
    show_play_controls = game_state.game_phase == "playing" and not game_state.bot_is_thinking
    show_new_game = game_state.game_phase in ["results", "betting"]
    
    # Advanced play controls (only show when relevant and it's player's turn)
    current_hand = None
    show_double = False
    show_split = False
    show_surrender = False
    
    if game_state.game_phase == "playing" and not game_state.bot_is_thinking:
        current_hand = game_state.player.get_current_hand()
        # Only show advanced options if current hand allows them and we have enough money
        show_double = (current_hand.can_double() and 
                      current_hand.bet <= game_state.player.bank)
        show_split = (current_hand.can_split() and 
                     current_hand.bet <= game_state.player.bank)  
        show_surrender = current_hand.can_surrender()
    
    return (
        dealer_display,
        player_display, 
        bot_display,
        strategy_bot_display,
        player_bank,
        bot_bank,
        strategy_bot_bank,
        performance_display,
        count_display,
        game_state.message,
        gr.update(visible=show_bet_controls),     # bet_row
        gr.update(visible=show_play_controls),    # play_row
        gr.update(visible=show_new_game),         # new_game_btn
        gr.update(visible=show_double),           # double_btn
        gr.update(visible=show_split),            # split_btn  
        gr.update(visible=show_surrender)         # surrender_btn
    )

# Gradio Interface
with gr.Blocks(title="BlackJack AI Trainer", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🃏 BlackJack AI Trainer")
    gr.Markdown("Train an AI bot by playing BlackJack! Watch both a **Learning Bot** (Q-learning) and **Strategy Bot** (basic strategy) compete!")
    gr.Markdown("💡 *Features card counting with Hi-Lo system. The Learning Bot saves progress to Hugging Face Hub.*")
    gr.Markdown("🆕 **Actions**: Split pairs, double down, surrender. **Tracking**: Player stats, bot performance, and card count!")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🎯 Dealer")
            dealer_cards = gr.Textbox(label="Dealer's Hand", value="", interactive=False)
            
            # Card Count Display
            gr.Markdown("### 📊 Card Count")
            count_display = gr.Textbox(label="Hi-Lo Count", value="Count: +0 | True: +0.0", interactive=False)
        
        with gr.Column():
            gr.Markdown("### 👤 Player")
            player_cards = gr.Textbox(label="Your Hand(s)", value="", interactive=False, lines=3)
            player_bank_display = gr.Textbox(label="Your Bank", value="Player Bank: $1000", interactive=False)
        
        with gr.Column():
            gr.Markdown("### 🤖 Learning Bot")
            bot_cards = gr.Textbox(label="Learning Bot's Hand", value="", interactive=False, lines=2)
            bot_bank_display = gr.Textbox(label="Learning Bot's Bank", value="Learning Bot Bank: $1000", interactive=False)
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🎯 Strategy Bot")
            strategy_bot_cards = gr.Textbox(label="Strategy Bot's Hand", value="", interactive=False, lines=2)
            strategy_bot_bank_display = gr.Textbox(label="Strategy Bot's Bank", value="Strategy Bot Bank: $1000", interactive=False)
    
    # Performance statistics for all players
    performance_display = gr.Textbox(label="Performance Stats", value="No games played yet", interactive=False, lines=3)
    
    # Game message
    message_display = gr.Textbox(label="Game Status", value="Click 'New Game' to start!", interactive=False, lines=2)
    
    # Betting controls
    with gr.Row(visible=False) as bet_row:
        bet_amount = gr.Number(label="Bet Amount ($)", value=10, minimum=1, maximum=100)
        place_bet_btn = gr.Button("Place Bet", variant="primary")
    
    # Basic playing controls  
    with gr.Row(visible=False) as play_row:
        hit_btn = gr.Button("Hit", variant="primary")
        stand_btn = gr.Button("Stand", variant="secondary")
    
    # Advanced playing controls
    with gr.Row():
        double_btn = gr.Button("Double Down", variant="secondary", visible=False)
        split_btn = gr.Button("Split Pair", variant="secondary", visible=False)  
        surrender_btn = gr.Button("Surrender", variant="secondary", visible=False)
    
    # Game management controls
    with gr.Row():
        new_game_btn = gr.Button("New Game", variant="primary")
        reset_banks_btn = gr.Button("Reset All Banks", variant="secondary")
    
    # All outputs for update functions
    all_outputs = [dealer_cards, player_cards, bot_cards, strategy_bot_cards, 
                   player_bank_display, bot_bank_display, strategy_bot_bank_display,
                   performance_display, count_display, message_display, 
                   bet_row, play_row, new_game_btn, double_btn, split_btn, surrender_btn]
    
    # Event handlers
    new_game_btn.click(start_new_game, outputs=all_outputs)
    
    place_bet_btn.click(place_bet, inputs=[bet_amount], outputs=all_outputs)
    
    hit_btn.click(player_hit, outputs=all_outputs)
    stand_btn.click(player_stand, outputs=all_outputs)
    double_btn.click(player_double, outputs=all_outputs)
    split_btn.click(player_split, outputs=all_outputs)
    surrender_btn.click(player_surrender, outputs=all_outputs)
    
    reset_banks_btn.click(reset_banks, outputs=all_outputs)
    
    # Initialize display
    demo.load(start_new_game, outputs=all_outputs)

if __name__ == "__main__":
    demo.launch()
