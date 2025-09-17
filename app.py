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
    
    def deal_card(self) -> Card:
        if len(self.cards) < 20:  # Reshuffle when deck gets low
            self.reset()
        return self.cards.pop()

class Hand:
    def __init__(self):
        self.cards: List[Card] = []
        self.bet = 0
        self.is_doubled = False
        self.is_split = False
        self.is_busted = False
        self.is_blackjack = False
    
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
        if len(self.cards) == 2 and self.get_value() == 21:
            self.is_blackjack = True
    
    def _check_bust(self):
        if self.get_value() > 21:
            self.is_busted = True
    
    def can_split(self) -> bool:
        return len(self.cards) == 2 and self.cards[0].value == self.cards[1].value
    
    def can_double(self) -> bool:
        return len(self.cards) == 2 and not self.is_doubled
    
    def get_display(self, hide_first: bool = False) -> str:
        if hide_first and len(self.cards) > 0:
            cards_str = "XX " + " ".join(str(card) for card in self.cards[1:])
            return f"{cards_str} (Hidden)"
        else:
            cards_str = " ".join(str(card) for card in self.cards)
            value = self.get_value()
            status = ""
            if self.is_blackjack:
                status = " (BLACKJACK!)"
            elif self.is_busted:
                status = " (BUST!)"
            elif self.is_soft() and value <= 21:
                status = " (Soft)"
            
            return f"{cards_str} = {value}{status}"

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
    
    def __post_init__(self):
        if self.hands is None:
            self.hands = [Hand()]

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
        
        # Epsilon-greedy exploration
        if random.random() < self.epsilon:
            actions = ['hit', 'stand']
            if hand.can_double():
                actions.append('double')
            if hand.can_split():
                actions.append('split')
            return random.choice(actions)
        
        # Choose best known action
        q_values = self.q_table[state_key]
        if not q_values:
            return 'stand'  # Default action
        
        return max(q_values.items(), key=lambda x: x[1])[0]
    
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
        self.bot = BlackjackBot()
        self.game_phase = "betting"  # betting, playing, dealer, results
        self.message = "Place your bets!"
        self.bot_states_actions = []  # For learning
    
    def reset_banks(self):
        self.player.bank = 1000
        self.bot.bank = 1000
        return "Banks reset to $1000!"

# Global game state
game_state = GameState()

# Game Logic Functions
def start_new_game():
    """Start a new round of blackjack"""
    game_state.dealer.new_hand()
    game_state.player.hands = [Hand()]
    game_state.game_phase = "betting"
    game_state.message = "Place your bets!"
    game_state.bot_states_actions = []
    
    return update_display()

def place_bet(bet_amount: int):
    """Place bet for both player and bot"""
    if game_state.game_phase != "betting":
        return update_display()
    
    if bet_amount > game_state.player.bank or bet_amount > game_state.bot.bank:
        game_state.message = "Insufficient funds!"
        return update_display()
    
    if bet_amount <= 0:
        game_state.message = "Bet must be greater than 0!"
        return update_display()
    
    # Place bets
    game_state.player.hands[0].bet = bet_amount
    game_state.player.bank -= bet_amount
    game_state.bot.bank -= bet_amount
    
    # Deal initial cards
    for _ in range(2):
        game_state.player.hands[0].add_card(game_state.deck.deal_card())
        game_state.dealer.hand.add_card(game_state.deck.deal_card())
    
    game_state.game_phase = "playing"
    game_state.message = f"Bet placed: ${bet_amount}. Make your move!"
    
    return update_display()

def player_hit():
    """Player chooses to hit"""
    if game_state.game_phase != "playing":
        return update_display()
    
    hand = game_state.player.hands[0]
    hand.add_card(game_state.deck.deal_card())
    
    if hand.is_busted:
        game_state.message = "Player busted! Bot's turn."
        bot_play()
    else:
        game_state.message = "Card dealt! Hit or Stand?"
    
    return update_display()

def player_stand():
    """Player chooses to stand"""
    if game_state.game_phase != "playing":
        return update_display()
    
    game_state.message = "Player stands! Bot's turn."
    bot_play()
    
    return update_display()

def bot_play():
    """Bot plays its hand"""
    bot_hand = Hand()
    bot_hand.bet = game_state.player.hands[0].bet
    
    # Deal bot's initial cards
    for _ in range(2):
        bot_hand.add_card(game_state.deck.deal_card())
    
    states_actions = []
    
    # Bot plays using its strategy
    while not bot_hand.is_busted and not bot_hand.is_blackjack:
        state_key = game_state.bot.get_state_key(bot_hand, game_state.dealer.hand.cards[0])
        action = game_state.bot.get_action(bot_hand, game_state.dealer.hand.cards[0])
        
        states_actions.append((state_key, action))
        
        if action == "hit":
            bot_hand.add_card(game_state.deck.deal_card())
        elif action == "stand":
            break
        elif action == "double" and bot_hand.can_double():
            bot_hand.add_card(game_state.deck.deal_card())
            bot_hand.is_doubled = True
            game_state.bot.bank -= bot_hand.bet
            bot_hand.bet *= 2
            break
    
    game_state.bot_states_actions = states_actions
    
    # Store bot hand for results
    game_state.bot_hand = bot_hand
    
    # Now dealer plays
    game_state.dealer.play_hand()
    
    # Calculate results
    calculate_results()

def calculate_results():
    """Calculate and display game results"""
    dealer_value = game_state.dealer.hand.get_value()
    player_hand = game_state.player.hands[0]
    bot_hand = game_state.bot_hand
    
    # Calculate player results
    player_result = get_hand_result(player_hand, dealer_value)
    player_winnings = calculate_winnings(player_hand, player_result)
    game_state.player.bank += player_winnings
    
    # Calculate bot results
    bot_result = get_hand_result(bot_hand, dealer_value)
    bot_winnings = calculate_winnings(bot_hand, bot_result)
    game_state.bot.bank += bot_winnings
    
    # Update bot learning
    if bot_result == "win":
        reward = 1.0
        game_state.bot.wins += 1
    elif bot_result == "lose":
        reward = -1.0
        game_state.bot.losses += 1
    else:  # push
        reward = 0.0
    
    game_state.bot.games_played += 1
    game_state.bot.learn_from_game(game_state.bot_states_actions, reward)
    game_state.bot.save_model()
    
    # Update message
    game_state.message = f"Results - Player: {player_result} (${player_winnings}), Bot: {bot_result} (${bot_winnings})"
    game_state.game_phase = "results"

def get_hand_result(hand: Hand, dealer_value: int) -> str:
    """Determine if hand wins, loses, or pushes"""
    hand_value = hand.get_value()
    dealer_busted = dealer_value > 21
    
    if hand.is_busted:
        return "lose"
    elif dealer_busted:
        return "win"
    elif hand_value > dealer_value:
        return "win"
    elif hand_value < dealer_value:
        return "lose"
    else:
        return "push"

def calculate_winnings(hand: Hand, result: str) -> int:
    """Calculate winnings based on result"""
    if result == "win":
        if hand.is_blackjack:
            return int(hand.bet * 2.5)  # Blackjack pays 3:2
        else:
            return hand.bet * 2  # Normal win pays 1:1
    elif result == "push":
        return hand.bet  # Return original bet
    else:
        return 0  # Lose bet

def reset_banks():
    """Reset both player and bot banks"""
    game_state.player.bank = 1000
    game_state.bot.bank = 1000
    game_state.message = "Banks reset to $1000!"
    return update_display()

def update_display():
    """Update all display components"""
    # Dealer display
    dealer_display = game_state.dealer.hand.get_display(
        hide_first=(game_state.game_phase == "playing")
    )
    
    # Player display
    player_display = game_state.player.hands[0].get_display()
    
    # Bot display
    bot_display = ""
    if hasattr(game_state, 'bot_hand'):
        bot_display = game_state.bot_hand.get_display()
    
    # Banks
    player_bank = f"Player Bank: ${game_state.player.bank}"
    bot_bank = f"Bot Bank: ${game_state.bot.bank}"
    
    # Bot stats
    bot_stats = game_state.bot.get_stats()
    
    # Game controls visibility
    show_bet_controls = game_state.game_phase == "betting"
    show_play_controls = game_state.game_phase == "playing"
    show_new_game = game_state.game_phase in ["results", "betting"]
    
    return (
        dealer_display,
        player_display, 
        bot_display,
        player_bank,
        bot_bank,
        bot_stats,
        game_state.message,
        gr.update(visible=show_bet_controls),  # bet_row
        gr.update(visible=show_play_controls), # play_row
        gr.update(visible=show_new_game)       # new_game_btn
    )

# Gradio Interface
with gr.Blocks(title="BlackJack AI Trainer", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🃏 BlackJack AI Trainer")
    gr.Markdown("Train an AI bot by playing BlackJack! The bot learns from each game and **saves its progress to Hugging Face Hub** so learning persists across restarts.")
    gr.Markdown("💡 *The bot uses Q-learning to improve its strategy. 🌐 = loaded from Hub, 💾 = local file*")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🎯 Dealer")
            dealer_cards = gr.Textbox(label="Dealer's Hand", value="", interactive=False)
        
        with gr.Column():
            gr.Markdown("### 👤 Player")
            player_cards = gr.Textbox(label="Your Hand", value="", interactive=False)
            player_bank_display = gr.Textbox(label="Your Bank", value="Player Bank: $1000", interactive=False)
        
        with gr.Column():
            gr.Markdown("### 🤖 AI Bot")
            bot_cards = gr.Textbox(label="Bot's Hand", value="", interactive=False)
            bot_bank_display = gr.Textbox(label="Bot's Bank", value="Bot Bank: $1000", interactive=False)
    
    # Bot statistics
    bot_stats_display = gr.Textbox(label="Bot Performance", value="Bot Stats: No games played yet", interactive=False)
    
    # Game message
    message_display = gr.Textbox(label="Game Status", value="Click 'New Game' to start!", interactive=False)
    
    # Betting controls
    with gr.Row(visible=False) as bet_row:
        bet_amount = gr.Number(label="Bet Amount", value=10, minimum=1, maximum=100)
        place_bet_btn = gr.Button("Place Bet", variant="primary")
    
    # Playing controls  
    with gr.Row(visible=False) as play_row:
        hit_btn = gr.Button("Hit", variant="primary")
        stand_btn = gr.Button("Stand", variant="secondary")
    
    # Game management controls
    with gr.Row():
        new_game_btn = gr.Button("New Game", variant="primary")
        reset_banks_btn = gr.Button("Reset Banks", variant="secondary")
    
    # Event handlers
    new_game_btn.click(
        start_new_game,
        outputs=[dealer_cards, player_cards, bot_cards, player_bank_display, 
                bot_bank_display, bot_stats_display, message_display, 
                bet_row, play_row, new_game_btn]
    )
    
    place_bet_btn.click(
        place_bet,
        inputs=[bet_amount],
        outputs=[dealer_cards, player_cards, bot_cards, player_bank_display, 
                bot_bank_display, bot_stats_display, message_display, 
                bet_row, play_row, new_game_btn]
    )
    
    hit_btn.click(
        player_hit,
        outputs=[dealer_cards, player_cards, bot_cards, player_bank_display, 
                bot_bank_display, bot_stats_display, message_display, 
                bet_row, play_row, new_game_btn]
    )
    
    stand_btn.click(
        player_stand,
        outputs=[dealer_cards, player_cards, bot_cards, player_bank_display, 
                bot_bank_display, bot_stats_display, message_display, 
                bet_row, play_row, new_game_btn]
    )
    
    reset_banks_btn.click(
        reset_banks,
        outputs=[dealer_cards, player_cards, bot_cards, player_bank_display, 
                bot_bank_display, bot_stats_display, message_display, 
                bet_row, play_row, new_game_btn]
    )
    
    # Initialize display
    demo.load(
        start_new_game,
        outputs=[dealer_cards, player_cards, bot_cards, player_bank_display, 
                bot_bank_display, bot_stats_display, message_display, 
                bet_row, play_row, new_game_btn]
    )

if __name__ == "__main__":
    demo.launch()
