import tkinter as tk
from tkinter import ttk, messagebox, font
import torch
import torch.nn as nn
import torch.nn.functional as F # Need F for DQN forward
import numpy as np
import os
import random

# --- Constants ---
MODEL_FILENAME = "blackjack_dqn_policy_50m.pth"
N_OBSERVATIONS = 3 # player_sum, dealer_card, usable_ace
N_ACTIONS = 2      # Stand, Hit

# --- Card Data (unchanged) ---
RANKS = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
VALUES = {'A': 11, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, 'J': 10, 'Q': 10, 'K': 10}
SUITS = {'Spades': '♠', 'Hearts': '♥', 'Diamonds': '♦', 'Clubs': '♣'}

# --- Card Class (unchanged) ---
class Card:
    def __init__(self, rank):
        if rank not in RANKS:
            raise ValueError(f"Invalid rank: {rank}")
        self.rank = rank
        self.value = VALUES[rank]
        self.suit = SUITS['Spades'] # Defaulting for compatibility
        try:
            base = 0x1F0A0
            offset = RANKS.index(rank) + 1
            if rank == 'J': offset = 11
            if rank == 'Q': offset = 13
            if rank == 'K': offset = 14
            if offset >= 12: offset += 1 
            self.unicode_char = chr(base + offset)
        except IndexError:
            self.unicode_char = rank + self.suit
    def __str__(self):
        return self.unicode_char

# --- Hand Calculation (unchanged) ---
def calculate_hand_value(hand):
    num_aces = sum(1 for card in hand if card.rank == 'A')
    total_value = sum(card.value for card in hand)
    while total_value > 21 and num_aces > 0:
        total_value -= 10
        num_aces -= 1
    temp_value_no_ace_as_11 = sum(VALUES[card.rank] if card.rank != 'A' else 1 for card in hand)
    num_aces_original = sum(1 for card in hand if card.rank == 'A')
    usable_ace = False
    if num_aces_original > 0:
        if temp_value_no_ace_as_11 + 11 + (num_aces_original - 1) * 1 <= 21:
            usable_ace = True
    return total_value, usable_ace

# --- Define the DQN Network (Needs to match the one used for training) ---
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
             # Convert state tuple/list to tensor for inference
             x = torch.tensor(x, dtype=torch.float32)
        # Ensure correct device placement if using GPU during training
        # For inference, CPU is usually fine unless batching predictions
        if x.dim() == 1: # Add batch dimension if single state
             x = x.unsqueeze(0)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

# --- Main GUI Class ---
class BlackjackAdvisorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Blackjack DQN Advisor")
        self.root.geometry("650x500")
        self.root.resizable(True, True)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Advisor using device: {self.device}")
        self.policy_net = self.load_model()
        
        self.player_hand = []
        self.dealer_card = None

        # --- Styles and Layout (mostly unchanged) ---
        self.style = ttk.Style()
        self.style.configure("TLabel", font=("Arial", 12))
        self.style.configure("TButton", font=("Arial", 10), padding=5)
        self.style.configure("Card.TLabel", font=("Arial", 36), padding=5)
        self.style.configure("Result.TLabel", font=("Arial", 24, "bold"))
        self.style.configure("Header.TLabel", font=("Arial", 14, "bold"))
        
        # --- Layout Frames --- 
        self.dealer_frame = ttk.Frame(root, padding="10")
        self.dealer_frame.grid(row=0, column=0, columnspan=2, sticky="ew")
        self.player_frame = ttk.Frame(root, padding="10")
        self.player_frame.grid(row=1, column=0, columnspan=2, sticky="ew")
        self.controls_frame = ttk.Frame(root, padding="10")
        self.controls_frame.grid(row=2, column=0, sticky="nsew")
        self.actions_frame = ttk.Frame(root, padding="10")
        self.actions_frame.grid(row=2, column=1, sticky="nsew")
        self.result_display_frame = ttk.Frame(root, padding="10")
        self.result_display_frame.grid(row=3, column=0, columnspan=2, sticky="ew")
        self.status_frame = ttk.Frame(root, relief=tk.SUNKEN)
        self.status_frame.grid(row=4, column=0, columnspan=2, sticky="ew")
        
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(1, weight=1)

        # --- Dealer Section --- 
        ttk.Label(self.dealer_frame, text="Dealer's Card:", style="Header.TLabel").pack(side=tk.LEFT, padx=5)
        self.dealer_card_label = ttk.Label(self.dealer_frame, text="?", style="Card.TLabel", relief=tk.GROOVE, anchor=tk.CENTER, width=3)
        self.dealer_card_label.pack(side=tk.LEFT, padx=5)

        # --- Player Section --- 
        ttk.Label(self.player_frame, text="Player's Hand:", style="Header.TLabel").pack(anchor=tk.W)
        self.player_hand_frame = ttk.Frame(self.player_frame)
        self.player_hand_frame.pack(fill=tk.X, expand=True, pady=5)
        self.player_value_label = ttk.Label(self.player_frame, text="Value: 0")
        self.player_value_label.pack(anchor=tk.W)
        self.update_player_display()

        # --- Controls Section (unchanged) --- 
        ttk.Label(self.controls_frame, text="Add Card:", style="Header.TLabel").grid(row=0, column=0, columnspan=4, sticky="w")
        row, col = 1, 0
        for rank in RANKS:
            btn = ttk.Button(self.controls_frame, text=rank, 
                             command=lambda r=rank: self.add_player_card(r), width=3)
            btn.grid(row=row, column=col, padx=2, pady=2, sticky="ew")
            col += 1; row = row + 1 if col > 3 else row; col = 0 if col > 3 else col
                
        ttk.Label(self.controls_frame, text="Dealer Shows:", style="Header.TLabel").grid(row=row+1, column=0, columnspan=4, sticky="w", pady=(10,0))
        row, col = row+2, 0
        for rank in RANKS:
            btn = ttk.Button(self.controls_frame, text=rank, 
                             command=lambda r=rank: self.set_dealer_card(r), width=3)
            btn.grid(row=row, column=col, padx=2, pady=2, sticky="ew")
            col += 1; row = row + 1 if col > 3 else row; col = 0 if col > 3 else col
        for i in range(4): self.controls_frame.columnconfigure(i, weight=1)

        # --- Actions Section (unchanged) --- 
        self.btn_get_advice = ttk.Button(self.actions_frame, text="Get Advice", command=self.get_advice, style="TButton")
        self.btn_get_advice.pack(pady=10, fill=tk.X)
        self.btn_clear = ttk.Button(self.actions_frame, text="Clear Hand", command=self.clear_hand, style="TButton")
        self.btn_clear.pack(pady=10, fill=tk.X)
        self.actions_frame.rowconfigure(0, weight=1)
        self.actions_frame.rowconfigure(1, weight=1)

        # --- Result Display (unchanged) --- 
        self.result_label = ttk.Label(self.result_display_frame, text="Select cards and click 'Get Advice'", anchor=tk.CENTER)
        self.result_label.pack(pady=5)
        self.recommendation_label = ttk.Label(self.result_display_frame, text="", style="Result.TLabel", anchor=tk.CENTER)
        self.recommendation_label.pack(pady=10)
        self.result_display_frame.columnconfigure(0, weight=1)

        # --- Status Bar --- 
        self.status_var = tk.StringVar()
        self.status_bar = ttk.Label(self.status_frame, textvariable=self.status_var, anchor=tk.W)
        self.status_bar.pack(fill=tk.X, padx=5, pady=2)
        self.update_status("DQN model loaded." if self.policy_net else f"MODEL FILE ({MODEL_FILENAME}) NOT FOUND.")

        self.root.bind("<Return>", lambda event: self.get_advice()) # Bind Enter key
        
    def update_status(self, message):
        self.status_var.set(message)

    def load_model(self):
        """Load the trained DQN model state dictionary."""
        if not os.path.exists(MODEL_FILENAME):
            messagebox.showerror("Error", f"Model file not found: {MODEL_FILENAME}\nPlease run train_dqn_policy.py first.")
            return None
        try:
            model = DQN(N_OBSERVATIONS, N_ACTIONS).to(self.device)
            model.load_state_dict(torch.load(MODEL_FILENAME, map_location=self.device)) # map_location handles CPU/GPU difference
            model.eval() # Set model to evaluation mode
            print(f"DQN model loaded successfully from {MODEL_FILENAME}")
            return model
        except Exception as e:
            messagebox.showerror("Error Loading Model", f"Failed to load model: {e}")
            return None

    # --- add_player_card, set_dealer_card, clear_hand, update_player_display (unchanged from previous version) --- 
    def add_player_card(self, rank):
        if len(self.player_hand) >= 8:
             self.update_status("Max player cards reached.")
             return
        try:
            card = Card(rank)
            self.player_hand.append(card)
            self.update_player_display()
        except ValueError as e:
             messagebox.showerror("Error", str(e))
             
    def set_dealer_card(self, rank):
        try:
            self.dealer_card = Card(rank)
            self.dealer_card_label.config(text=str(self.dealer_card))
            self.update_status(f"Dealer card set to {rank}.")
        except ValueError as e:
             messagebox.showerror("Error", str(e))
             
    def clear_hand(self):
        self.player_hand = []
        self.dealer_card = None
        self.update_player_display()
        self.dealer_card_label.config(text="?")
        self.result_label.config(text="Select cards and click 'Get Advice'")
        self.recommendation_label.config(text="", foreground="black")
        self.update_status("Hand cleared.")

    def update_player_display(self):
        for widget in self.player_hand_frame.winfo_children():
            widget.destroy()
        if not self.player_hand:
            ttk.Label(self.player_hand_frame, text="No cards yet", style="TLabel").pack()
        else:
            for card in self.player_hand:
                lbl = ttk.Label(self.player_hand_frame, text=str(card), style="Card.TLabel", relief=tk.GROOVE, anchor=tk.CENTER, width=3)
                lbl.pack(side=tk.LEFT, padx=3)
        value, usable_ace = calculate_hand_value(self.player_hand)
        ace_text = " (Usable Ace)" if usable_ace else ""
        bust_text = " BUST!" if value > 21 else ""
        self.player_value_label.config(text=f"Value: {value}{ace_text}{bust_text}")
    # --- End of unchanged methods --- 

    def get_advice(self):
        if not self.policy_net:
            messagebox.showerror("Error", "DQN model not loaded.")
            return
        if not self.dealer_card:
            messagebox.showwarning("Input Needed", "Please select the dealer's card.")
            return
        if not self.player_hand:
             messagebox.showwarning("Input Needed", "Please add cards to the player's hand.")
             return

        player_sum, usable_ace = calculate_hand_value(self.player_hand)
        dealer_val = VALUES[self.dealer_card.rank]
        if self.dealer_card.rank == 'A':
            dealer_val = 1 # Gymnasium state uses 1 for Ace
            
        if player_sum > 21:
            self.result_label.config(text=f"Hand: {player_sum} - BUSTED")
            self.recommendation_label.config(text="STAND", foreground="red")
            self.update_status("Player busted.")
            return
            
        # Create state representation for the DQN model
        # Typically: (player_sum, dealer_showing_card_value, usable_ace_flag)
        state_features = [player_sum, dealer_val, 1.0 if usable_ace else 0.0]
        state_tensor = torch.tensor(state_features, dtype=torch.float32, device=self.device)

        try:
            with torch.no_grad(): # Inference doesn't need gradient calculation
                q_values = self.policy_net(state_tensor) # Get Q-values from the network
                action = torch.argmax(q_values).item() # Choose action with highest Q-value
            
            recommendation = "HIT" if action == 1 else "STAND"
            color = "green" if action == 1 else "red"
            
            self.result_label.config(text=f"Hand: {player_sum}, Dealer: {self.dealer_card.rank}, Usable Ace: {usable_ace}")
            self.recommendation_label.config(text=recommendation, foreground=color)
            self.update_status(f"DQN Advice for state {tuple(state_features)}: {recommendation}")

        except Exception as e:
            messagebox.showerror("Prediction Error", f"Could not get prediction from model: {e}")
            self.update_status("Error during prediction.")

if __name__ == "__main__":
    root = tk.Tk()
    # Font configuration (unchanged)
    try:
        default_font = font.nametofont("TkDefaultFont")
        # Try common fonts supporting card symbols
        prefered_fonts = ["Segoe UI Symbol", "Arial Unicode MS", "DejaVu Sans"]
        available_families = list(font.families())
        chosen_family = default_font["family"]
        for pf in prefered_fonts:
            if pf in available_families:
                chosen_family = pf
                break
        default_font.configure(family=chosen_family, size=10)
        root.option_add("*Font", default_font)
        print(f"Using font: {chosen_family}")
    except Exception as e:
        print(f"Font setting error (continuing with default): {e}")
    
    app = BlackjackAdvisorGUI(root)
    root.mainloop() 