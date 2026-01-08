import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

class ActorCritic(nn.Module): 
    """Policy Network and Value Network.""" 
    def __init__(self, state_dim, action_dim): 
        super(ActorCritic, self).__init__() 
        self.base = nn.Sequential(nn.Linear(state_dim, 64), nn.Tanh()) 
        self.actor = nn.Linear(64, action_dim) 
        self.critic = nn.Linear(64, 1) 
         
    def forward(self, state): 
        x = self.base(state) 
        return self.actor(x), self.critic(x) 

class CATEnvironment: 
    """ 
    Simulates Adaptive Testing Environment. 
    Computes Multi-Objective Rewards. 
    """ 
    def __init__(self, ncdm, q_matrix, config): 
        self.ncdm = ncdm 
        self.q_matrix = q_matrix 
        self.config = config
        self.exposure_counts = np.zeros(config.n_questions) 
         
    def reset(self, user_id): 
        self.user_id = user_id 
        self.step_count = 0 
        self.exposed = set() 
        # Initial state (zero vector) 
        return torch.zeros(1, 1, self.config.embed_dim).to(self.config.device) 
         
    def step(self, action, item_emb): 
        q_idx = action.item() 
        self.step_count += 1 
        self.exposed.add(q_idx) 
        self.exposure_counts[q_idx] += 1 
         
        # 1. Simulate Response 
        with torch.no_grad(): 
            prob = self.ncdm(torch.tensor([self.user_id]).to(self.config.device),  
                             torch.tensor([q_idx]).to(self.config.device)).item() 
        correct = 1 if random.random() < prob else 0 
         
        # 2. Compute Rewards 
        # R1: Quality (Uncertainty reduction) - closest to 0.5 is best 
        r_qual = 1.0 - (abs(prob - 0.5) * 2) 
         
        # R2: Diversity (Concept novelty) - Simplified 
        r_div = 1.0 # (Ideally check if concept was tested before) 
         
        # R3: Novelty (Exposure penalty) 
        r_nov = 1.0 / np.log(self.exposure_counts[q_idx] + 2) 
         
        reward = (self.config.w_quality * r_qual) + (self.config.w_diversity * r_div) + (self.config.w_novelty * r_nov) 
         
        done = self.step_count >= self.config.max_step 
        next_state = item_emb.unsqueeze(0).unsqueeze(0) # Dummy next state update
         
        return next_state, reward, done 
