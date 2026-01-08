import os
import torch

class Config:
    def __init__(self):
        # --- PATHS ---
        self.BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.DATA_DIR = os.path.join(self.BASE_DIR, 'data')
        self.LOG_DIR = os.path.join(self.BASE_DIR, 'logs')
        self.RESULT_DIR = os.path.join(self.BASE_DIR, 'results')
        
        # --- EXPERIMENT CONTROL ---
        self.seed = 42
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset_name = 'assistments' # 'assistments' or 'dbekt22'
        
        # --- MODEL HYPERPARAMS ---
        self.embed_dim = 64
        self.gnn_hidden = 64
        self.ncdm_hidden = 128
        self.rnn_hidden = 64
        self.dropout = 0.2
        
        # --- TRAINING ---
        self.batch_size = 256
        self.ncdm_epochs = 5
        self.lr_ncdm = 0.001
        self.lr_rl = 0.0005
        
        # --- RL / CAT SETTINGS ---
        self.max_step = 20        # Max items per student session
        self.n_rl_episodes = 500  # Number of simulated students for RL training
        
        # --- MULTI-OBJECTIVE WEIGHTS (Wang 2023) ---
        self.w_quality = 1.0      # Information Gain
        self.w_diversity = 0.5    # Concept Coverage
        self.w_novelty = 0.2      # Exposure Control

        # Placeholders (Auto-filled by Data Loader)
        self.n_users = 0
        self.n_questions = 0
        self.n_concepts = 0

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
