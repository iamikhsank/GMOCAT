import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from .models.gnn import build_graph, GraphAggregator, HAS_PYG
from .models.ncdm import NeuralCDM
from .models.calibration import CalibratedModel
from .models.rl import CATEnvironment, ActorCritic
from .utils import calculate_metrics, eval_adaptive_quality
import torch.nn.functional as F
import numpy as np
import random

def train_and_evaluate(config, dataset_loader):
    # --- PHASE 1: DATA LOADING ---
    df, q_matrix = dataset_loader.load_data()
    if df is None: return None
    
    q_matrix_tensor = q_matrix.to(config.device)
    # Using simple wrapper from data_loader
    from .data_loader import CATDataset
    dataset = CATDataset(torch.tensor(df.values, dtype=torch.long))
    train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    # --- PHASE 2: GNN & NCDM PRE-TRAINING ---
    logging.info("[PHASE 2] Training NCDM (Cognitive Model)...")
    
    gnn = None
    if HAS_PYG: 
        edge_index, n_nodes = build_graph(q_matrix, config.device) 
        gnn = GraphAggregator(n_nodes, config.embed_dim, config.device).to(config.device) 
        logging.info("GNN Initialized.") 
    
    ncdm = NeuralCDM(config.n_users, config.n_questions, config.n_concepts, q_matrix_tensor, config.ncdm_hidden, config.device).to(config.device) 
    opt = optim.Adam(ncdm.parameters(), lr=config.lr_ncdm) 
    loss_fn = nn.BCELoss() 
    
    for ep in range(config.ncdm_epochs): 
        ncdm.train() 
        total_loss = 0 
        for batch in train_loader: 
            u, q, r = batch[:,0], batch[:,1], batch[:,2] 
            u, q, r = u.to(config.device), q.to(config.device), r.float().to(config.device) 
             
            opt.zero_grad() 
            pred = ncdm(u, q) 
            loss = loss_fn(pred, r) 
            loss.backward() 
            opt.step() 
            total_loss += loss.item() 
        logging.info(f"Epoch {ep+1}: Loss {total_loss/len(train_loader):.4f}") 

    # --- PHASE 3: CALIBRATION ---
    logging.info("[PHASE 3] Temperature Scaling...") 
    calibrated_model = CalibratedModel(ncdm).to(config.device) 
    logging.info(f"Calibrated Temperature: {calibrated_model.temperature.item():.2f}") 
    
    # --- PHASE 4: RL AGENT TRAINING (GMOCAT) ---
    logging.info("[PHASE 4] Training Policy (Simulated)...")
    env = CATEnvironment(calibrated_model, q_matrix_tensor, config) 
    agent = ActorCritic(config.embed_dim, config.n_questions).to(config.device) 
    
    # Simulation Loop
    for ep in range(config.n_rl_episodes): 
        uid = np.random.randint(0, config.n_users) 
        state = env.reset(uid) 
        done = False 
        ep_rew = 0 
         
        # Get Item Embeddings from GNN if available 
        if gnn:  
            node_embs = gnn(edge_index) 
            item_embs = node_embs[:config.n_questions] 
        else: 
            item_embs = torch.randn(config.n_questions, config.embed_dim).to(config.device) 
             
        while not done: 
            # Action Selection 
            logits, val = agent(state.squeeze(0))
            
            # Logits shape: [batch_size, n_questions] or [n_questions]
            if logits.ndim == 1:
                logits = logits.unsqueeze(0) # [1, n_questions]
             
            # Masking exposed items 
            mask = torch.full_like(logits, -1e9) 
            valid_items = [i for i in range(config.n_questions) if i not in env.exposed] 
            
            if not valid_items: # Safety break if all items exhausted
                break
            
            # Mask is [1, N] or [N] depending on logits.
            # If mask is [1, N], mask[0, idxs] works if idxs are valid.
            # If logits came out [1, 1, N] somehow, that would be an issue.
            
            if mask.ndim == 3: # [1, 1, N] - Possible if state was [1, 1, D]
                 mask = mask.squeeze(0) # [1, N]
            
            # Safe masking logic
            if mask.ndim == 2:
                # Iterate or use advanced indexing safely
                # Ensure we are not indexing out of bounds
                valid_items_tensor = torch.tensor(valid_items, dtype=torch.long, device=config.device)
                mask[0].index_fill_(0, valid_items_tensor, 0)
            elif mask.ndim == 1:
                valid_items_tensor = torch.tensor(valid_items, dtype=torch.long, device=config.device)
                mask.index_fill_(0, valid_items_tensor, 0)
             
            probs = F.softmax(logits + mask, dim=-1) 
            action = torch.distributions.Categorical(probs).sample() 
             
            next_state, reward, done = env.step(action, item_embs[action]) 
            ep_rew += reward 
            state = next_state 
        
        if (ep+1) % 100 == 0:
            logging.info(f"RL Episode {ep+1} | Reward: {ep_rew:.4f} | Steps: {env.step_count}")

    # --- PHASE 5: EVALUATION ---
    logging.info("[PHASE 5] Final Evaluation...") 
    y_true, y_prob = [], [] 
    with torch.no_grad(): 
        for batch in train_loader: 
            u, q, r = batch[:,0], batch[:,1], batch[:,2] 
            u, q = u.to(config.device), q.to(config.device) 
            pred = calibrated_model(u, q) 
            y_true.extend(r.cpu().numpy()) 
            y_prob.extend(pred.cpu().numpy()) 
            if len(y_true) > 5000: break 
             
    auc, acc, ece = calculate_metrics(y_true, y_prob) 
    cov, exp_rate = eval_adaptive_quality(env.exposure_counts, config.n_questions) 
    
    results = {
        'dataset': config.dataset_name,
        'seed': config.seed,
        'auc': auc,
        'acc': acc,
        'ece': ece,
        'coverage': cov,
        'high_exposure_rate': exp_rate,
        'temperature': calibrated_model.temperature.item()
    }
    
    logging.info(f"Results: {results}")
    return results
