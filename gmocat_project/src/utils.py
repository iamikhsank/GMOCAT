import logging
import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, accuracy_score

def setup_logging(log_dir, experiment_name):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{experiment_name}.log")
    
    # Reset logger to avoid duplicates
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logging initialized. Saving to {log_file}")

def set_seed(seed):
    """Ensure Reproducibility across runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    logging.info(f"[SYSTEM] Random Seed Locked: {seed}")

def calculate_metrics(y_true, y_prob):
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = 0.5 # Handle case with only one class
        
    pred = [1 if p > 0.5 else 0 for p in y_prob]
    acc = accuracy_score(y_true, pred)
    
    # ECE Calculation
    bins = np.linspace(0, 1, 11)
    ece = 0.0
    y_prob = np.array(y_prob)
    y_true = np.array(y_true)
    for i in range(10):
        mask = (y_prob > bins[i]) & (y_prob <= bins[i+1])
        if np.sum(mask) > 0:
            ece += np.abs(np.mean(y_true[mask]) - np.mean(y_prob[mask])) * np.mean(mask)
    return auc, acc, ece

def eval_adaptive_quality(counts, n_items):
    coverage = np.count_nonzero(counts) / n_items
    # Exposure rate: items shown more than average
    threshold = np.mean(counts) if np.mean(counts) > 0 else 1
    high_exp = np.sum(counts > threshold) / n_items
    return coverage, high_exp
