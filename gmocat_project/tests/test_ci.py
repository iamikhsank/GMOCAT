import os
import shutil
import tempfile
import pandas as pd
import pytest
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import Config
from src.data_loader import DataProcessor
from src.train import train_and_evaluate

@pytest.fixture
def temp_assistments_env():
    # Create temp dir structure
    temp_dir = tempfile.mkdtemp()
    data_dir = os.path.join(temp_dir, 'data')
    assist_dir = os.path.join(data_dir, 'assistments')
    os.makedirs(assist_dir)
    
    # Create dummy assistments csv (minimal valid structure)
    csv_path = os.path.join(assist_dir, 'assistments_2009.csv')
    df = pd.DataFrame({
        'user_id': [1, 2, 1, 3, 2],
        'problem_id': [101, 102, 103, 101, 104],
        'list_skill_ids': ['1;2', '2', '3', '1', '2;3'],
        'correct': [1, 0, 1, 1, 0]
    })
    df.to_csv(csv_path, index=False)
    
    # Yield config pointing to this temp dir
    config = Config()
    config.BASE_DIR = temp_dir
    config.DATA_DIR = data_dir
    config.LOG_DIR = os.path.join(temp_dir, 'logs')
    config.RESULT_DIR = os.path.join(temp_dir, 'results')
    config.dataset_name = 'assistments'
    config.device = 'cpu' # Force CPU for CI
    
    # Speed up training
    config.batch_size = 2
    config.ncdm_epochs = 1
    config.n_rl_episodes = 2
    config.embed_dim = 4
    config.gnn_hidden = 4
    config.ncdm_hidden = 8
    
    yield config
    
    # Cleanup
    shutil.rmtree(temp_dir)

def test_assistments_pipeline(temp_assistments_env):
    """
    Test the full pipeline (Data -> GNN -> NCDM -> RL -> Eval) 
    using a minimal dataset to ensure no runtime errors.
    """
    config = temp_assistments_env
    loader = DataProcessor(config)
    
    # Run pipeline
    metrics = train_and_evaluate(config, loader)
    
    assert metrics is not None
    assert 'auc' in metrics
    assert metrics['auc'] >= 0
    assert metrics['coverage'] >= 0
