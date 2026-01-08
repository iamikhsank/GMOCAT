import os
import csv
import logging
from src.config import Config
from src.utils import setup_logging, set_seed
from src.data_loader import DataProcessor
from src.train import train_and_evaluate

def main():
    datasets = ['assistments', 'dbekt22']
    seeds = [42, 1024, 2024]
    
    config = Config()
    
    # Prepare Result File
    os.makedirs(config.RESULT_DIR, exist_ok=True)
    summary_file = os.path.join(config.RESULT_DIR, 'experiment_summary.csv')
    
    with open(summary_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['dataset', 'seed', 'auc', 'acc', 'ece', 'coverage', 'high_exposure_rate', 'temperature'])
        writer.writeheader()

    for d_name in datasets:
        for seed in seeds:
            exp_name = f"{d_name}_seed{seed}"
            setup_logging(config.LOG_DIR, exp_name)
            logging.info(f"Starting Experiment: {exp_name}")
            
            # Update Config
            config.update(dataset_name=d_name, seed=seed)
            set_seed(seed)
            
            # Prepare Data
            loader = DataProcessor(config)
            
            # Run Training
            try:
                metrics = train_and_evaluate(config, loader)
                if metrics:
                    with open(summary_file, 'a', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=['dataset', 'seed', 'auc', 'acc', 'ece', 'coverage', 'high_exposure_rate', 'temperature'])
                        writer.writerow(metrics)
                    logging.info(f"Finished Experiment: {exp_name}")
                else:
                    logging.error(f"Experiment failed (No Data): {exp_name}")
            except Exception as e:
                logging.error(f"Experiment crashed: {exp_name} | Error: {e}", exc_info=True)

    print(f"All experiments completed. Results saved to {summary_file}")

if __name__ == "__main__":
    main()
