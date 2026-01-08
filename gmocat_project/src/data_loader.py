import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import logging

class CATDataset(Dataset):
    def __init__(self, data_array):
        self.data = data_array
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.user_map = {}
        self.item_map = {}
        self.concept_map = {}

    def _get_id(self, val, mapper):
        if val not in mapper:
            mapper[val] = len(mapper)
        return mapper[val]

    def load_data(self):
        if self.config.dataset_name == 'assistments':
            return self._load_assistments()
        elif self.config.dataset_name == 'dbekt22':
            return self._load_dbekt22()
        else:
            raise ValueError(f"Unknown dataset: {self.config.dataset_name}")

    def _load_assistments(self):
        path = os.path.join(self.config.DATA_DIR, 'assistments', 'assistments_2009.csv')
        logging.info(f"Loading ASSISTments from {path}")
        
        try:
            df = pd.read_csv(path, encoding='latin-1', low_memory=False)
        except Exception as e:
            logging.error(f"Failed to load ASSISTments: {e}")
            return None, None

        # Columns: user_id, problem_id, list_skill_ids, correct
        # Filter rows with skills
        df = df.dropna(subset=['user_id', 'problem_id', 'correct', 'list_skill_ids'])
        
        # Standardize
        data = []
        q_matrix_entries = []
        
        # We need to process potentially multiple skills per item.
        # But for NeuralCDM, usually we treat (user, item) -> correct.
        # The Q-matrix handles the item->concepts mapping.
        
        # First, build mappings
        for _, row in df.iterrows():
            u_raw = row['user_id']
            i_raw = row['problem_id']
            skills_raw = str(row['list_skill_ids'])
            correct = int(row['correct'])
            
            u_idx = self._get_id(u_raw, self.user_map)
            i_idx = self._get_id(i_raw, self.item_map)
            
            # Interactions
            data.append([u_idx, i_idx, correct])
            
            # Q-Matrix entries
            # skills can be "1;2"
            skills = skills_raw.split(';')
            for s in skills:
                try:
                    s_idx = self._get_id(s.strip(), self.concept_map)
                    q_matrix_entries.append((i_idx, s_idx))
                except:
                    continue
                    
        return self._finalize(data, q_matrix_entries)

    def _load_dbekt22(self):
        raw_dir = os.path.join(self.config.DATA_DIR, 'dbekt22', 'raw')
        logging.info(f"Loading DBE-KT22 from {raw_dir}")
        
        try:
            trans_path = os.path.join(raw_dir, 'Transaction.csv')
            rels_path = os.path.join(raw_dir, 'Question_KC_Relationships.csv')
            
            trans_df = pd.read_csv(trans_path)
            rels_df = pd.read_csv(rels_path)
        except Exception as e:
            logging.error(f"Failed to load DBE-KT22 files: {e}")
            return None, None
            
        # Process Transactions
        # StudentId, QuestionId, IsCorrect
        # Real DBE-KT22 headers: student_id, question_id, answer_state
        data = []
        for _, row in trans_df.iterrows():
            # Adjust column names based on actual file inspection
            # Fallback to known variants if needed
            uid = row.get('student_id', row.get('StudentId'))
            qid = row.get('question_id', row.get('QuestionId'))
            # answer_state is 'true'/'false' or boolean
            is_correct_raw = row.get('answer_state', row.get('IsCorrect'))
            
            if uid is None or qid is None: continue
            
            if isinstance(is_correct_raw, str):
                correct = 1 if is_correct_raw.lower() == 'true' else 0
            else:
                correct = int(is_correct_raw)

            u_idx = self._get_id(uid, self.user_map)
            i_idx = self._get_id(qid, self.item_map)
            data.append([u_idx, i_idx, correct])
            
        # Process Q-Matrix
        # QuestionId, KCId
        # Real DBE-KT22 headers: question_id, knowledgecomponent_id
        q_matrix_entries = []
        for _, row in rels_df.iterrows():
            qid = row.get('question_id', row.get('QuestionId'))
            kcid = row.get('knowledgecomponent_id', row.get('KCId'))
            
            if qid in self.item_map: # Only include items present in interactions
                i_idx = self.item_map[qid]
                c_idx = self._get_id(kcid, self.concept_map)
                q_matrix_entries.append((i_idx, c_idx))
                
        return self._finalize(data, q_matrix_entries)

    def _finalize(self, data, q_matrix_entries):
        # Update Config
        self.config.n_users = len(self.user_map)
        self.config.n_questions = len(self.item_map)
        self.config.n_concepts = len(self.concept_map)
        
        logging.info(f"Stats: Users={self.config.n_users}, Items={self.config.n_questions}, Concepts={self.config.n_concepts}")
        
        # Build Q-Matrix Tensor
        q_matrix = torch.zeros(self.config.n_questions, self.config.n_concepts)
        for i, c in q_matrix_entries:
            q_matrix[int(i), int(c)] = 1.0
            
        # Create Dataset
        df = pd.DataFrame(data, columns=['user_id', 'item_id', 'correct'])
        
        return df, q_matrix
