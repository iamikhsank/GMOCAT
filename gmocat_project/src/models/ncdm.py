import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralCDM(nn.Module): 
    """ 
    NCDM Framework (Wang 2023) with Monotonicity Constraints. 
    """ 
    def __init__(self, n_users, n_items, n_concepts, q_matrix, hidden_dim, device): 
        super(NeuralCDM, self).__init__() 
        self.q_matrix = q_matrix.to(device) 
        self.student_emb = nn.Embedding(n_users, n_concepts) 
        self.diff_emb = nn.Embedding(n_items, 1) 
        self.disc_emb = nn.Embedding(n_items, 1) 
         
        # Neural Interaction Layers 
        self.net = nn.Sequential( 
            nn.Linear(n_concepts, hidden_dim), 
            nn.Tanh(), 
            nn.Linear(hidden_dim, 1) 
        ) 
        # Initialize positive weights for monotonicity 
        for layer in self.net: 
            if isinstance(layer, nn.Linear): 
                nn.init.xavier_uniform_(layer.weight) 

    def forward(self, u_idx, q_idx): 
        # 1. Student Knowledge State 
        theta = torch.sigmoid(self.student_emb(u_idx)) # (0, 1) 
         
        # 2. Get relevant concepts for the question 
        q_mask = self.q_matrix[q_idx] 
         
        # 3. Knowledge Retrieval (Element-wise product) 
        input_x = theta * q_mask 
         
        # 4. Monotonic Neural Network 
        # Enforce positive weights during forward pass 
        x = input_x 
        for layer in self.net: 
            if isinstance(layer, nn.Linear): 
                x = F.linear(x, torch.abs(layer.weight), layer.bias) 
            else: 
                x = layer(x) 
                 
        # 5. IRT Integration 
        diff = torch.sigmoid(self.diff_emb(q_idx)) 
        disc = torch.sigmoid(self.disc_emb(q_idx)) 
         
        logits = (x - diff) * disc * 10 
        return torch.sigmoid(logits).view(-1)
