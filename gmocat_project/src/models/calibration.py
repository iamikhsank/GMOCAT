import torch
import torch.nn as nn

class CalibratedModel(nn.Module): 
    """ 
    Wraps NCDM to optimize probability calibration (ECE). 
    """ 
    def __init__(self, model): 
        super(CalibratedModel, self).__init__() 
        self.model = model 
        self.temperature = nn.Parameter(torch.ones(1) * 1.5) 
 
    def forward(self, u, q): 
        probs = self.model(u, q) 
        logits = torch.logit(probs, eps=1e-6) 
        return torch.sigmoid(logits / self.temperature) 
