import torch
import torch.nn as nn
import torch.nn.functional as F

class PerceptronRouter(nn.Module):
    def __init__(self, input_size, n_experts):
        super().__init__()
        self.fc = nn.Linear(input_size, n_experts)
        
    def forward(self, x):
        # x: (tensor): embedding
        return torch.argmax(self.fc(x), dim=1, keepdim=True) # softmax prediction

