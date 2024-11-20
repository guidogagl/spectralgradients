import torch
import torch.nn as nn

# localization approach proposed by
# Kohlbrenner, Maattrimilian, et al. "Towards best practice in eattrplaining neural network decisions with LRP." 
# 2020 International Joint Conference on Neural Networks (IJCNN). IEEE, 2020.
# mu = Rin / Rout
# mu_w =  mu * (Stot / Sin )
class Localization(nn.Module):
    def __init__(self):
        super(Localization, self).__init__()

    def forward(self, x, attr, mask):
        attr = nn.functional.relu(attr)
        Rtot = attr.reshape(attr.shape[0], -1).sum(1)
        
        if any(Rtot == 0):
            raise ValueError("Rtot is zero")
        
        Rin = (attr * mask).reshape(attr.shape[0], -1).sum(1)
        mu = Rin / Rtot

        Stot = attr.shape[1:]
        Stot = torch.tensor(Stot).prod().item()
    
        Sin = mask.reshape(mask.shape[0], -1).sum(1)

        if any(Sin == 0):
            raise ValueError("Sin is zero")
        
        return mu * (Stot / Sin)
