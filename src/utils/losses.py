import torch
import torch.nn as nn
import torch.nn.functional as F
from .config_loader import Config

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)  
        
        return 1 - dice

class BCEWithLogitsLossAndDiceLoss(nn.Module):
    def __init__(self, dice_weight=1, bce_weight=1):
        super(BCEWithLogitsLossAndDiceLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, inputs, targets):
        dice_loss = self.dice_loss(inputs, targets)
        bce_loss = self.bce_loss(inputs, targets)
        return self.dice_weight * dice_loss + self.bce_weight * bce_loss

def get_loss_function():
    loss_config = Config.get_instance().config['loss']
    if loss_config['name'] == 'DiceLoss':
        return DiceLoss(smooth=loss_config.get('smooth', 1.))
    elif loss_config['name'] == 'BCEWithLogitsLossAndDiceLoss':
        return BCEWithLogitsLossAndDiceLoss(dice_weight=loss_config.get('dice_weight', 1),
                                            bce_weight=loss_config.get('bce_weight', 1))
    elif loss_config['name'] == 'CrossEntropyLoss':
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unknown loss function: {loss_config['name']}")
