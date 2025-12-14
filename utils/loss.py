import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target, smooth=1e-5):
        pred = torch.sigmoid(pred)
        intersection = torch.sum(pred * target)
        return 1 - (2. * intersection + smooth) / (torch.sum(pred) + torch.sum(target) + smooth)

bce_loss = nn.BCEWithLogitsLoss()
dice_loss = DiceLoss()
cross_entropy_loss = nn.BCELoss()

def combined_gan_loss(pred, target, adversarial_loss_weight=0.1):
    return bce_loss(pred, target) + dice_loss(pred, target) #* (1 - adversarial_loss_weight)

def discriminator_loss(real_scores, fake_scores):
    return -(torch.mean(real_scores) - torch.mean(fake_scores))

def generator_loss(fake_scores):
    return -torch.mean(fake_scores)

# Modified Loss Function with Confidence Weighting
class DiceLoss4CWL(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss4CWL, self).__init__()
        self.smooth = smooth
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum()
        return 1 - (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)

class ConfidenceWeightedLoss_T(nn.Module):
    def __init__(self, threshold=0.5):
        super(ConfidenceWeightedLoss_T, self).__init__()
        self.ce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.dice_loss = DiceLoss4CWL()
        self.threshold = threshold  # Define the threshold value T

    def forward(self, pred, target, confidence_map):
        # Ensure confidence_map has the same spatial dimensions as pred and target
        if confidence_map.shape[2:] != pred.shape[2:]:
            confidence_map = F.interpolate(confidence_map, size=pred.shape[2:], mode="bilinear", align_corners=False)
        confidence_map = (confidence_map > self.threshold).float() # Binarize confidence map
        ce_loss = self.ce_loss(pred, target) # Compute per-pixel cross-entropy loss
        weighted_ce_loss = (confidence_map * ce_loss).mean() # Apply confidence weighting
        dice_loss = self.dice_loss(pred, target)
        return weighted_ce_loss + dice_loss  # Combined loss
