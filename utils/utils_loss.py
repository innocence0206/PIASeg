import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable

class SoftCEDiceLoss(nn.Module):
    def __init__(self, weight_ce=1.0, weight_dice=1.0, alpha=0.5, beta=0.5, smooth=1e-4,):
        super(SoftCEDiceLoss, self).__init__()
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        
    def forward(self, preds, labels):
        preds = torch.sigmoid(preds)
        logits = torch.cat([1.0 - preds, preds], dim=1)
        logits = torch.clamp(logits, 1e-4, 1.0 - 1e-4)

        loss_CE = (-labels * logits.log()).sum(dim=1).mean()
        
        log_seq = logits.view(logits.shape[0], 2, -1)
        lab_seq = labels.view(labels.shape[0], 2, -1)

        log_card = torch.norm(log_seq, p=1, dim=2)
        lab_card = torch.norm(lab_seq, p=1, dim=2)
        diff_card = torch.norm(log_seq - lab_seq, p=1, dim=2)

        tp = (log_card + lab_card - diff_card) / 2
        fp = log_card - tp
        fn = lab_card - tp

        tp = torch.sum(tp, dim=0)
        fp = torch.sum(fp, dim=0)
        fn = torch.sum(fn, dim=0)
        tversky = tp / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        loss_Dice = torch.mean(1.0 - tversky)
        
        loss = self.weight_ce * loss_CE + self.weight_dice * loss_Dice

        return loss
    
class SoftCEDiceLossv2(nn.Module):
    def __init__(self, weight_ce=1.0, weight_dice=1.0, alpha=0.5, beta=0.5, smooth=1e-4,):
        super(SoftCEDiceLossv2, self).__init__()
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        
    def forward(self, preds, labels):
        logits = torch.softmax(preds, dim=1)
        logits = torch.clamp(logits, 1e-4, 1.0 - 1e-4)

        loss_CE = (-labels * logits.log()).sum(dim=1).mean()
        
        log_seq = logits.view(logits.shape[0], 2, -1)
        lab_seq = labels.view(labels.shape[0], 2, -1)

        log_card = torch.norm(log_seq, p=1, dim=2)
        lab_card = torch.norm(lab_seq, p=1, dim=2)
        diff_card = torch.norm(log_seq - lab_seq, p=1, dim=2)

        tp = (log_card + lab_card - diff_card) / 2
        fp = log_card - tp
        fn = lab_card - tp

        tp = torch.sum(tp, dim=0)
        fp = torch.sum(fp, dim=0)
        fn = torch.sum(fn, dim=0)
        tversky = tp / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        loss_Dice = torch.mean(1.0 - tversky)
        
        loss = self.weight_ce * loss_CE + self.weight_dice * loss_Dice

        return loss

class PriorContloss(nn.Module):
    def __init__(self, temperature=0.07):
        super(PriorContloss, self).__init__()
        self.temperature = temperature  # Temperature for scaling logits
        
    def forward(self, x, x_pos, x_negs):
        # Normalize tensors to calculate cosine similarity
        x = F.normalize(x, dim=-1)
        x_pos = F.normalize(x_pos, dim=-1)
        x_negs = F.normalize(x_negs, dim=-1)

        # Compute cosine similarities
        pos_sim = torch.matmul(x, x_pos.T)  # (1 x 1)
        neg_sim = torch.matmul(x, x_negs.T)  # (1 x Pnum)

        # Concatenate positive and negative similarities
        logits = torch.cat([pos_sim, neg_sim], dim=1)  # (1 x Pnum+1)
        logits /= self.temperature  # Scale logits by temperature

        # Create labels: positive is at index 0
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=x.device)

        # Compute contrastive loss (Cross-Entropy over softmax probabilities)
        con_loss = F.cross_entropy(logits, labels)
        
        return con_loss

class CEDiceLoss(nn.Module):
    def __init__(self, weight_ce=1.0, weight_dice=1.0, alpha=0.5, beta=0.5, smooth=1e-4,):
        super(CEDiceLoss, self).__init__()
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

        self.ce = nn.BCEWithLogitsLoss()

    def forward(self, preds, labels):
        loss_CE = self.ce(preds, labels.float())
        
        preds = torch.sigmoid(preds)
        axes = tuple(range(2, preds.ndim))
        intersection = (preds * labels).sum(axes)
        sum_pred = preds.sum(axes)
        sum_gt = labels.sum(axes)
        summ = sum_pred + sum_gt
    
        dice = 2 * intersection / (summ + self.smooth)
        loss_Dice = torch.mean(1.0 - dice)
        
        loss = self.weight_ce * loss_CE + self.weight_dice * loss_Dice
        
        return loss
    
class CEDiceLossv2(nn.Module):
    def __init__(self, weight_ce=1.0, weight_dice=1.0, alpha=0.5, beta=0.5, smooth=1e-4,):
        super(CEDiceLossv2, self).__init__()
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

        self.ce = nn.CrossEntropyLoss()

    def forward(self, preds, labels):
        loss_CE = self.ce(preds, labels[:,0].long())
        
        preds = torch.softmax(preds, dim=1)
        axes = tuple(range(2, preds.ndim))
        intersection = (preds * labels).sum(axes)
        sum_pred = preds.sum(axes)
        sum_gt = labels.sum(axes)
        summ = sum_pred + sum_gt
    
        dice = 2 * intersection / (summ + self.smooth)
        loss_Dice = torch.mean(1.0 - dice)
        
        loss = self.weight_ce * loss_CE + self.weight_dice * loss_Dice
        
        return loss
    
class MemoryEfficientSoftDiceLoss(nn.Module):
    def __init__(self, apply_nonlin: Callable = None, batch_dice: bool = False, do_bg: bool = True, smooth: float = 1.,
                 ddp: bool = True):
        """
        saves 1.6 GB on Dataset017 3d_lowres
        """
        super(MemoryEfficientSoftDiceLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.ddp = ddp

    def forward(self, x, y, loss_mask=None):
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        # make everything shape (b, c)
        axes = tuple(range(2, x.ndim))

        with torch.no_grad():
            if x.ndim != y.ndim:
                y = y.view((y.shape[0], 1, *y.shape[1:]))

            if x.shape == y.shape:
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = y
            else:
                y_onehot = torch.zeros(x.shape, device=x.device, dtype=torch.bool)
                y_onehot.scatter_(1, y.long(), 1)

            if not self.do_bg:
                y_onehot = y_onehot[:, 1:]

            sum_gt = y_onehot.sum(axes) if loss_mask is None else (y_onehot * loss_mask).sum(axes)

        # this one MUST be outside the with torch.no_grad(): context. Otherwise no gradients for you
        if not self.do_bg:
            x = x[:, 1:]

        if loss_mask is None:
            intersect = (x * y_onehot).sum(axes)
            sum_pred = x.sum(axes)
        else:
            intersect = (x * y_onehot * loss_mask).sum(axes)
            sum_pred = (x * loss_mask).sum(axes)

        if self.batch_dice:
            intersect = intersect.sum(0)
            sum_pred = sum_pred.sum(0)
            sum_gt = sum_gt.sum(0)

        dc = (2 * intersect + self.smooth) / (torch.clip(sum_gt + sum_pred + self.smooth, 1e-8))

        dc = dc.mean()
        return -dc
    
class DC_and_BCE_loss(nn.Module):
    def __init__(self, bce_kwargs, soft_dice_kwargs, weight_ce=1, weight_dice=1, 
                 dice_class=MemoryEfficientSoftDiceLoss):
        """
        DO NOT APPLY NONLINEARITY IN YOUR NETWORK!

        target mut be one hot encoded
        IMPORTANT: We assume use_ignore_label is located in target[:, -1]!!!

        :param soft_dice_kwargs:
        :param bce_kwargs:
        :param aggregate:
        """
        super(DC_and_BCE_loss, self).__init__()
        
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce

        self.ce = nn.BCEWithLogitsLoss(**bce_kwargs)
        self.dc = dice_class(apply_nonlin=torch.sigmoid, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        dc_loss = self.dc(net_output, target, loss_mask=None)
        ce_loss = self.ce(net_output, target.float())
        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result