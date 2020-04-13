import torch
from torch import nn
import torch.nn.functional as F

class LMLoss(nn.Module):
    ''' Classic LM Loss '''
    def __init__(self, lm_criterion, opt=None):
        super(LMLoss, self).__init__()
        self.lm_criterion = lm_criterion
        self.opt = opt

    def forward(self, lm_logits, X, mask):
        x_shifted = X[:, 1:, 0].contiguous().view(-1)
        mask = mask[:, 1:].view(-1, mask.size(-1) - 1).float()
        lm_logits = lm_logits[:, :-1, :].contiguous().view(-1, lm_logits.size(-1))
        lm_losses = self.lm_criterion(lm_logits, x_shifted)
        lm_losses = lm_losses.view(X.size(0), X.size(1) - 1)
        lm_losses = lm_losses * mask
        lm_losses = lm_losses.sum(1) / torch.sum(mask, 1)
        return lm_losses



class ParagraphLoss(nn.Module):
    ''' LM Loss but ignoring the first n_ctx tokens '''
    def __init__(self, lm_criterion, opt=None, n_ctx=102, gen_len=401):
        super(ParagraphLoss, self).__init__()
        self.lm_criterion = lm_criterion
        self.opt = opt
        self.ctx = n_ctx
        self.tgt = gen_len

    def forward(self, lm_logits, X, mask):
        ## LM Loss, but ignoring the ctx tokens
        x_shifted = X[:, self.ctx:].contiguous().view(-1) #[102:] (text only)
        mask = mask[:, self.ctx:].view(-1, mask.size(-1) - (self.ctx)).float() #[102:]
        lm_logits = lm_logits[:, self.ctx-1:-1, :].contiguous().view(-1, lm_logits.size(-1)) #shifted over predictions
        lm_losses = self.lm_criterion(lm_logits, x_shifted)
        lm_losses = lm_losses.view(X.size(0), -1)
        lm_losses = lm_losses * mask
        lm_losses = lm_losses.sum(1) / torch.sum(mask, 1)
        return lm_losses
