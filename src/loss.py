import torch
from torch import nn
import torch.nn.functional as F

class LMLoss(nn.Module):
    def __init__(self, lm_criterion, opt=None):
        super(LMLoss, self).__init__()
        self.lm_criterion = lm_criterion
        self.opt = opt

    def forward(self, lm_logits, X, mask):
        #print (lm_logits.size())
        x_shifted = X[:, 1:, 0].contiguous().view(-1)
        mask = mask[:, 1:].view(-1, mask.size(-1) - 1).float()
        lm_logits = lm_logits[:, :-1, :].contiguous().view(-1, lm_logits.size(-1))
        lm_losses = self.lm_criterion(lm_logits, x_shifted)
        lm_losses = lm_losses.view(X.size(0), X.size(1) - 1)
        lm_losses = lm_losses * mask
        lm_losses = lm_losses.sum(1) / torch.sum(mask, 1)
        return lm_losses



class ParagraphLoss(nn.Module):
    def __init__(self, lm_criterion, opt=None, n_ctx=102, gen_len=401):
        super(ParagraphLoss, self).__init__()
        self.lm_criterion = lm_criterion
        self.opt = opt
        self.ctx = n_ctx
        self.tgt = gen_len

    def forward(self, lm_logits, X, mask, splitlosses=False):
        #X=[B,S,2] lml =[B,S,V] mask=[B,S]
        x_shifted = X[:, self.ctx:].contiguous().view(-1) #[w1...end pad]
        #xshifted = [B*400] ==: [0,0 0,1...0,399 1,0 1,1...]
        mask = mask[:, self.ctx:].view(-1, mask.size(-1) - (self.ctx)).float() #[11...10]
        #mask = [B,400]
        lm_logits = lm_logits[:, self.ctx-1:-1, :].contiguous().view(-1, lm_logits.size(-1)) #[w'1...]
        #lmlogits = [B*400,V]
        lm_losses = self.lm_criterion(lm_logits, x_shifted)
        #lm_losses = [B*400]
        lm_losses = lm_losses.view(X.size(0), -1)
        #lm_losses = [B,400]
        lm_losses = lm_losses * mask
        lm_losses = lm_losses.sum(1) / torch.sum(mask, 1)
        if splitlosses:
            return [lm_losses]+3*[torch.zeros(1, device=lm_losses.device)]
        return lm_losses
