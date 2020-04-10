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


class ParagraphAndAuxLoss(ParagraphLoss):
    def __init__(self, lm_criterion, opt=None, n_ctx=102, gen_len=401):
        super(ParagraphAndAuxLoss, self).__init__(lm_criterion,opt,n_ctx,gen_len)
        self.lm_criterion = lm_criterion
        self.opt = opt
        self.ctx = n_ctx
        self.tgt = gen_len

    def forward(self, lm_logits, hdec, *args, splitlosses=False):
        X, mask, prevparrep, currparrep = args
        lm_losses = super().forward(lm_logits, X, mask)
        simloss = 1 - F.cosine_similarity(hdec, currparrep.type_as(hdec), dim = -1)
        diffloss = F.cosine_similarity(hdec, prevparrep.type_as(hdec), dim = -1)
        array = [lm_losses, simloss, diffloss]
        if splitlosses:
            return array + [torch.zeros(1, device=lm_losses.device)]
        return sum(array)


class SummaryLoss(nn.Module):

    def __init__(self, lm_criterion, opt=None, n_ctx=511, n_tgt=110):
        super(SummaryLoss, self).__init__()
        self.lm_criterion = lm_criterion
        self.opt = opt
        self.ctx = n_ctx
        self.tgt = n_tgt

    def forward(self, lm_logits, X, mask):
        x_shifted = X[:, 1+self.ctx+1:, 0].contiguous().view(-1)
        mask = mask[:, 1+self.ctx+1:].view(-1, mask.size(-1) - (self.ctx+2)).float()
        lm_logits = lm_logits[:, 1+self.ctx:-1, :].contiguous().view(-1, lm_logits.size(-1))
        lm_losses = self.lm_criterion(lm_logits, x_shifted)
        lm_losses = lm_losses.view(X.size(0), -1)
        lm_losses = lm_losses * mask
        lm_losses = lm_losses.sum(1) / torch.sum(mask, 1)
        return lm_losses


class SummaryAndPlanLoss(SummaryLoss):

    def __init__(self, lm_criterion, opt=None, n_ctx=511, n_tgt=110):
        super(SummaryAndPlanLoss, self).__init__(lm_criterion,opt,n_ctx,n_tgt)
        self.planlossalpha = opt.get('planlossalpha', 0)
        self.spreadalpha = opt.get('spreadalpha', 0)
        self.entalpha = opt.get('entalpha', 0)
        
    def forward(self, lm_logits, prec, plans, Smat, Amat, X, mask, split=False):
        #lm_logits, prec, plans, Smat, _  = output
        #if lm_logits.size(1) < self.ctx+self.tgt+3:
        #    lm_logits = torch.cat( (torch.zeros(lm_logits.size(0), self.ctx+1, lm_logits.size(2)).cuda(), lm_logits), dim = 1)
        lm_losses = super().forward(lm_logits, X, mask)
        lossterms = [lm_losses]
        lossterms.append(prec.mean()) #[B,1]
        if not split:
            lossterms[-1] *= self.planlossalpha

        if (plans!=0).any():
            plansim = plans
            #print(plans[0,:,:3])
            plansim = F.normalize(plansim+1E-7, dim=-1)
            dotprod = torch.matmul(plansim, torch.transpose(plansim,1,2)) #(B,C,D) * (B,D,C) == (B,C,C)
            spread = 1-dotprod.tril(diagonal=-1).sum((1,2))/(torch.ones(dotprod.shape,device="cuda").tril(diagonal=-1).sum((1,2)))
            lossterms.append(-spread.mean()) #[B,1]
            if not split:
                lossterms[-1] *= self.spreadalpha
        else:
            lossterms.append(plans.mean())
            
        if (Smat>0).all():
            S = Smat  #[B,S,C]
            #print (Smat[0,:3,:])
            b = S * S.log()/S.size(1)
            b = -1.0 * b.sum((1,2))
            lossterms.append(b.mean()) #[B,1]
            if not split:
                lossterms[-1] *= self.entalpha
        else:
            lossterms.append(Smat.mean())

        if split:
            return lossterms
        return lossterms[0]+lossterms[1]+lossterms[2]+lossterms[3]
