import copy
import json
import math
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def swish(x):
    return x * torch.sigmoid(x)


ACT_FNS = {
    'relu': nn.ReLU,
    'swish': swish,
    'gelu': gelu
}


class LayerNorm(nn.Module):
    "Construct a layernorm module in the OpenAI style (epsilon inside the square root)."

    def __init__(self, n_state, e=1e-5):
        super(LayerNorm, self).__init__()
        self.g = nn.Parameter(torch.ones(n_state))
        self.b = nn.Parameter(torch.zeros(n_state))
        self.e = e

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.e)
        return self.g * x + self.b


class Conv1D(nn.Module):
    def __init__(self, nf, rf, nx):
        super(Conv1D, self).__init__()
        self.rf = rf
        self.nf = nf
        if rf == 1:  # faster 1x1 conv
            w = torch.empty(nx, nf)
            nn.init.normal_(w, std=0.02)
            self.w = Parameter(w)
            self.b = Parameter(torch.zeros(nf))
        else:  # was used to train LM
            raise NotImplementedError

    def forward(self, x):
        if self.rf == 1:
            size_out = x.size()[:-1] + (self.nf,)
            try:
                x = torch.addmm(self.b, x.view(-1, x.size(-1)), self.w)
            except:
                x = x.contiguous()
                x = torch.addmm(self.b, x.view(-1, x.size(-1)), self.w)
            x = x.view(*size_out)
        else:
            raise NotImplementedError
        return x


class EncoderAttention(nn.Module):
    def __init__(self, nx, n_ctx, cfg, scale=False):
        super(EncoderAttention, self).__init__()
        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        assert n_state % cfg.n_head == 0
        self.register_buffer('b', torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        self.n_head = cfg.n_head
        self.split_size = n_state
        self.scale = scale
        self.c_attn = Conv1D(n_state * 3, 1, nx)
        self.c_proj = Conv1D(n_state, 1, nx)
        self.attn_dropout = nn.Dropout(cfg.attn_pdrop)
        self.resid_dropout = nn.Dropout(cfg.resid_pdrop)
        self.cfg = cfg

    def _attn(self, q, k, v, mask, extra=None):
        w = torch.matmul(q, k) # [B,nh,S,D] [B,nh,D,S] --> [B,nh,S,S]
        #print(w.shape, q.shape, k.shape)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        b = self.b[:, :, :w.size(-2), :w.size(-2)]
        b = b * mask[:,:b.size(-1)].unsqueeze(1).unsqueeze(2).type_as(b)
        #print(b.size())
        w = w * b + -1e9 * (1 - b)
        #print(w.size())
        w = nn.Softmax(dim=-1)(w)
        postW = self.attn_dropout(w)
        return torch.matmul(postW, v), w

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def merge_adj(self, x):

        x = x.permute(0, 2, 3, 1).contiguous() #B,s,s,nh
        return x.mean(dim=3) #B, s, s

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)
        else:
            return x.permute(0, 2, 1, 3)

    def forward(self, x, mask, extra=None, sim=False, ctx=False):
        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        a,adj = self._attn(query, key, value, mask)  #w=[B,nh,s,s] a=[B,nh,s,d/nh]
        adj = self.merge_adj(adj) 
        a = self.merge_heads(a) #[B,nh,s,d]
        a = self.c_proj(a)
        a = self.resid_dropout(a)
        if sim:
            return a, adj
        return a


class DecoderSelfAttention(nn.Module):
    def __init__(self, nx, n_ctx, cfg, scale=False):
        super(DecoderSelfAttention, self).__init__()
        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        assert n_state % cfg.n_head == 0
        self.register_buffer('b', torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        self.n_head = cfg.n_head
        self.split_size = n_state
        self.scale = scale
        self.c_attn = Conv1D(n_state * 3, 1, nx)
        self.c_proj = Conv1D(n_state, 1, nx)
        self.c_extra = Conv1D(n_state * 2, 1, nx)
        self.attn_dropout = nn.Dropout(cfg.attn_pdrop)
        self.resid_dropout = nn.Dropout(cfg.resid_pdrop)
        self.cfg = cfg

    def _attn(self, q, k, v, mask, extra=None, extramask=None):
        w = torch.matmul(q, k) # [B,nh,S,D] [B,nh,D,S] --> [B,nh,S,S]
        #print(w.shape, q.shape, k.shape)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        # w = w * self.b + -1e9 * (1 - self.b)  # TF implem method: mask_attn_weights
        # XD: self.b may be larger than w, so we need to crop it
        #b = self.b * mask.unsqueeze(1).unsqueeze(1).type_as(self.b)
        #print(w.size(), mask.unsqueeze(1).unsqueeze(1).size())
        b = self.b[:, :, :w.size(-2), :w.size(-2)]
        b = b * mask[:,:b.size(-1)].unsqueeze(1).unsqueeze(2).type_as(b)
        if extra is not None:
            p = extra.size(1)
            if extramask is not None:
                pt = torch.ones(b.size(0),b.size(1),b.size(2), p).cuda()
                pt = pt * extramask[:, :p].unsqueeze(1).unsqueeze(2).type_as(pt)
                b = torch.cat((pt, b), dim = 3)
            else:
                b = torch.cat((torch.ones(b.size(0),b.size(1),b.size(2), p).cuda(),b),dim=3)

     
        #print(b.size())
        w = w * b + -1e9 * (1 - b)
        #print(w.size())
        w = nn.Softmax(dim=-1)(w)
        postW = self.attn_dropout(w)
        
        return torch.matmul(postW, v), w

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def merge_adj(self, x):

        x = x.permute(0, 2, 3, 1).contiguous() #B,s,s,nh
        return x.mean(dim=3) #B, s, s

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)
        else:
            return x.permute(0, 2, 1, 3)

    def forward(self, x, mask,  extra=None, sim=False, ctx=False, extramask=None):
        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)
        if extra is not None:
            ek,ev = self.c_extra(extra).split(self.split_size,dim=2)
            key = torch.cat((ek,key), dim=1)
            value = torch.cat((ev,value), dim=1)

        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        a,adj = self._attn(query, key, value, mask, extra, extramask)  #w=[B,nh,s,s] a=[B,nh,s,d/nh]
        adj = self.merge_adj(adj) 
        a = self.merge_heads(a) #[B,nh,s,d]
        a = self.c_proj(a)
        a = self.resid_dropout(a)
        if sim:
            return a, adj
        return a



class DecoderNotSelfAttention(nn.Module):
    def __init__(self, nx, n_ctx, cfg, scale=False):
        super(DecoderNotSelfAttention, self).__init__()
        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        assert n_state % cfg.n_head == 0
        self.register_buffer('b', torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        self.n_head = cfg.n_head
        self.split_size = n_state
        self.scale = scale
        self.c_attn = Conv1D(n_state, 1, nx)
        self.c_proj = Conv1D(n_state, 1, nx)
        self.c_extra = Conv1D(n_state * 2, 1, nx)
        self.attn_dropout = nn.Dropout(cfg.attn_pdrop)
        self.resid_dropout = nn.Dropout(cfg.resid_pdrop)
        self.cfg = cfg

    def _attn(self, q, k, v, mask, extra=None, extramask=None):
        w = torch.matmul(q, k) # [B,nh,S,D] [B,nh,D,S] --> [B,nh,S,S]
        #print(w.shape, q.shape, k.shape)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        # w = w * self.b + -1e9 * (1 - self.b)  # TF implem method: mask_attn_weights
        # XD: self.b may be larger than w, so we need to crop it
        #b = self.b * mask.unsqueeze(1).unsqueeze(1).type_as(self.b)
        #print(w.size(), mask.unsqueeze(1).unsqueeze(1).size())
        b = self.b[:, :, :w.size(-2), :w.size(-2)]
        b = b * mask[:,:b.size(-1)].unsqueeze(1).unsqueeze(2).type_as(b)
        if extra is not None:
            p = extra.size(1)
            if extramask is not None:
                #print(b.size(), p, extramask.size())
                b = torch.ones(b.size(0),b.size(1),b.size(2), p).cuda()
                b = b * extramask[:,:p].unsqueeze(1).unsqueeze(2).type_as(b)
            else:
                b = torch.ones(b.size(0),b.size(1),b.size(2), p).cuda()
     
        #print(b.size())
        w = w * b + -1e9 * (1 - b)
        #print(w.size())
        w = nn.Softmax(dim=-1)(w)
        postW = self.attn_dropout(w)
        return torch.matmul(postW, v), w

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def merge_adj(self, x):

        x = x.permute(0, 2, 3, 1).contiguous() #B,s,s,nh
        return x.mean(dim=3) #B, s, s

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)
        else:
            return x.permute(0, 2, 1, 3)

    def forward(self, x, mask, extra=None, sim=False, ctx=False, extramask=None):
        query = self.c_attn(x)
        #print(extra.size())
        key, value = self.c_extra(extra).split(self.split_size,dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        a,adj = self._attn(query, key, value, mask, extra, extramask)  #w=[B,nh,s,s] a=[B,nh,s,d/nh]
        adj = self.merge_adj(adj) 
        a = self.merge_heads(a) #[B,nh,s,d]
        a = self.c_proj(a)
        a = self.resid_dropout(a)
        if sim:
            return a, adj
        return a

class BiAttention(EncoderAttention):
    def __init__(self, nx=768, n_ctx=512, cfg=None, scale=False):
        super(BiAttention, self).__init__(nx, n_ctx, cfg, scale)
        self.register_buffer('b', torch.ones(n_ctx, n_ctx).view(1, 1, n_ctx, n_ctx))

class MLP(nn.Module):
    def __init__(self, n_state, cfg):  # in MLP: n_state=3072 (4 * n_embd)
        super(MLP, self).__init__()
        nx = cfg.n_embd
        self.c_fc = Conv1D(n_state, 1, nx)
        self.c_proj = Conv1D(nx, 1, n_state)
        self.act = ACT_FNS[cfg.afn]
        self.dropout = nn.Dropout(cfg.resid_pdrop)

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)


class EncoderBlock(nn.Module):
    def __init__(self, n_ctx, cfg, scale=False):
        super(EncoderBlock, self).__init__()
        nx = cfg.n_embd
        self.attn = EncoderAttention(nx, n_ctx, cfg, scale)
        self.ln_1 = LayerNorm(nx)
        self.mlp = MLP(4 * nx, cfg)
        self.ln_2 = LayerNorm(nx)
        self.cfg = cfg

    def forward(self, x, mask, extra=None, A=False, ctx_extra=None):
        a,sim = self.attn(x, mask, None, sim=True)
        n = self.ln_1(x+a)
        m = self.mlp(n)
        h = self.ln_2(n + m)
        if A:
            return h, sim
        return h

class DecoderSingleSelfBlock(nn.Module):
    def __init__(self, n_ctx, cfg, scale=False):
        super(DecoderSingleSelfBlock, self).__init__()
        nx = cfg.n_embd
        self.attn = DecoderSelfAttention(nx, n_ctx, cfg, scale)
        self.ln_1 = LayerNorm(nx)
        self.mlp = MLP(4 * nx, cfg)
        self.ln_2 = LayerNorm(nx)
        self.cfg = cfg

    def forward(self, x, mask, extra=None, A=False, ctx_extra=None, extramask1 = None, extramask2 =None):
        a, sim = self.attn(x, mask, extra, sim=True, extramask = extramask1)
        n = self.ln_1(x+a)
        m = self.mlp(n)
        h = self.ln_2(n + m)
        if A:
            return h, sim
        return h

class DecoderMultiSelfBlock(nn.Module):
    def __init__(self, n_ctx, cfg, scale=False):
        super(DecoderMultiSelfBlock, self).__init__()
        nx = cfg.n_embd
        self.attn = DecoderSelfAttention(nx, n_ctx, cfg, scale)
        self.ln_1 = LayerNorm(nx)
        self.attnlast = DecoderSelfAttention(nx, n_ctx, cfg, scale)
        self.ln_last = LayerNorm(nx)
        self.mlp = MLP(4 * nx, cfg)
        self.ln_2 = LayerNorm(nx)
        self.cfg = cfg

    def forward(self, x, mask, extra=None, A=False, ctx_extra=None, extramask1 = None, extramask2 =None):
        a, sim = self.attn(x, mask, extra, sim=True, extramask = extramask1)
        n = self.ln_1(x+a)
        n2 = self.attnlast(n, mask, ctx_extra, sim=False, extramask = extramask2 )
        nout = self.ln_last(n+n2)
        m = self.mlp(nout)
        h = self.ln_2(nout + m)
        if A:
            return h, sim
        return h


class DecoderSingleNotSelfBlock(nn.Module):
    def __init__(self, n_ctx, cfg, scale=False):
        super(DecoderSingleNotSelfBlock, self).__init__()
        nx = cfg.n_embd
        self.attn = EncoderAttention(nx, n_ctx, cfg, scale)
        self.attnfirst = DecoderNotSelfAttention(nx, n_ctx, cfg, scale)
        self.ln_1 = LayerNorm(nx)
        self.ln_first = LayerNorm(nx)
        self.mlp = MLP(4 * nx, cfg)
        self.ln_2 = LayerNorm(nx)
        self.cfg = cfg

    def forward(self, x, mask, extra=None, A=False, ctx_extra=None, extramask1 = None, extramask2 =None):
        a, sim = self.attn(x, mask, sim=True)
        n = self.ln_1(x+a)
        n2 = self.attnfirst(n, mask, extra, sim=False, extramask = extramask1)
        nout = self.ln_first(n+n2)
        m = self.mlp(nout)
        h = self.ln_2(nout + m)
        if A:
            return h, sim
        return h

class DecoderMultiNotSelfBlock(nn.Module):
    def __init__(self, n_ctx, cfg, scale=False):
        super(DecoderMultiNotSelfBlock, self).__init__()
        nx = cfg.n_embd
        self.attn = EncoderAttention(nx, n_ctx, cfg, scale)
        self.ln_1 = LayerNorm(nx)
        self.attnlast = DecoderNotSelfAttention(nx, n_ctx, cfg, scale)
        self.ln_last = LayerNorm(nx)
        self.attnfirst = DecoderNotSelfAttention(nx, n_ctx, cfg, scale)
        self.ln_first = LayerNorm(nx)
        self.mlp = MLP(4 * nx, cfg)
        self.ln_2 = LayerNorm(nx)
        self.cfg = cfg

    def forward(self, x, mask, extra=None, A=False, ctx_extra=None, extramask1 =None, extramask2 =None):
        a, sim = self.attn(x, mask, extra, sim=True)
        n = self.ln_1(x+a)
        n2 = self.attnfirst(n, mask, extra, sim=False, extramask = extramask1)
        nout = self.ln_first(n+n2)
        n2 = self.attnlast(nout, mask, ctx_extra, sim=False, extramask = extramask2)
        nout = self.ln_last(nout+n2)
        m = self.mlp(nout)
        h = self.ln_2(nout + m)
        if A:
            return h, sim
        return h


class EncoderModel(nn.Module):
    """ Transformer model """

    def __init__(self, cfg, vocab=40990, n_ctx=512):
        super(EncoderModel, self).__init__()
        self.vocab = vocab
        self.embed = nn.Embedding(vocab, cfg.n_embd)
        self.drop = nn.Dropout(cfg.embd_pdrop)
        block = EncoderBlock(n_ctx, cfg, scale=True)
        self.h = nn.ModuleList([copy.deepcopy(block) for _ in range(cfg.n_layer)])
        nn.init.normal_(self.embed.weight, std=0.02)
        self.n_ctx = n_ctx

    def forward(self, x, mask, extra=None, returnadj = False, ctx_extra=None):
        x = x.view(-1, x.size(-2), x.size(-1))
        e = self.embed(x)
        # Add the position information to the input embeddings
        h = e.sum(dim=2)
        A=None
        for block in self.h:
            h, A = block(h, mask, extra, A=True, ctx_extra=ctx_extra)
        if returnadj:
            return h, A
        return h

class DecoderModel(nn.Module):
    """ Transformer model """

    def __init__(self, cfg, vocab=40990, n_ctx=512, multi=False, selfatt=True):
        super(DecoderModel, self).__init__()
        self.vocab = vocab
        self.embed = nn.Embedding(vocab, cfg.n_embd)
        if cfg.positions:
            self.pos = True
            self.ppos = nn.Embedding(cfg.n_plans, cfg.n_embd)
            nn.init.normal_(self.ppos.weight, 0, 0.01)
            if multi:
                #self.multi = True
                self.hpos = nn.Embedding(n_ctx, cfg.n_embd)
                nn.init.normal_(self.hpos.weight, 0, 0.01)
        else:
            self.pos=False
            
        self.multi = multi
        self.n_plans = cfg.n_plans

        self.drop = nn.Dropout(cfg.embd_pdrop)

        if not multi:
            if selfatt:
                block = DecoderSingleSelfBlock(n_ctx, cfg, scale=True)
            else:
                block = DecoderSingleNotSelfBlock(n_ctx, cfg, scale=True)
        else:
            if selfatt:
                block = DecoderMultiSelfBlock(n_ctx, cfg, scale=True)
            else:
                block = DecoderMultiNotSelfBlock(n_ctx, cfg, scale=True)

        self.h = nn.ModuleList([copy.deepcopy(block) for _ in range(cfg.n_layer)])
        nn.init.normal_(self.embed.weight, std=0.02)
        self.n_ctx = n_ctx

    def forward(self, x, mask, extra=None, returnadj = False, ctx_extra=None, em1 = None, em2 = None):
        x = x.view(-1, x.size(-2), x.size(-1))
        e = self.embed(x)
        # Add the position information to the input embeddings
        h = e.sum(dim=2)
        A=None

        if self.pos:
            if extra is not None:
                position_ids = torch.arange(self.n_plans, dtype=torch.long, device=extra.device)
                position_ids = position_ids.unsqueeze(0).expand_as(extra)
                extraemb = self.ppos(position_ids)
                extra += extraemb
            if ctx_extra is not None and self.multi:
                position_ids = torch.arange(ctx_extra.size(1), dtype=torch.long, device=ctx_extra.device)
                position_ids = position_ids.unsqueeze(0).expand_as(ctx_extra)
                ctxemb = self.hpos(position_ids)
                ctx_extra += ctxemb
        for block in self.h:
            h, A = block(h, mask, extra, A=True, ctx_extra=ctx_extra, extramask1=em1, extramask2 = em2)
        if returnadj:
            return h, A
        return h


class LMHead(nn.Module):
    """ Language Model Head for the transformer """

    def __init__(self, model, cfg, trunc_and_reshape=True):
        super(LMHead, self).__init__()
        self.n_embd = cfg.n_embd
        embed_shape = model.embed.weight.shape
        self.decoder = nn.Linear(embed_shape[1], embed_shape[0], bias=False)
        self.decoder.weight = model.embed.weight # Tied weights
        self.trunc_and_reshape = trunc_and_reshape  # XD

    def forward(self, h):
        # Truncated Language modeling logits (we remove the last token)
        h_trunc = h[:, :-1].contiguous().view(-1, self.n_embd) \
            if self.trunc_and_reshape else h  # XD
        lm_logits = self.decoder(h_trunc)
        return lm_logits

# XD
class LMModel(nn.Module):
    """ Transformer with language model head only """
    def __init__(self, cfg, vocab=40990, n_ctx=512, gen_len=110, return_probs=False):
        super(LMModel, self).__init__()
        self.transformer = EncoderModel(cfg, vocab=vocab, n_ctx= n_ctx+gen_len+3)
        self.lm_head = LMHead(self.transformer, cfg, trunc_and_reshape=False)
        self.return_probs = return_probs
        self.n_ctx=n_ctx
        self.cfg = cfg
        

        #if self.return_probs:
        pos_emb_mask = torch.zeros(1, 1, vocab)
        lstnum = max(n_ctx,gen_len)
        pos_emb_mask[:, :, -lstnum:] = -1e12
        self.register_buffer('pos_emb_mask', pos_emb_mask)

    def forward(self, pad_output, mask_output=None, text_encoder=None, device=None, beam=0, gen_len=110, k=0, p=0, decoding_strategy=0, log=True, generate=False, min_len=None, extralosses=False):
        if generate:
            return self.generate(pad_output, mask_output, text_encoder, device, beam, gen_len, k, p, decoding_strategy, min_len=min_len)
        return self._forward(pad_output, mask_output, log, extralosses=extralosses)


    def _forward(self, x, mask_output, log=True, return_probs=False, extralosses=False):
        h = self.transformer(x, mask_output)
        lm_logits = self.lm_head(h)
        if self.return_probs or return_probs:
            if log:
                lm_logits = F.log_softmax((lm_logits + self.pos_emb_mask), dim=-1)
            else:
                lm_logits = F.softmax((lm_logits + self.pos_emb_mask), dim=-1)
        if not extralosses:
            return lm_logits
        return lm_logits, torch.zeros(x.size(0),1).cuda(), torch.zeros(x.size(0),1).cuda(), torch.zeros(x.size(0),1).cuda(), torch.zeros(x.size(0),1).cuda()

    def append_batch(self, X, next_idx):
        next_pos = X[:, -1:, 1] + 1
        next_x = torch.cat((next_idx, next_pos), -1).unsqueeze(1)
        return torch.cat((X, next_x), 1)


    def sample(self, pad_output, mask, classify_idx, text_encoder, gen_len=110, k=0, p=0, decoding_strategy=0, min_len=None):
        XMB = pad_output
        seen_trigrams = [{} for _ in range(XMB.size(0))]
        for _ in range(gen_len):
            lm_probs = self._forward(XMB, mask, return_probs=True, log=False)
            dist = lm_probs[:, -1, :].squeeze(1)
            if k == 0 and p ==0:
                next_idx = torch.multinomial(lm_probs[:, -1, :], 1)
            else:
                if p == 0:
                    # Sample from top k
                    values, indices = dist.topk(k)
                    next_idx = indices.gather(-1, torch.multinomial(values, 1))
                    if _ == 2:
                        for i in range(XMB.size(0)):
                            bigram = (XMB[i, -2, 0].item(), XMB[i, -1, 0].item())
                            seen_trigrams[i][bigram] = [next_idx[i].item()]
                    elif _ > 2:
                        for i in range(XMB.size(0)):
                            bigram = (XMB[i, -2, 0].item(), XMB[i, -1, 0].item())
                            if bigram in seen_trigrams[i]:
                                for value in seen_trigrams[i][bigram]:
                                    dist[i, value] = 0
                        values, indices = dist.topk(k)
                        next_idx = indices.gather(-1, torch.multinomial(values, 1))
                        for i in range(XMB.size(0)):
                            bigram = (XMB[i, -2, 0].item(), XMB[i, -1, 0].item())
                            if bigram not in seen_trigrams[i]:
                                seen_trigrams[i][bigram] = []
                            seen_trigrams[i][bigram].append(next_idx[i].item())
                else:
                    # Sample from top p
                    # [.3,.1,.05,.3...],.65 
                    # --> [0,3,1,...]
                    indices = torch.argsort(dist,dim=1,descending=True)
                    # --> [.3,.3,.1,...]
                    values = dist.gather(-1,indices)
                    # --> [.3,.6,.7,...]
                    probsum = torch.cumsum(values,dim=1)
                    # --> [1,1,1,0,...]
                    include = 1- ((probsum.gt(p*.01)) & ((probsum-values).gt(p*.01)))
                    ## --> [0,3,1]
                    ##newindices = (indices,-1,include)
                    # --> [.3,.3,.1,-1e10]
                    ##print(values.shape, newindices.shape)
                    newdist = torch.where(include, values, torch.zeros_like(values) + 1e-10)#values.gather(-1,newindices)

                    next_idx = indices.gather(-1, torch.multinomial(newdist, 1))
                    if _ == 2:
                        for i in range(XMB.size(0)):
                            bigram = (XMB[i, -2, 0].item(), XMB[i, -1, 0].item())
                            seen_trigrams[i][bigram] = [next_idx[i].item()]
                    elif _ > 2:
                        for i in range(XMB.size(0)):
                            bigram = (XMB[i, -2, 0].item(), XMB[i, -1, 0].item())
                            if bigram in seen_trigrams[i]:
                                for value in seen_trigrams[i][bigram]:
                                    dist[i, value] = 0

                        # [.3,.1,.05,.3...],.65 
                        # --> [0,3,1,...]
                        indices = torch.argsort(dist,dim=1,descending=True)
                        # --> [.3,.3,.1,...]
                        values = dist.gather(-1,indices)
                        # --> [.3,.6,.7,...]
                        probsum = torch.cumsum(values,dim=1)
                        # --> [1,1,1,0,...]
                        include = 1- ((probsum.gt(p*.01)) & ((probsum-values).gt(p*.01)))
                        # --> [0,3,1]
                        ##newindices = torch.index_select(indices,-1,include)
                        # --> [.3,.3,.1, 0, 0]
                        newdist = torch.where(include, values, torch.zeros_like(values) + 1e-10) #torch.gather(values,newindices)

                        next_idx = indices.gather(-1, torch.multinomial(newdist, 1))
                        for i in range(XMB.size(0)):
                            bigram = (XMB[i, -2, 0].item(), XMB[i, -1, 0].item())
                            if bigram not in seen_trigrams[i]:
                                seen_trigrams[i][bigram] = []
                            seen_trigrams[i][bigram].append(next_idx[i].item())
                #else:
                #    raise NotImplementedError
            XMB = self.append_batch(XMB, next_idx)
        return XMB[:, -gen_len:, 0]

    def generate(self, pad_output, mask, text_encoder, device, beam=0, gen_len=110, k=0, p=0, decoding_strategy=0, min_len=None):
        classify_idx = text_encoder.encoder['_classify_']
        target_toks = pad_output[:, 1:, 0]
        mask_pad = torch.ones(mask.size()).type_as(mask)
        mask_pad[:, :1] = mask[:, :1]
        mask = mask_pad

        pad_output = pad_output.to(device)
        XMB = pad_output[:, :1]
        if beam == 0:
            generated_toks = self.sample(XMB, mask, classify_idx, text_encoder, gen_len, k, p, decoding_strategy, min_len=min_len)
        else:
            raise NotImplementedError
        output = generated_toks.type_as(XMB), target_toks.type_as(XMB)
        return output


class JustDecoderModel(LMModel):
    def __init__(self, cfg, vocab=40990, n_ctx=512, gen_len=110, return_probs=False, pnum=5):
        super(JustDecoderModel,self).__init__(cfg, vocab, n_ctx, gen_len, return_probs=return_probs)
        self.pnum = pnum
        self.n_ctx = n_ctx
        self.gen_len = gen_len
        del self.transformer
        self.decoder = DecoderModel(cfg,vocab=vocab, n_ctx=gen_len+2, multi=cfg.hierarchical,selfatt=not(cfg.notselfattn))
        self.lm_head = LMHead(self.decoder, cfg, trunc_and_reshape=False)

    def _forward(self, x, mask_output, plans , log=True, return_probs=False):
        n_ctx = self.n_ctx
        h_dec = self.decoder(x, mask_output, extra=plans, ctx_extra=None, em1 = None, em2 = None)
        lm_logits = self.lm_head(h_dec)
        if self.return_probs or return_probs:
            if log:
                lm_logits = F.log_softmax((lm_logits + self.pos_emb_mask), dim=-1)
            else:
                lm_logits = F.softmax((lm_logits + self.pos_emb_mask), dim=-1)
        return lm_logits

    def forward(self, x, mask_output, plans, text_encoder=None, device=None, beam=0, gen_len=110, k=0, p=0, decoding_strategy=0, log=True, generate=False, min_len=None):
        if generate:
            return self.generate(x, mask_output, plans, text_encoder, device, beam, gen_len, k, p, decoding_strategy, min_len=min_len)
        return self._forward(x, mask_output, plans, log)

    def sample(self, XMB, mask, plans, classify_idx, text_encoder, gen_len=110, k=0, p=0, decoding_strategy=0, min_len=None):
        seen_trigrams = [{} for _ in range(XMB.size(0))]
        for _ in range(gen_len):
            h_dec = self.decoder(XMB, mask, extra=plans, ctx_extra=None, em1 = None, em2 = None)
            lm_logits = self.lm_head(h_dec)
            lm_probs = F.softmax((lm_logits + self.pos_emb_mask), dim=-1)
            dist = lm_probs[:, -1, :].squeeze(1)
            if k == 0 and p == 0:
                next_idx = torch.multinomial(lm_probs[:, -1, :], 1)
            else:
                if p ==0:
                    # Sample from top k
                    values, indices = dist.topk(k)
                    next_idx = indices.gather(-1, torch.multinomial(values, 1))
                    if _ == 2:
                        for i in range(XMB.size(0)):
                            bigram = (XMB[i, -2, 0].item(), XMB[i, -1, 0].item())
                            seen_trigrams[i][bigram] = [next_idx[i].item()]
                    elif _ > 2:
                        for i in range(XMB.size(0)):
                            bigram = (XMB[i, -2, 0].item(), XMB[i, -1, 0].item())
                            if bigram in seen_trigrams[i]:
                                for value in seen_trigrams[i][bigram]:
                                    dist[i, value] = 0
                        values, indices = dist.topk(k)
                        next_idx = indices.gather(-1, torch.multinomial(values, 1))
                        for i in range(XMB.size(0)):
                            bigram = (XMB[i, -2, 0].item(), XMB[i, -1, 0].item())
                            if bigram not in seen_trigrams[i]:
                                seen_trigrams[i][bigram] = []
                            seen_trigrams[i][bigram].append(next_idx[i].item())
                else:
                    indices = torch.argsort(dist,dim=1,descending=True)
                    values = dist.gather(-1,indices)
                    probsum = torch.cumsum(values,dim=1)
                    include = 1- ((probsum.gt(p*.01)) & ((probsum-values).gt(p*.01)))
                    newdist = torch.where(include, values, torch.zeros_like(values) + 1e-10)

                    next_idx = indices.gather(-1, torch.multinomial(newdist, 1))
                    if _ == 2:
                        for i in range(XMB.size(0)):
                            bigram = (XMB[i, -2, 0].item(), XMB[i, -1, 0].item())
                            seen_trigrams[i][bigram] = [next_idx[i].item()]
                    elif _ > 2:
                        for i in range(XMB.size(0)):
                            bigram = (XMB[i, -2, 0].item(), XMB[i, -1, 0].item())
                            if bigram in seen_trigrams[i]:
                                for value in seen_trigrams[i][bigram]:
                                    dist[i, value] = 0
                        indices = torch.argsort(dist,dim=1,descending=True)
                        values = dist.gather(-1,indices)
                        probsum = torch.cumsum(values,dim=1)
                        include = 1- ((probsum.gt(p*.01)) & ((probsum-values).gt(p*.01)))
                        newdist = torch.where(include, values, torch.zeros_like(values) + 1e-10)

                        next_idx = indices.gather(-1, torch.multinomial(newdist, 1))
                        for i in range(XMB.size(0)):
                            bigram = (XMB[i, -2, 0].item(), XMB[i, -1, 0].item())
                            if bigram not in seen_trigrams[i]:
                                seen_trigrams[i][bigram] = []
                            seen_trigrams[i][bigram].append(next_idx[i].item())
                #else:
                #    raise NotImplementedError
            XMB = self.append_batch(XMB, next_idx)
        return XMB[:, -gen_len:, 0]

    def generate(self, pad_output, mask, plans, text_encoder, device, beam=0, gen_len=110, k=0, p=0, decoding_strategy=0, min_len=None):
        classify_idx = text_encoder.encoder['_classify_']
        target_toks = pad_output[:, 1:, 0]
        mask_pad = torch.ones(mask.size()).type_as(mask)
        mask_pad[:, :1] = mask[:, :1]
        mask = mask_pad

        pad_output = pad_output.to(device)
        XMB = pad_output[:, :1]
        if beam == 0:
            generated_toks = self.sample(XMB, mask,plans, classify_idx, text_encoder, gen_len, k, p, decoding_strategy, min_len=min_len)
        else:
            raise NotImplementedError
        output = generated_toks.type_as(XMB), target_toks.type_as(XMB)
        return output



#============================================================================================================================================================#





def load_openai_pretrained_model(model, n_ctx=-1, n_special=-1, n_transfer=12, n_embd=768, path='model/',
                                 path_names='model/'):
    # Load weights from TF model
    print("Loading weights...")
    names = json.load(open(path_names + 'parameters_names.json'))
    shapes = json.load(open(path + 'params_shapes.json'))
    offsets = np.cumsum([np.prod(shape) for shape in shapes])
    init_params = [np.load(path + 'params_{}.npy'.format(n)) for n in range(10)]
    init_params = np.split(np.concatenate(init_params, 0), offsets)[:-1]
    init_params = [param.reshape(shape) for param, shape in zip(init_params, shapes)]
    if n_ctx > 0:
        init_params[0] = init_params[0][:n_ctx]
    if n_special > 0:
        init_params[0] = np.concatenate(
            [init_params[1],
             (np.random.randn(n_special, n_embd) * 0.02).astype(np.float32),
             init_params[0]
             ], 0)
    else:
        init_params[0] = np.concatenate(
            [init_params[1],
             init_params[0]
             ], 0)
    del init_params[1]
    if n_transfer == -1:
        n_transfer = 0
    else:
        n_transfer = 1 + n_transfer * 12
    init_params = [arr.squeeze() for arr in init_params]

    try:
        assert model.embed.weight.shape == init_params[0].shape
    except AssertionError as e:
        e.args += (model.embed.weight.shape, init_params[0].shape)
        raise

    model.embed.weight.data = torch.from_numpy(init_params[0])

    for name, ip in zip(names[1:n_transfer], init_params[1:n_transfer]):
        name = name[6:]  # skip "model/"
        assert name[-2:] == ":0"
        name = name[:-2]
        name = name.split('/')
        pointer = model
        for m_name in name:
            if re.fullmatch(r'[A-Za-z]+\d+', m_name):
                l = re.split(r'(\d+)', m_name)
            else:
                l = [m_name]
            pointer = getattr(pointer, l[0])
            if len(l) >= 2:
                num = int(l[1])
                pointer = pointer[num]
        try:
            assert pointer.shape == ip.shape
        except AssertionError as e:
            e.args += (pointer.shape, ip.shape)
            raise
        pointer.data = torch.from_numpy(ip)


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


DEFAULT_CONFIG = dotdict({
    'n_embd': 768,
    'n_head': 12,
    'n_layer': 12,
    'embd_pdrop': 0.1,
    'attn_pdrop': 0.1,
    'resid_pdrop': 0.1,
    'afn': 'gelu',
    'clf_pdrop': 0.1})
