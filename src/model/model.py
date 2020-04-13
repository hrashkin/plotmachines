import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy

from transformers.modeling_gpt2 import *

class GPT2NeighborModel(GPT2Model):
    '''GPT2 model but with slightly altered foward function to include previous paragraph encoding as an additional input'''
    def __init__(self, config):
        super(GPT2NeighborModel, self).__init__(config)

    def forward(self, input_ids, past=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, labels=None, includeprev=False, x_prev=None):
        if includeprev:
            #if need to add previous paragraph
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            if token_type_ids is not None:
                token_type_ids = token_type_ids.view(-1, input_shape[-1])
            if position_ids is not None:
                position_ids = position_ids.view(-1, input_shape[-1])

            if past is None:
                past_length = 0
                past = [None] * len(self.h)
            else:
                past_length = past[0][0].size(-2)
            if position_ids is None:
                position_ids = torch.arange(past_length, input_ids.size(-1) + past_length, dtype=torch.long, device=input_ids.device)
                position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

            # Attention mask.
            if attention_mask is not None:
                attention_mask = attention_mask.view(-1, input_shape[-1])
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
                attention_mask = (1.0 - attention_mask) * -10000.0

            if head_mask is not None:
                if head_mask.dim() == 1:
                    head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                    head_mask = head_mask.expand(self.config.n_layer, -1, -1, -1, -1)
                elif head_mask.dim() == 2:
                    head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
                head_mask = head_mask.to(dtype=next(self.parameters()).dtype) # switch to fload if need + fp16 compatibility
            else:
                head_mask = [None] * self.config.n_layer

            inputs_embeds = self.wte(input_ids)
            position_embeds = self.wpe(position_ids)
            if token_type_ids is not None:
                token_type_embeds = self.wte(token_type_ids)
            else:
                token_type_embeds = 0

            ####### THIS IS THE PART that needs to be changed from inherited function:
            x_prev = x_prev.unsqueeze(1) #[b,1,d] + [d] = [b,1,d]
            inputs_embeds = torch.cat([x_prev, inputs_embeds [:,1:,:]], dim = 1)  #x_prev: [b, 1, d], h : [b, s, d]-->[B, s+1, D]
            ########### END HERE #########################

            hidden_states = inputs_embeds + position_embeds + token_type_embeds
            hidden_states = self.drop(hidden_states)

            output_shape = input_shape + (hidden_states.size(-1),)

            presents = ()
            all_attentions = []
            all_hidden_states = ()
            for i, (block, layer_past) in enumerate(zip(self.h, past)):
                if self.output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states.view(*output_shape),)

                outputs = block(hidden_states,
                                layer_past=layer_past,
                                attention_mask=attention_mask,
                                head_mask=head_mask[i])

                hidden_states, present = outputs[:2]
                if self.output_past:
                    presents = presents + (present,)

                if self.output_attentions:
                    all_attentions.append(outputs[2])

            hidden_states = self.ln_f(hidden_states)

            hidden_states = hidden_states.view(*output_shape)
            # Add last hidden state
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            outputs = (hidden_states,)
            if self.output_past:
                outputs = outputs + (presents,)
            if self.output_hidden_states:
                outputs = outputs + (all_hidden_states,)
            if self.output_attentions:
                # let the number of heads free (-1) so we can extract attention even after head pruning
                attention_output_shape = input_shape[:-1] + (-1,) + all_attentions[0].shape[-2:]
                all_attentions = tuple(t.view(*attention_output_shape) for t in all_attentions)
                outputs = outputs + (all_attentions,)

            return outputs  

        else:
            return super().forward(input_ids, past=past, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask)


class GPT2NeighborLMHeadModel(GPT2LMHeadModel):
    '''GPT2 LM Head model but with the GPT2NeighborModel Class'''
    def __init__(self, config):
        super(GPT2NeighborLMHeadModel, self).__init__(config)
        self.transformer = GPT2NeighborModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.init_weights()
        self.tie_weights()

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.lm_head,
                                   self.transformer.wte)

    def forward(self, input_ids, past=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, labels=None, includeprev=False, x_prev=None):
        transformer_outputs = self.transformer(input_ids,
                                               past=past,
                                               attention_mask=attention_mask,
                                               token_type_ids=token_type_ids,
                                               position_ids=position_ids,
                                               head_mask=head_mask,
                                               includeprev=includeprev,
                                               x_prev= x_prev)
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        outputs = (lm_logits,) + transformer_outputs[1:]
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1))
            outputs = (loss,) + outputs

        outputs = (hidden_states,) + outputs
        return outputs  # (loss), lm_logits, presents, (all hidden_states), (attentions)


''' Base GPT2 LM Head model
Uses slightly altered GPT2NeighborLMHeadModel and my decoding methods that have nucleus sampling
'''
class GPT2BaseModel(nn.Module):
    ''' base GPT2 model (no memory):
    init params:
    cfg: command line argument settings
    vocab: total vocab size including special tokens
    n_ctx: total context including delimiters
    gen_len: total generation length including end tokens
    includeprev: use the neighboring (previous) paragraph in input
    lastidx: eos index in tokenizer
    use_offline_gpt2: true if we've already downloaded from huggingface to server
    '''
    def __init__(self, cfg, vocab=40990, n_ctx=102, gen_len=401, return_probs=False, includeprev=False, lastidx=0,use_offline_gpt2=False ):
        ###ctx: [<h_prev>/<start> kw<=100 _<i/b/c/t> ] gen<=400 <end> == 503
        #LM mask:[0x101][1x401] 0 - padded
        super(GPT2BaseModel,self).__init__()

        if use_offline_gpt2:
            self.lmmodel = GPT2NeighborLMHeadModel.from_pretrained('./gpt2model', n_ctx=n_ctx+gen_len, n_positions=n_ctx+gen_len)
        elif cfg.debug_mode:
            self.lmmodel = GPT2NeighborLMHeadModel.from_pretrained('gpt2', n_ctx=n_ctx + gen_len,
                                                                n_positions=n_ctx + gen_len)
        else:
            self.lmmodel = GPT2NeighborLMHeadModel.from_pretrained('gpt2-medium', n_ctx=n_ctx + gen_len,
                                                                n_positions=n_ctx + gen_len)
        self.lmmodel.resize_token_embeddings(vocab) 
        self.includeprev = includeprev
        self.n_ctx = n_ctx
        self.gen_len = gen_len
        self.epsilon = 1e-8
        self.lastidx = lastidx
        self.cfg = cfg
        self.repeatfactor = self.cfg.repeattheta
        pos_emb_mask = torch.zeros(1, 1, vocab)  #+n_ctx+gen_len)
        self.register_buffer('pos_emb_mask', pos_emb_mask)


    def _forward(self, x,mask_output,prev, log=False, return_probs=False, returnlast=False, returnnewmem=False, past=None, returnpasts=False):
        lmout = self.lmmodel(x, past=past, attention_mask=mask_output, includeprev=self.includeprev, x_prev=prev)
        h_dec = lmout[0]
        lm_logits = lmout[1]
        presents = lmout[2]
        if returnpasts:
            return lm_logits,presents
        if returnlast:
            lasttoken = torch.where(x[:,:,0] == self.lastidx, torch.ones_like(x[:,:,0]), torch.zeros_like(x[:,:,0])).unsqueeze(-1) #[B,503,1]
            lasttoken = lasttoken.type_as(h_dec)*h_dec   
            hdecmasked = lasttoken.sum(dim=1) #[B,768]
            return lm_logits, hdecmasked
        return lm_logits

    '''
    Forward function:
    Either performs decoding, training step- default is to just do training step
    @param:
    *args: tuple of model inputs
    generate: if True, then generate new tokens using decoding method

    text_encoder: tokenizer
    device: cpu, cuda
    beam, decoding_strategy, log: old params for compatability that are not in use
    k: if using top k sampling
    p: if using nucleus sampling
    gen_len: maximum length for decoding
    min_len: minimum length for decoding
    returnlast: training parameter - return the last token hidden state (this is not in use in the latest codebase)
    '''
    def forward(self, *args, text_encoder=None, device=None, beam=0, gen_len=401, k=0, p=0, decoding_strategy=0, log=False, generate=False, min_len=None, returnlast=False, returnnewmem=False):
        if generate:
            return self.generate(*args,text_encoder=text_encoder, device=device, beam=beam, gen_len=gen_len, k=k, p=p, decoding_strategy=decoding_strategy, min_len=min_len)
        return self._forward(*args, log=log, returnlast=returnlast)

    def sample(self, *args, classify_idx=0, text_encoder=None, gen_len=401, k=0, p=0, decoding_strategy=0, min_len=None, eos_idx=None):
        XMB, mask, prev, seen_unigrams, idxes = args
        pasts = None
        for _ in range(gen_len):
            lm_logits = self._forward(XMB, mask[:, :XMB.size(-1)], prev)
            pem = copy.deepcopy(self.pos_emb_mask)
            if _ < min_len:
            	pem[:,:,eos_idx] = -1e12 #don't let it stop decoding early

            # penalize seen unigrams
            lm_logits[:,-1, :] =  lm_logits[:,-1,:] / seen_unigrams
            lm_probs = F.softmax((lm_logits + pem), dim=-1)
            dist = lm_probs[:, -1, :].squeeze(1)
            if k == 0 and p == 0:
                # Pure Sampling
                next_idx = torch.multinomial(lm_probs[:, -1, :], 1)
            else:
                if p ==0:
                    # Top K Sampling
                    values, indices = dist.topk(k)
                    next_idx = indices.gather(-1, torch.multinomial(values, 1))
                else:
                    # Nucleus Sampling
                    indices = torch.argsort(dist,dim=1,descending=True)
                    values = dist.gather(-1,indices)
                    probsum = torch.cumsum(values,dim=1)
                    include = ~ ((probsum.gt(p*.01)) & ((probsum-values).gt(p*.01)))
                    newdist = torch.where(include, values, torch.zeros_like(values) + 1e-10)
                    next_idx = indices.gather(-1, torch.multinomial(newdist, 1))
            for i in range(XMB.size(0)):
                seen_unigrams[i, next_idx[i]] = self.repeatfactor #add a new seen unigram
            XMB = self.append_batch(XMB, next_idx)
        return XMB[:, -gen_len:], seen_unigrams

    def append_batch(self, X, next_idx):
        return torch.cat((X, next_idx), 1)

    def generate(self, *args, text_encoder=None, device=None, beam=0, gen_len=401, k=0, p=0, decoding_strategy=0, min_len=None):
        ##print(len(args))
        #if len(args) == 5:
        pad_output, mask, prev, seen_trigrams, idxes = args
        #else:
        #    pad_output, mask, prev = args
        #    seen_trigrams = torch.ones(pad_output.size(0), len(text_encoder)).to(pad_output.device)
        #    idxes = None
        classify_idx = None  # don't use this in the code anymore
        eos_idx = text_encoder.eos_token_id
        input_toks = pad_output[:, :self.n_ctx] # includes delimiter
        target_toks = pad_output[:, -gen_len:]
        mask_pad = torch.ones(mask.size()).type_as(mask)
        mask_pad[:, :self.n_ctx] = mask[:, :self.n_ctx]
        mask = mask_pad
        pad_output = pad_output.to(device)
        XMB = pad_output[:, :self.n_ctx]
        if beam == 0:
            generated_toks, seen = self.sample(XMB, mask, prev, seen_trigrams, idxes, classify_idx=classify_idx, text_encoder=text_encoder, gen_len=gen_len, k=k, p=p, decoding_strategy=decoding_strategy, min_len=min_len, eos_idx=eos_idx)
        else:
            raise NotImplementedError
        output = generated_toks.type_as(XMB), input_toks.type_as(XMB), target_toks.type_as(XMB), seen
        return output

#############################################
# PlotMachines classes below:
#############################################

class MemoryAttention(nn.Module):
    '''An Attention Block for attending over the memory slots with word tokens as queries'''
    def __init__(self, nx, n_ctx, config, scale=False):
        super(MemoryAttention, self).__init__()
        self.output_attentions = config.output_attentions

        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        assert n_state % config.n_head == 0
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale
        self.c_attn = Conv1D(n_state, nx)
        self.c_memory = Conv1D(n_state * 2, nx)
        self.c_proj = Conv1D(n_state, nx)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.n_head, self.split_size // self.n_head)
        heads = set(heads) - self.pruned_heads  # Convert to set and emove already pruned heads
        for head in heads:
            # Compute how many pruned heads are before the head and move the index accordingly
            head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        index_attn = torch.cat([index, index + self.split_size, index + (2*self.split_size)])

        # Prune conv1d layers
        self.c_attn = prune_conv1d_layer(self.c_attn, index_attn, dim=1)
        self.c_memory = prune_conv1d_layer(self.c_memory, index_attn, dim=1)
        self.c_proj = prune_conv1d_layer(self.c_proj, index, dim=0)

        # Update hyper params
        self.split_size = (self.split_size // self.n_head) * (self.n_head - len(heads))
        self.n_head = self.n_head - len(heads)
        self.pruned_heads = self.pruned_heads.union(heads)

    def _attn(self, q, k, v, attention_mask=None, head_mask=None, M=None, Mmask=None):
        w = torch.matmul(q, k)  ## w = b,h,s,p
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        nd, ns = w.size(-2), w.size(-1)  

        if M is not None: # if there is indeed memory being used (should always be true)
            #There may be some memory slots might need to be ignored (if a key point list was padded)
            p = M.size(1)
            temp = Mmask.unsqueeze(1).unsqueeze(2).float()  #temp = b,1,1,p 
            attention_mask = (1.0 - temp) * -10000.0 #b,1,1,p
            w = w + attention_mask  #b,h,s,p

        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)
        # Mask heads if we want to
        if head_mask is not None:
            w = w * head_mask

        outputs = [torch.matmul(w, v)]  #b,h,s,p * b,h,p,d
        if self.output_attentions:
            outputs.append(w)
        return outputs

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def forward(self, x, layer_past=None, attention_mask=None, head_mask=None, M=None, Mmask=None):
        x = self.c_attn(x)
        query = x
        key, value = self.c_memory(M).split(self.split_size,dim=2)

        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        if layer_past is not None:
            past_key, past_value = layer_past[0].transpose(-2, -1), layer_past[1]  # transpose back cf below
            key = torch.cat((past_key, key), dim=-1)
            value = torch.cat((past_value, value), dim=-2)
        present = torch.stack((key.transpose(-2, -1), value))  # transpose to have same shapes for stacking

        attn_outputs = self._attn(query, key, value, attention_mask=attention_mask, head_mask=head_mask, M=M, Mmask=Mmask)
        a = attn_outputs[0]

        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)

        outputs = [a, present] + attn_outputs[1:]
        return outputs  # a, present, (attentions)




class  GPT2MemoryBlock(nn.Module):
    def __init__(self, n_ctx, config, scale=False):
        super(GPT2MemoryBlock, self).__init__()
        nx = config.n_embd
        self.ln_1 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.attnextra = MemoryAttention(nx, n_ctx, config, scale)
        self.attn = Attention(nx, n_ctx, config, scale)
        self.ln_2 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.mlp = MLP(4 * nx, config)

    def forward(self, x, layer_past=None, attention_mask=None, head_mask=None, M=None, Mmask=None):
        lnx = self.ln_1(x)
        output_attnR = self.attn(lnx,
                                layer_past=layer_past,
                                attention_mask=attention_mask,
                                head_mask=head_mask)
        aR = output_attnR[0]  # output_attn: a, present, (attentions)
        output_attnL = self.attnextra(lnx,
                                layer_past=layer_past,
                                head_mask=head_mask,
                                M= F.normalize(1e-7 + M, dim=-1), 
                                Mmask=Mmask)
        aL = output_attnL[0]  # output_attn: a, present, (attentions)

        a = (aL + aR) / 2.0
        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m
        outputs = [x] + output_attnR[1:]
        return outputs  # x, present, (attentions)


class GPT2MemModel(GPT2Model):
    def __init__(self, config, use_dual_att=False):
        super(GPT2MemModel, self).__init__(config)
        del self.h
        self.h = nn.ModuleList([GPT2MemoryBlock(config.n_ctx, config, scale=True) for _ in range(config.n_layer)])

    def forward(self, input_ids, past=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, M=None, Mmask=None, includeprev=False, x_prev=None):
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        else:
            past_length = past[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_ids.size(-1) + past_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        # Attention mask.
        if attention_mask is not None:
            attention_mask = attention_mask.view(-1, input_shape[-1])
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0


        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.n_layer, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype) # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.n_layer

        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
        else:
            token_type_embeds = 0


        ####### changed from inherited function: #####
        if includeprev:
            x_prev = x_prev.unsqueeze(1) #[b,1,d] + [d] = [b,1,d]
            ## asli : input_embeds is not even used, commenting it out right now.
            inputs_embeds = torch.cat([x_prev, inputs_embeds[:,1:,:]], dim = 1)  #x_prev: [b, 1, d], h : [b, s, d]-->[B, s+1, D]
        ########### END HERE #########################

        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        presents = ()
        all_attentions = []
        all_hidden_states = ()
        for i, (block, layer_past) in enumerate(zip(self.h, past)):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states.view(*output_shape),)

            outputs = block(hidden_states,
                            layer_past=None,
                            attention_mask=attention_mask,
                            head_mask=head_mask[i],
                            M= M,  #changed from inherited function
                            Mmask=Mmask) #changed from inherited function

            hidden_states, present = outputs[:2]
            if self.output_past:
                presents = presents + (present,)

            if self.output_attentions:
                all_attentions.append(outputs[2])

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(*output_shape)
        # Add last hidden state
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_past:
            outputs = outputs + (presents,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            # let the number of heads free (-1) so we can extract attention even after head pruning
            attention_output_shape = input_shape[:-1] + (-1,) + all_attentions[0].shape[-2:]
            all_attentions = tuple(t.view(*attention_output_shape) for t in all_attentions)
            outputs = outputs + (all_attentions,)
        return outputs  # last hidden state, (presents), (all hidden_states), (attentions)

class GPT2MemLMHeadModel(GPT2LMHeadModel):

    def __init__(self, config):
        super(GPT2MemLMHeadModel, self).__init__(config)
        self.transformer = GPT2MemModel(config)
        self.init_weights()
        self.tie_weights()

    def forward(self, input_ids, past=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                labels=None, M=None, Mmask=None, includeprev=False, x_prev=None):
        transformer_outputs = self.transformer(input_ids,
                                               past=past,
                                               attention_mask=attention_mask,
                                               token_type_ids=token_type_ids,
                                               position_ids=position_ids,
                                               head_mask=head_mask,
                                               M = M,
                                               Mmask = Mmask, includeprev=includeprev, x_prev=x_prev)
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        outputs = (lm_logits,) + transformer_outputs[1:]
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1))
            outputs = (loss,) + outputs

        outputs = (hidden_states,) + outputs
        return outputs  # (loss), hidden_states, lm_logits, presents, (all hidden_states), (attentions)



class GatedMemoryUpdate(nn.Module):
    """ Transformer model """
    def __init__(self, cfg, n_ctx):
        super(GatedMemoryUpdate, self).__init__()

        self.W1 = torch.nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.W2 = torch.nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.W3 = torch.nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.W4 = torch.nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)

    def forward(self, Y, Mi, Ymask, Mmask):
        #Y=[B,1xd] Ymask=[B,1x1] Mi=[B,mxd] Mmask=[B,mx1] 
        Mhat = torch.tanh(self.W1(Y.expand(-1, Mi.size(1), -1))+self.W2(Mi))
        g = torch.sigmoid(self.W3(Y.expand(-1, Mi.size(1), -1))+self.W4(Mi))
        Mnext = torch.mul(1-g, Mi) + torch.mul(g, Mhat)
        return F.normalize(1e-7+Mnext, dim = -1)


class PlotMachinesModel(nn.Module):
    ''' full PlotMachines model:
    init params:
    cfg: command line argument settings
    vocab: total vocab size including special tokens
    n_ctx: total context including delimiters
    gen_len: total generation length including end tokens
    includeprev: use the neighboring (previous) paragraph in input
    lastidx: eos index in tokenizer
    use_offline_gpt2: true if we've already downloaded from huggingface to server
    '''
    def __init__(self, cfg, vocab=40990, n_ctx=102, gen_len=401, return_probs=False, includeprev=False, lastidx=0,  use_offline_gpt2=False):
        ###ctx: [<h_prev>/<start> kw<=100 _<i/b/c/t> ] gen<=400 <end> == 503
        #LM mask:[0x101][1x401] 0 - padded
        super(PlotMachinesModel,self).__init__()
        self.n_ctx = n_ctx
        self.gen_len = gen_len
        self.lastidx = lastidx

        self.memupd = GatedMemoryUpdate(cfg, n_ctx-2+cfg.memstatesize)
        if use_offline_gpt2:
            self.lmmodel = GPT2MemLMHeadModel.from_pretrained('./gpt2model', n_positions=n_ctx + gen_len)
        elif cfg.debug_mode:
            self.lmmodel = GPT2MemLMHeadModel.from_pretrained('gpt2', n_positions=n_ctx + gen_len)
        else:
            self.lmmodel = GPT2MemLMHeadModel.from_pretrained('gpt2-medium', n_positions=n_ctx + gen_len)

        self.lmmodel.resize_token_embeddings(vocab)
        self.epsilon = 1e-8
        self.cfg = cfg
        pos_emb_mask = torch.zeros(1, 1, vocab)  #+n_ctx+gen_len)
        self.includeprev = includeprev
        self.repeatfactor = cfg.repeattheta
        self.register_buffer('pos_emb_mask', pos_emb_mask)


    ''' Training step
    *args are expected to be in this format:
    x: [B, S] - batch of paragraphs encoded as token ids
    mask_output: [B, S] - masks over padding
    mem: [B, Memsize, D] - the initial memory
    mmask: [B, Memsize] - masks over any padded memory cells
    prev: [B, 10, D] - up to 10 previous paragraph encodings with which to update the memory
    pmask: [B, 10] - mask over previous paragraphs that aren't there
    pvect: [B, D] - previous paragraph encoding to use as neighboring input vector
    '''
    def _forward(self, *args, log=False, return_probs=False, returnnewmem=False, returnlast=False, past=None, returnpasts=False):

        x, mask_output, mem, mmask, prev, pmask, pvect = args

        n_ctx = self.n_ctx
        #print(mem)
        if prev is not None:
            mem,mmask= self.updatememory(x,mem,mmask,prev,pmask)

        lmout = self.lmmodel(x, past=past, attention_mask=mask_output, M=mem, Mmask=mmask, includeprev=self.includeprev, x_prev=pvect)
        h_dec = lmout[0]
        lm_logits = lmout[1]
        presents = lmout[2]
        if returnpasts:
            return lm_logits,presents
        if returnlast:
            lasttoken = torch.where(x[:,:] == self.lastidx, torch.ones_like(x[:,:]), torch.zeros_like(x[:,:])).unsqueeze(-1) #[B,503,1]
            lasttoken = lasttoken.type_as(h_dec)*h_dec   
            hdecmasked = lasttoken.sum(dim=1) #[B,768]
            return lm_logits, hdecmasked
        return lm_logits

    def updatememory(self, *args):
        x, mem, mmask, prev, pmask = args  #xraw = [B,T]
        mem[:,: self.n_ctx-2, :]  = self.lmmodel.transformer.wte(x[:,1:self.n_ctx-1])
        if prev is not None:
            for p in range(prev.size(1)):
                U = prev[:,p,:]
                Umask = pmask[:,p,:]#.squeeze(1)
                update = (Umask.sum(dim=-1) != 0).view(-1,1)
                oldmem = mem
                if update.any() > 0:
                   mem = (1-update.view(-1,1,1).float())* mem + (update.view(-1,1,1).float()) * self.memupd(U, mem, Umask, mmask)
        return mem, mmask


    '''
    Forward function:
    Either performs decoding, training step, or updates the memory depending on parameters, default is to just do training step
    @param:
    *args: tuple of model inputs
    returnnewmem: if True, then update the memory
    generate: if True, then generate new tokens using decoding method

    text_encoder: tokenizer
    device: cpu, cuda
    beam, decoding_strategy, log: old params for compatability that are not in use
    k: if using top k sampling
    p: if using nucleus sampling
    gen_len: maximum length for decoding
    min_len: minimum length for decoding
    returnlast: training parameter - return the last token hidden state (this is not in use in the latest codebase)
    '''
    def forward(self, *args, text_encoder=None, device=None, beam=0, gen_len=401, k=0, p=0, decoding_strategy=0, log=False, generate=False, min_len=None, returnlast=False, returnnewmem=False):
        if returnnewmem:
            return self.updatememory(*args)
        elif generate:
            return self.generate(*args, text_encoder=text_encoder, device=device, beam=beam, gen_len=gen_len, k=k, p=p, decoding_strategy=decoding_strategy, min_len=min_len, returnnewmem=returnnewmem)
        return self._forward(*args, log=log, returnlast=returnlast)

    def sample(self, *args, classify_idx=None, text_encoder=None, gen_len=401, k=0, p=0, decoding_strategy=0, min_len=None, eos_idx=None, returnnewmem = False):
        XMB, mask, mem, mmask, prev, pmask,pvect, seen_unigrams, idxes = args
        mem,mmask = self.updatememory(XMB, mem, mmask, prev, pmask)

        pasts = None
        for _ in range(gen_len):
             
            fargs =(XMB, mask[:, :XMB.size(-1)], mem, mmask, None, None, pvect)
            lm_logits = self._forward(*fargs) # past=pasts, returnpasts=True)
            lm_logits[:,-1, :] =  lm_logits[:,-1,:] / seen_unigrams
            pem = copy.deepcopy(self.pos_emb_mask)
            if _ < min_len:
                pem[:,:,eos_idx] = -1e12


            lm_probs = F.softmax((lm_logits + pem), dim=-1)
            dist = lm_probs[:, -1, :].squeeze(1)
            if k == 0 and p == 0:
                next_idx = torch.multinomial(lm_probs[:, -1, :], 1)
            else:
                if p ==0:
                    # Sample from top k

                    values, indices = dist.topk(k)
                    next_idx = indices.gather(-1, torch.multinomial(values, 1))
                else:
                    indices = torch.argsort(dist,dim=1,descending=True)
                    values = dist.gather(-1,indices)
                    probsum = torch.cumsum(values,dim=1)
                    include = ~ ((probsum.gt(p*.01)) & ((probsum-values).gt(p*.01)))
                    newdist = torch.where(include, values, torch.zeros_like(values) + 1e-10)
                    next_idx = indices.gather(-1, torch.multinomial(newdist, 1))
            for i in range(XMB.size(0)):
                seen_unigrams[i, next_idx[i]] = self.repeatfactor 
            XMB = self.append_batch(XMB, next_idx)
        return XMB[:, -gen_len:], seen_unigrams



    '''  Generate:
    *args are expected to be in this format:

    pad_output: [B, S] - batch of plot outline contexts encoded as token ids
    mask: [B, S] - masks over padding in outline contexts
    mem: [B, Memsize, D] - the initial memory
    mmask: [B, Memsize] - masks over any padded memory cells
    prev: [B, 10, D] - up to 10 previous paragraph encodings with which to update the memory
    pmask: [B, 10] - mask over previous paragraphs that aren't there
    xprev: [B, D] - previous paragraph encoding to use as neighboring input vector
    seen_unigrams [B, V] - previously used tokens in previous paragraphs
    idxes: [B] - the doc ids
    note: S= ctx + gen_len even though the generation will be blank tokens before decoding
    '''
    def generate(self, *args, text_encoder=None, device=None, beam=0, gen_len=401, k=0, p=0, decoding_strategy=0, min_len=None, returnnewmem=False):
        
        if len(args) == 9:
            pad_output, mask, mem, mmask, prev, pmask, xprev, seen_unigrams, idxes = args
        else:
            pad_output, mask, mem, mmask, prev, pmask, xprev = args
            seen_unigrams = torch.ones(pad_output.size(0), len(text_encoder)).to(pad_output.device)
            idxes = None
        classify_idx = None #not in use by generation code anymore
        eos_idx = text_encoder.eos_token_id
        input_toks = pad_output[:, :self.n_ctx] # includes delimiter
        target_toks = pad_output[:, -gen_len:]
        mask_pad = torch.ones(mask.size()).type_as(mask)
        mask_pad[:, :self.n_ctx] = mask[:, :self.n_ctx]
        mask = mask_pad
        pad_output = pad_output.to(device)
        XMB = pad_output[:, :self.n_ctx]
        if beam == 0:
            generated_toks, seen_unigrams = self.sample(XMB, mask, mem, mmask, prev, pmask, xprev, seen_unigrams, idxes, classify_idx=classify_idx, text_encoder=text_encoder, gen_len=gen_len, k=k, p=p, decoding_strategy=decoding_strategy, min_len=min_len, eos_idx=eos_idx, returnnewmem=returnnewmem)
            return generated_toks.type_as(XMB), input_toks.type_as(XMB), target_toks.type_as(XMB), seen_unigrams
        else:
            raise NotImplementedError
       


    def append_batch(self, X, next_idx):
        return torch.cat((X, next_idx), 1)
 
