from model_pytorch import DecoderSelfAttention
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy

from transformers.modeling_gpt2 import *

class GPT2WPrevModel(GPT2Model):
    def __init__(self, config):
        super(GPT2WPrevModel, self).__init__(config)

    def forward(self, input_ids, past=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, labels=None, includeprev=False, x_prev=None):
        if includeprev:
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

            ####### THIS IS THE PART that needs to be changed from inherited function, really:
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


class GPT2WPrevLMHeadModel(GPT2LMHeadModel):

    def __init__(self, config):
        super(GPT2WPrevLMHeadModel, self).__init__(config)
        self.transformer = GPT2WPrevModel(config)
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


class DocumentDecoderModel(nn.Module):
    def __init__(self, cfg, vocab=40990, n_ctx=102, gen_len=401, return_probs=False, includeprev=False, lastidx=0,use_offline_gpt2=False ):
        ###ctx: [<h_prev>/<start> kw<=100 _<i/b/c/t> ] gen<=400 <end> == 503
        #LM mask:[0x101][1x401] 0 - padded
        super(DocumentDecoderModel,self).__init__()

        if use_offline_gpt2:
            self.lmmodel = GPT2WPrevLMHeadModel.from_pretrained('./gpt2model', n_ctx=n_ctx+gen_len, n_positions=n_ctx+gen_len)
        elif cfg.debug_mode:
            self.lmmodel = GPT2WPrevLMHeadModel.from_pretrained('gpt2', n_ctx=n_ctx + gen_len,
                                                                n_positions=n_ctx + gen_len)

        else:
            self.lmmodel = GPT2WPrevLMHeadModel.from_pretrained('gpt2-medium', n_ctx=n_ctx + gen_len,
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


    # def _forward(self, *args, log=False, return_probs=False, returnlast=False, returnnewmem=False, past=None, returnpasts=False):
    def _forward(self, x,mask_output,prev,curr, log=False, return_probs=False, returnlast=False, returnnewmem=False, past=None, returnpasts=False):
        # (x,mask_output,prev,curr) = args
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

    def forward(self, *args, text_encoder=None, device=None, beam=0, gen_len=401, k=0, p=0, decoding_strategy=0, log=False, generate=False, min_len=None, returnlast=False, returnnewmem=False):
        if generate:
            return self.generate(*args,text_encoder=text_encoder, device=device, beam=beam, gen_len=gen_len, k=k, p=p, decoding_strategy=decoding_strategy, min_len=min_len)
        return self._forward(*args, log=log, returnlast=returnlast)

    def sample(self, *args, classify_idx=0, text_encoder=None, gen_len=401, k=0, p=0, decoding_strategy=0, min_len=None, eos_idx=None):
        XMB, mask, prev, curr, seen_unigrams, idxes = args
        pasts = None
        for _ in range(gen_len):
            # fargs =(XMB, mask, prev, curr)
            # lm_logits, pasts = self._forward(fargs, past=pasts, returnpasts=True)
            lm_logits = self._forward(XMB, mask[:, :XMB.size(-1)], prev, curr)
            pem = copy.deepcopy(self.pos_emb_mask)
            if _ < min_len:
            	pem[:,:,eos_idx] = -1e12
            lm_logits[:,-1, :] =  lm_logits[:,-1,:] / seen_unigrams

            lm_probs = F.softmax((lm_logits + pem), dim=-1)
            dist = lm_probs[:, -1, :].squeeze(1)
            if k == 0 and p == 0:
                next_idx = torch.multinomial(lm_probs[:, -1, :], 1)
            else:
                if p ==0:
                    # Sample from top k

                    values, indices = dist.topk(k)
                    next_idx = indices.gather(-1, torch.multinomial(values, 1))
                    #for i in range(XMB.size(0)):
                    #    seen_unigrams[i][next_idx[i].item()] = 1  #.append(next_idx[i].item())
                else:
                    indices = torch.argsort(dist,dim=1,descending=True)
                    values = dist.gather(-1,indices)
                    probsum = torch.cumsum(values,dim=1)
                    include = ~ ((probsum.gt(p*.01)) & ((probsum-values).gt(p*.01)))
                    newdist = torch.where(include, values, torch.zeros_like(values) + 1e-10)
                    next_idx = indices.gather(-1, torch.multinomial(newdist, 1))
            #######seen_unigrams[:, next_idx] = self.repeatfactor
            for i in range(XMB.size(0)):
                #print(seen_unigrams.size(), next_idx[i])
                seen_unigrams[i, next_idx[i]] = self.repeatfactor  #.append(next_idx[i].item())
                #else:
                #    raise NotImplementedError
            XMB = self.append_batch(XMB, next_idx)
        return XMB[:, -gen_len:], seen_unigrams

    def append_batch(self, X, next_idx):
        return torch.cat((X, next_idx), 1)

    def generate(self, *args, text_encoder=None, device=None, beam=0, gen_len=401, k=0, p=0, decoding_strategy=0, min_len=None):
        ##print(len(args))
        if len(args) == 6:
            pad_output, mask, prev, curr, seen_trigrams, idxes = args
        else:
            pad_output, mask, prev, curr = args
            seen_trigrams = torch.ones(pad_output.size(0), len(text_encoder)).to(pad_output.device)
            idxes = None
        # classify_idx = text_encoder.added_tokens_encoder['_classify_'] # text_encoder.encoder['_classify_']
        classify_idx = None  # text_encoder._convert_token_to_id()
        eos_idx = text_encoder.eos_token_id
        # eos_idx = text_encoder.added_tokens_encoder['_end_'] #text_encoder.encoder['_end_']
        input_toks = pad_output[:, :self.n_ctx] # includes delimiter
        target_toks = pad_output[:, -gen_len:]
        mask_pad = torch.ones(mask.size()).type_as(mask)
        mask_pad[:, :self.n_ctx] = mask[:, :self.n_ctx]
        mask = mask_pad
        pad_output = pad_output.to(device)
        XMB = pad_output[:, :self.n_ctx]
        if beam == 0:
            generated_toks, seen = self.sample(XMB, mask, prev, curr, seen_trigrams, idxes, classify_idx=classify_idx, text_encoder=text_encoder, gen_len=gen_len, k=k, p=p, decoding_strategy=decoding_strategy, min_len=min_len, eos_idx=eos_idx)
        else:
            raise NotImplementedError
        output = generated_toks.type_as(XMB), input_toks.type_as(XMB), target_toks.type_as(XMB), seen
        return output

###############################################################################################################


class ExtraAttention(nn.Module):
    def __init__(self, nx, n_ctx, config, scale=False):
        super(ExtraAttention, self).__init__()
        self.output_attentions = config.output_attentions

        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        assert n_state % config.n_head == 0
        self.register_buffer("bias", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale

        self.c_attn = Conv1D(n_state * 3, nx)
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
        # M shape = bxpxd
        # Mmask = bxp
        # q = bxhxsxd
        # k = bxhxdxs+1
        # v = bxhxs+1xd
        #attention_mask = b,1,1,s
        w = torch.matmul(q, k) # w == b,h,s,(s+1)

        if self.scale:
            w = w / math.sqrt(v.size(-1))
        nd, ns = w.size(-2), w.size(-1)  #nd = s, ns = s+1
        b = self.bias[:, :, :nd, :nd]  # b = [1,1,s,s]

        #print(attention_mask.size())
        if M is not None:
            p = M.size(1)
            b = torch.cat((torch.ones(b.size(0),b.size(1),b.size(2), p).cuda(),b),dim=3)  # b = [1,1,s,s+1]'''
        #print(w.size(), b.size())
        w = w * b - 1e4 * (1 - b)

        if attention_mask is not None:
            # Apply the attention mask
            if M is not None:
                p = M.size(1)
                temp = Mmask.unsqueeze(1).unsqueeze(2).float()  #temp = b,1,1,p #torch.zeroes(attention_mask.size(0),attention_mask.size(1), attention_mask.size(2), p).cuda()
                temp = (1.0 - temp) * -10000.0
                attention_mask = torch.cat((temp, attention_mask), dim=3) #b,1,1,s+1

            w = w + attention_mask

        w = nn.Softmax(dim=-1)(w)


        w = self.attn_dropout(w)

        # Mask heads if we want to
        if head_mask is not None:
            w = w * head_mask

        outputs = [torch.matmul(w, v)]
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
        query, key, value = x.split(self.split_size, dim=2)

        if M is not None:
            ek,ev = self.c_memory(M).split(self.split_size,dim=2)
            key = torch.cat((ek,key), dim=1)
            value = torch.cat((ev,value), dim=1)

        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        if layer_past is not None:
            past_key, past_value = layer_past[0].transpose(-2, -1), layer_past[1]  # transpose back cf below
            key = torch.cat((past_key, key), dim=-1)
            value = torch.cat((past_value, value), dim=-2)
        present = torch.stack((key.transpose(-2, -1), value))  # transpose to have same shapes for stacking

        attn_outputs = self._attn(query, key, value, attention_mask, head_mask, M=M, Mmask=Mmask)
        a = attn_outputs[0]

        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)

        outputs = [a, present] + attn_outputs[1:]
        return outputs  # a, present, (attentions)



class ExtraNotSelfAttention(nn.Module):
    def __init__(self, nx, n_ctx, config, scale=False):
        super(ExtraNotSelfAttention, self).__init__()
        self.output_attentions = config.output_attentions

        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        assert n_state % config.n_head == 0
        ###self.register_buffer("bias", torch.ones(n_ctx, n_ctx).view(1, 1, n_ctx, n_ctx))
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

        if M is not None:
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



class  GPT2ExtraBlock(nn.Module):
    def __init__(self, n_ctx, config, scale=False):
        super(GPT2ExtraBlock, self).__init__()
        nx = config.n_embd
        self.ln_1 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.attn = ExtraAttention(nx, n_ctx, config, scale)
        self.ln_2 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.mlp = MLP(4 * nx, config)

    def forward(self, x, layer_past=None, attention_mask=None, head_mask=None, M=None, Mmask=None):
        output_attn = self.attn(self.ln_1(x),
                                layer_past=layer_past,
                                attention_mask=attention_mask,
                                head_mask=head_mask,
                                M= F.normalize(1e-7 + M, dim=-1), 
                                Mmask=Mmask)
        a = output_attn[0]  # output_attn: a, present, (attentions)

        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m

        outputs = [x] + output_attn[1:]
        return outputs  # x, present, (attentions)


class  GPT2DualBlock(nn.Module):
    def __init__(self, n_ctx, config, scale=False):
        super(GPT2DualBlock, self).__init__()
        nx = config.n_embd
        self.ln_1 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.attnextra = ExtraNotSelfAttention(nx, n_ctx, config, scale)
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


class GPT2WMemModel(GPT2Model):
    def __init__(self, config, use_dual_att=False):
        super(GPT2WMemModel, self).__init__(config)
        del self.h
        if use_dual_att:
            print("Using Dual Attentions...")
            self.h = nn.ModuleList([GPT2DualBlock(config.n_ctx, config, scale=True) for _ in range(config.n_layer)])
        else:
            self.h = nn.ModuleList([GPT2ExtraBlock(config.n_ctx, config, scale=True) for _ in range(config.n_layer)])

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


        ####### THIS IS THE PART that needs to be changed from inherited function, really:
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
                            M= M,
                            Mmask=Mmask)

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

class GPT2WMemLMHeadModel(GPT2LMHeadModel):

    def __init__(self, config, use_dual_att=False):
        super(GPT2WMemLMHeadModel, self).__init__(config)
        self.transformer = GPT2WMemModel(config, use_dual_att=use_dual_att)
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
        return outputs  # (loss), lm_logits, presents, (all hidden_states), (attentions)



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


class DocumentMemoryDecoderModel(nn.Module):
    def __init__(self, cfg, vocab=40990, n_ctx=102, gen_len=401, return_probs=False, includeprev=False, lastidx=0,  use_offline_gpt2=False):
        ###ctx: [<h_prev>/<start> kw<=100 _<i/b/c/t> ] gen<=400 <end> == 503
        #LM mask:[0x101][1x401] 0 - padded
        super(DocumentMemoryDecoderModel,self).__init__()
        self.n_ctx = n_ctx
        self.gen_len = gen_len
        self.lastidx = lastidx

        self.memupd = GatedMemoryUpdate(cfg, n_ctx-2+cfg.memstatesize)
        if use_offline_gpt2:
            self.lmmodel = GPT2WMemLMHeadModel.from_pretrained('./gpt2model', n_positions=n_ctx + gen_len,
                                                               use_dual_att=cfg.use_dual_att)
        elif cfg.debug_mode:
            self.lmmodel = GPT2WMemLMHeadModel.from_pretrained('gpt2', 
                                                                n_positions=n_ctx + gen_len,
                                                                use_dual_att=cfg.use_dual_att)
        else:
            self.lmmodel = GPT2WMemLMHeadModel.from_pretrained('gpt2-medium', n_positions=n_ctx + gen_len,
                                                                       use_dual_att=cfg.use_dual_att)

        self.lmmodel.resize_token_embeddings(vocab)
        self.epsilon = 1e-8
        self.cfg = cfg
        pos_emb_mask = torch.zeros(1, 1, vocab)  #+n_ctx+gen_len)
        self.includeprev = includeprev
        self.repeatfactor = cfg.repeattheta
        self.register_buffer('pos_emb_mask', pos_emb_mask)


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
        #print(len(args))
        x, mem, mmask, prev, pmask = args  #xraw = [B,T]
        if self.cfg.use_kwmem:
            mem[:,: self.n_ctx-2, :]  = self.lmmodel.transformer.wte(x[:,1:self.n_ctx-1])
        #print(mem)
        if prev is not None:
            for p in range(prev.size(1)):
                U = prev[:,p,:]
                Umask = pmask[:,p,:]#.squeeze(1)
                update = (Umask.sum(dim=-1) != 0).view(-1,1)
                oldmem = mem
                if update.any() > 0:
                   #print(mem.size(), update.size(), Umask.size())
                   #print(update, Umask)
                   mem = (1-update.view(-1,1,1).float())* mem + (update.view(-1,1,1).float()) * self.memupd(U, mem, Umask, mmask)
                   #print(mem-oldmem)
        return mem, mmask

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
                    #for i in range(XMB.size(0)):
                    #    seen_unigrams[i][next_idx[i].item()] = 1  #.append(next_idx[i].item())
                else:
                    indices = torch.argsort(dist,dim=1,descending=True)
                    values = dist.gather(-1,indices)
                    probsum = torch.cumsum(values,dim=1)
                    include = ~ ((probsum.gt(p*.01)) & ((probsum-values).gt(p*.01)))
                    newdist = torch.where(include, values, torch.zeros_like(values) + 1e-10)
                    next_idx = indices.gather(-1, torch.multinomial(newdist, 1))
            #######seen_unigrams[:, next_idx] = self.repeatfactor
            for i in range(XMB.size(0)):
                #print(seen_unigrams.size(), next_idx[i])
                seen_unigrams[i, next_idx[i]] = self.repeatfactor  #.append(next_idx[i].item())
                #else:
                #    raise NotImplementedError
            XMB = self.append_batch(XMB, next_idx)
        return XMB[:, -gen_len:], seen_unigrams


    def generate(self, *args, text_encoder=None, device=None, beam=0, gen_len=401, k=0, p=0, decoding_strategy=0, min_len=None, returnnewmem=False):
        
        if len(args) == 9:
            pad_output, mask, mem, mmask, prev, pmask, xprev, seen_unigrams, idxes = args
        else:
            pad_output, mask, mem, mmask, prev, pmask, xprev = args
            seen_unigrams = torch.ones(pad_output.size(0), len(text_encoder)).to(pad_output.device)
            idxes = None
        classify_idx = None#text_encoder._convert_token_to_id()
        eos_idx = text_encoder.eos_token_id
        input_toks = pad_output[:, :self.n_ctx] # includes delimiter
        target_toks = pad_output[:, -gen_len:]
        mask_pad = torch.ones(mask.size()).type_as(mask)
        mask_pad[:, :self.n_ctx] = mask[:, :self.n_ctx]
        mask = mask_pad
        pad_output = pad_output.to(device)
        XMB = pad_output[:, :self.n_ctx]
        #print(XMB.size())
        if beam == 0:
            generated_toks, seen_unigrams = self.sample(XMB, mask, mem, mmask, prev, pmask, xprev, seen_unigrams, idxes, classify_idx=classify_idx, text_encoder=text_encoder, gen_len=gen_len, k=k, p=p, decoding_strategy=decoding_strategy, min_len=min_len, eos_idx=eos_idx, returnnewmem=returnnewmem)
            return generated_toks.type_as(XMB), input_toks.type_as(XMB), target_toks.type_as(XMB), seen_unigrams
        else:
            raise NotImplementedError
       


    def append_batch(self, X, next_idx):
        return torch.cat((X, next_idx), 1)
 
