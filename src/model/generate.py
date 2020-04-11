import argparse
import os
import random

import numpy as np
import torch
from tqdm import tqdm
from pytorch_transformers import *


def generate_paragraph(model, args, text_encoder, device, beam, gen_len, k, p, decoding_strategy, min_len=None):
    src_strs, tgt_strs, gen_strs = [], [], []
    mask = args[1]    
    n_gpu = torch.cuda.device_count()
    outputs = model(*args, text_encoder=text_encoder, device=device, beam=beam, gen_len=gen_len, k=k, p=p, decoding_strategy=decoding_strategy, generate=True, min_len=min_len)
    #print(len(outputs[0]))
   # for i in range(len(outputs[0])):
    if n_gpu == 1:
        outputs = [outputs]
    for generated_toks, input_toks, target_toks, _  in outputs: ##outputs[0][i],outputs[1][i],outputs[2][i] 
        for idx in range(generated_toks.size(0)):
                src_str = toks_to_str(input_toks[idx], text_encoder, is_input=True, mask=mask[idx], ctx=102)
                src_strs.append(src_str)
                tgt_str = toks_to_str(target_toks[idx], text_encoder)
                tgt_strs.append(tgt_str)
                gen_str = toks_to_str(generated_toks[idx], text_encoder)
                gen_strs.append(gen_str)
    return src_strs, tgt_strs, gen_strs



def toks_to_str(toks, text_encoder, is_input=False, mask=None, ctx=102):
    str_rep = []
    end_tok = text_encoder.convert_tokens_to_ids('_endkw_') if is_input else text_encoder.convert_tokens_to_ids('_end_')
    
    for token in toks:
        if token.item() == end_tok : #or token.item() == 0:# or x.item() == end_idx:
            break        
        str_rep.append( text_encoder.convert_ids_to_tokens(token.item()))

    if is_input:
        str_rep.append(text_encoder.convert_ids_to_tokens(toks[ctx-1].item()))
    str_rep = text_encoder.convert_tokens_to_string(str_rep)

    # This makes sure rouge scorers doesn't complain about no sentences
    if not str_rep:
        str_rep = "unk."
    elif "." not in str_rep:
        str_rep += "."

    return str_rep
