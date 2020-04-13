import argparse
import csv
import glob
import json
import os
import random
import re

from nltk.tokenize import sent_tokenize
import numpy as np
import torch
import torch.nn as nn
import rouge
from transformers import *

from tqdm import tqdm
from generate_utils  import generate_paragraph

def clear_dirs(gen_dir, tgt_dir):
    for f in glob.glob("{}/*".format(tgt_dir)):
        os.remove(f)
    for f in glob.glob("{}/*".format(gen_dir)):
        os.remove(f)
    os.makedirs(tgt_dir, exist_ok=True)
    os.makedirs(gen_dir, exist_ok=True)

def format_text(text, max_len, stop_words=[]):
    text = "\n".join(sent_tokenize(text)).replace("<", "&lt").replace(">", "&gt")
    for stop_word in stop_words:
        text = text.replace(" {} ".format(stop_word), " ")
    if max_len is not None:
        text = " ".join(text.split(" ")[:max_len])
    return text.encode('ascii','ignore').decode("ascii","ignore")

def get_average_scores(jsonfile, srcs,hyps, refs,maxlen=110, stop_words=[]):
    rouge_scorer = rouge.Rouge()
    averaged_scores = {'rouge-1': {'f': 0, 'p': 0, 'r': 0},
                       'rouge-2': {'f': 0, 'p': 0, 'r': 0},
                       'rouge-l': {'f': 0, 'p': 0, 'r': 0}}

    scores = rouge_scorer.get_scores(hyps, refs)
    for metric in averaged_scores.keys():
        for values in scores:
            for sub_metric in averaged_scores[metric]:
                averaged_scores[metric][sub_metric] += values[metric][sub_metric]
    for key in averaged_scores.keys():
        for sub_key in averaged_scores[key].keys():
            averaged_scores[key][sub_key] /= len(hyps)
    for i in range(len(srcs)):
        jsonfile.write(json.dumps({'r1': scores[i]['rouge-1'], 'r2': scores[i]['rouge-2'], 'rl': scores[i]['rouge-l'],'hyp':hyps[i], 'ref':refs[i],'src':srcs[i]})+"\n")
    return averaged_scores




def evaluate_doc_model(model, val_loader, text_encoder, device, beam, gen_len, k, p, decoding_strategy, save_file, gen_dir="gen", tgt_dir="tgt", max_len=110, stop_words=[], args=None):
    data = {"src": [], "gen": [], "tgt": []}
    srcs, hyps, refs = [], [], []
    model.eval()
    for batchargs in tqdm(val_loader):
        with torch.no_grad():
            # Generating outputs for evaluation
            src_strs, tgt_strs, gen_strs = generate_paragraph(model, batchargs, text_encoder, device, beam, gen_len, k, p, decoding_strategy, min_len=args.min_len)
            data["src"].extend(src_strs)
            data["gen"].extend(gen_strs)
            data["tgt"].extend(tgt_strs)
            
    jsf = open(save_file+".output.json","w")
    for i in range(min(len(data['src']),50)):
        print("*" * 50)
        try:
            print("Source: {}".format(data['src'][i]))
            print('Hypothesis: {}'.format(data['gen'][i]))
            print("Reference: {}".format(data['tgt'][i]))
        except:
            pass

    with open(save_file, "w") as f:
        json.dump(
            #get_rouge_scores(gen_dir, tgt_dir),
            get_average_scores(jsf,data['src'],data['gen'],data['tgt'],max_len,stop_words),
            f,
            indent=4,
            sort_keys=True
        )



