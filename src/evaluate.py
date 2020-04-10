import argparse
import csv
import glob
import json
import os
import random
import re

from nltk.tokenize import sent_tokenize
import numpy as np
from pyrouge import Rouge155
import torch
import torch.nn as nn
import rouge
from transformers import *

from tqdm import tqdm
from generate  import generate_paragraph
from data_loader import get_paragraph_input_loader, get_paragraph_history_input_loader
from decodermodules import DocumentDecoderModel, DocumentMemoryDecoderModel

from loss import ParagraphLoss
from parallel import DataParallelModel, DataParallelCriterion

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


def evaluate_thesis_model(model, val_loader, text_encoder, device, beam, gen_len, k, p, decoding_strategy, save_file, gen_dir="gen", tgt_dir="tgt", max_len=110, stop_words=[], args=None):
    data = {"src": [], "gen": [], "tgt": []}
    srcs, hyps, refs = [], [], []
    model.eval()
    for pad_seq, mask_seq, kw, kwmask in tqdm(val_loader):
        with torch.no_grad():
            # Generating outputs for evaluation
            src_strs, tgt_strs, gen_strs = generate_outputs_from_kw(model, pad_seq, mask_seq, kw, kwmask, text_encoder, device, beam, gen_len, k, p, decoding_strategy, min_len=gen_len)
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


def evaluate_model(model, val_loader, text_encoder, device, beam, gen_len, k, p, decoding_strategy, save_file, gen_dir="gen", tgt_dir="tgt", max_len=110, stop_words=[], args=None):
    data = {"src": [], "gen": [], "tgt": []}
    srcs, hyps, refs = [], [], []

    model.eval()
    for pad_seq, mask_seq, plans,other in tqdm(val_loader):
        with torch.no_grad():
            # Generating outputs for evaluation
            src_strs, tgt_strs, gen_strs = generate_outputs(model, pad_seq, mask_seq, plans, text_encoder, device, beam, gen_len, k, p, decoding_strategy, min_len=args.min_len)
            data["src"].extend(src_strs)
            data["gen"].extend(gen_strs)
            data["tgt"].extend(tgt_strs)

    '''for i in range(len(data["src"])):
        with open(os.path.join(gen_dir, "gen.{}.txt".format(i)), "w") as gen_file:
            gen_file.write(format_text(data["gen"][i], max_len, stop_words))
        with open(os.path.join(tgt_dir, "tgt.{}.txt".format(i)), "w") as tgt_file:
            tgt_file.write(format_text(data["tgt"][i], max_len, stop_words))'''
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

def get_rouge_scores(gen_dir, tgt_dir, gen_pattern='gen.(\d+).txt', tgt_pattern='tgt.#ID#.txt'):
    r = Rouge155()
    r.system_dir = gen_dir
    r.model_dir = tgt_dir
    r.system_filename_pattern = gen_pattern
    r.model_filename_pattern = tgt_pattern
    output = r.convert_and_evaluate()
    return r.output_to_dict(output)

def get_average_scores(jsonfile, srcs,hyps, refs,maxlen=110, stop_words=[]):
       
    rouge_scorer = rouge.Rouge()#metrics=['rouge-n', 'rouge-l'],
    #                           max_n=4,
    #                           limit_length=True,
    #                           length_limit=110,
    #                           length_limit_type='words',
    #                           apply_avg=False,
    #                           apply_best=False,
    #                           alpha=0.5, # Default F1_score
    #                           weight_factor=1.2,
    #                           stemming=True)

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


def run_batch(model, args, device, compute_loss_fct, splitlosses, auxloss=False):
    for arg in args:
        if arg is not None:
            arg = arg.to(device)

    output = model(*args)
    allloss = compute_loss_fct(output, args[0], args[1], splitlosses=splitlosses)

    if splitlosses:
        return allloss
    return allloss.mean()

def evaluate(val_loader, model, device, compute_loss_fct, foutname, splitlosses,  auxloss=False):
    fout = open(foutname,"w")
    val_loss = 0
    if splitlosses:
        val_loss = {}
    for j, args in enumerate(tqdm(val_loader)):
        with torch.no_grad():
            l = run_batch(model, args, device, compute_loss_fct, splitlosses=splitlosses, auxloss=auxloss)
            if splitlosses:
                for idx in range(len(l)):
                    val_loss[idx] = val_loss.get(idx,0) + float(l[idx].mean().item())
            else:
                val_loss += float(l.item())
            fout.write(str(l[0].mean().item()) + "\t" + str(l[1].mean().item())+ "\t" + str(l[2].mean().item()) + "\t" + str(l[3].mean().item()) + "\n")
    return val_loss


def init(args):

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

def main(args):
    init(args)
    #Args setup:

    beam = args.beam
    p = args.p
    n_ctx = args.n_ctx
    gen_len = args.gen_len
    k = args.k
    decoding_strategy = args.decoding_strategy
    accum_iter = args.accum_iter
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print("device", device, "n_gpu", n_gpu)
    data_dir = args.data_dir
    #Text Encoder

    if args.debug_mode:
        text_encoder = GPT2Tokenizer.from_pretrained('gpt2')
    else:
        text_encoder = GPT2Tokenizer.from_pretrained('gpt2-medium')
    text_encoder.add_special_tokens({'bos_token':'_start_',
                                     'cls_token':'_classify_',
                                     'eos_token':'_end_',
                                     'additional_special_tokens': ['_kw_','_endkw_', '_t_', '_i_', '_b_', '_c_']
                                    })

    vocab = len(text_encoder)

    datafile = os.path.join(data_dir, "test_encoded.jsonl") if args.testset else os.path.join(data_dir, "val_encoded.jsonl")
    print("Loading dataset...")


    if args.use_model == "full":
        val_loader = get_paragraph_input_loader(datafile, n_gpu, text_encoder, 
                                                    num_workers=0, shuffle=False, gen_len=gen_len, n_ctx=n_ctx, 
                                                    include_neigh= args.use_neighbor_feat, include_curr= False, 
                                                    include_kw = not args.exclude_kw, max_size=args.max_ex, dim = args.n_embd, debug_mode=args.debug_mode)


    elif  args.use_model == "vanilla":
        val_loader = get_paragraph_input_loader(datafile, n_gpu, text_encoder, 
                                                    num_workers=0, shuffle=False, gen_len=gen_len, n_ctx=n_ctx, 
                                                    include_neigh= False, include_discourse_type=False,
                                                    include_kw = not args.exclude_kw, max_size=args.max_ex, dim = args.n_embd, debug_mode=args.debug_mode)

    elif args.use_model == "memory":
        val_loader = get_paragraph_history_input_loader(datafile,n_gpu, text_encoder,
                                                    num_workers=0, shuffle=False, gen_len=gen_len, n_ctx=n_ctx,
                                                    include_kw = not args.exclude_kw, max_size=args.max_ex, use_kwmem=args.use_kwmem, dim = args.n_embd, debug_mode=args.debug_mode)


    # print(len(val_loader))
    #asli
    if args.use_model == "memory":
        doc_model = DocumentMemoryDecoderModel(args, vocab=vocab, n_ctx=n_ctx, gen_len=gen_len, lastidx=text_encoder.eos_token_id, includeprev=args.use_neighbor_feat)

    else:
        doc_model = DocumentDecoderModel(args, vocab=vocab, n_ctx=n_ctx, gen_len=gen_len, lastidx=text_encoder.eos_token_id, includeprev=args.use_neighbor_feat)

    criterion = nn.CrossEntropyLoss(reduction="none")

    lm_loss = ParagraphLoss(criterion, n_ctx=n_ctx, gen_len=gen_len)
    auxloss=False

    doc_model.to(device)
    if n_gpu > 1:
        doc_model = DataParallelModel(doc_model)
        lm_loss = DataParallelCriterion(lm_loss)

    prevloss = []
    upd = []
    start_iter, running_loss = 1,0
    load_dir = args.load_dir
    bestcheck = os.path.join(load_dir,"checkpoint_best.pt")
    checkpoint = torch.load(bestcheck, map_location='cpu')
    state_dict = checkpoint["state_dict"]
    if state_dict.get('module.pos_emb_mask') is None and  doc_model.state_dict().get('module.pos_emb_mask') is not None:
        state_dict['module.pos_emb_mask'] = doc_model.state_dict().get('module.pos_emb_mask') 
    doc_model.load_state_dict(state_dict)
    print("Parallelized")

    vort = 'test' if args.testset else 'val'
    evaluate(val_loader, doc_model, device, lm_loss, os.path.join(args.save_dir,vort+'LOSS.txt'), splitlosses=True, auxloss=False)
    evaluate_doc_model(doc_model, val_loader, text_encoder, device, beam, gen_len, k, p, args.decoding_strategy, os.path.join(args.save_dir,vort+'ROUGE.json'), 'gen','tgt', gen_len, [], args)





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--desc', type=str, help="Description")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_hidden_states', action='store_true')
    parser.add_argument('--output_attentions', action='store_true')
    parser.add_argument('--output_past', action='store_true')
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--n_batch', type=int, default=2)
    parser.add_argument('--max_grad_norm', type=int, default=1)
    parser.add_argument('--lr', type=float, default=6.25e-5)
    parser.add_argument('--lr_warmup', type=float, default=0.002)
    parser.add_argument('--n_embd', type=int, default=1024)
    parser.add_argument('--n_head', type=int, default=12)
    parser.add_argument('--n_layer', type=int, default=12)
    parser.add_argument('--embd_pdrop', type=float, default=0.1)
    parser.add_argument('--attn_pdrop', type=float, default=0.1)
    parser.add_argument('--resid_pdrop', type=float, default=0.1)
    parser.add_argument('--clf_pdrop', type=float, default=0.1)
    parser.add_argument('--l2', type=float, default=0.01)
    parser.add_argument('--vector_l2', action='store_true')
    parser.add_argument('--opt', type=str, default='adam')
    parser.add_argument('--afn', type=str, default='gelu')
    parser.add_argument('--lr_schedule', type=str, default='warmup_linear')
    parser.add_argument('--encoder_path', type=str, default='model/encoder_bpe_40000.json')
    parser.add_argument('--vocab_path', type=str, default='model/vocab_40000.bpe')
    parser.add_argument('--n_transfer', type=int, default=12)
    parser.add_argument('--lm_coef', type=float, default=0.5)
    parser.add_argument('--b1', type=float, default=0.9)
    parser.add_argument('--b2', type=float, default=0.999)
    parser.add_argument('--e', type=float, default=1e-8)
    parser.add_argument('--min_len', type=int, default=100)
    parser.add_argument('--repeattheta', type=float, default=1.5)
    # Custom
    parser.add_argument('--load_dir', type=str, default="output")
    parser.add_argument('--save_dir', type=str, default="output")
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--max_ex', type=int, default=None)

    parser.add_argument('--beam', type=int, default=0)
    parser.add_argument('--k', type=int, default=0)
    parser.add_argument('--p', type=int, default=0)
    parser.add_argument('--decoding_strategy', type=int, default=0)
    parser.add_argument('--accum_iter', type=int, default=2)
    parser.add_argument('--gen_len', type=int, default=922)
    parser.add_argument('--n_ctx', type=int, default=102)
    parser.add_argument('--bodynum', type=int, default=3)
    parser.add_argument('--show_progress', action='store_true')
    parser.add_argument('--use_neighbor_feat', action='store_true')
    parser.add_argument('--use_kwmem', action='store_true')
    parser.add_argument('--exclude_kw', action='store_true')
    parser.add_argument('--testset', action='store_true')
    parser.add_argument('--memstatesize', type=int, default=10)
    parser.add_argument('--use_model', type=str, choices=['vanilla', 'full', 'memory'])
    parser.add_argument('--use_dual_att', action='store_true')
    parser.add_argument('--debug_mode', action='store_true')
    args = parser.parse_args()
    print(torch.__version__)
    print(args)
    main(args)
