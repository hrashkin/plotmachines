import shutil
import argparse
import os
import random

import numpy as np
import rouge
import torch
from torch import nn
from tqdm import tqdm
from evaluate import format_text
import math
from data_loader import get_paragraph_input_loader, get_paragraph_history_input_loader
from evaluate import evaluate_doc_model
from generate import generate_paragraph
from decodermodules import DocumentDecoderModel, DocumentMemoryDecoderModel

from logger import Logger
from loss import ParagraphLoss, ParagraphAndAuxLoss
from model_pytorch import load_openai_pretrained_model
from opt import OpenAIAdam
from parallel import DataParallelModel, DataParallelCriterion
from text_utils import TextEncoder
from transformers import *
def get_average_scores(hyps, refs, maxlen=400, stop_words=[]):       
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
    return averaged_scores

def run_batch(model, args, device, compute_loss_fct, splitlosses, auxloss=False):
    for arg in args:
        if arg is not None:
            arg = arg.to(device)

    if not auxloss:
        output = model(*args)
        allloss = compute_loss_fct(output, args[0], args[1], splitlosses=splitlosses)
        #print(allloss, args[1][:,-401:].sum(dim=1))
    elif auxloss:
        output = model(*args, returnlast=True)
        if torch.cuda.device_count() == 1:
            allloss = compute_loss_fct(*output, *args, splitlosses=splitlosses)
        else:
            allloss = compute_loss_fct(output, *args, splitlosses=splitlosses)
  
    if splitlosses:
        return allloss
    return allloss.mean()

# def save_checkpoint(iter_num, running_loss, model_state_dict, optimizer_state_dict, save_dir):
#     print('Saving a checkpoint...')
#     torch.save({
#         "iter": iter_num,
#         "running_loss": running_loss,
#         "state_dict": model_state_dict,
#         "optimizer": optimizer_state_dict
#     }, os.path.join(save_dir, "checkpoint_best.pt"))

def save_checkpoint(iter_num, running_loss, model_state_dict, optimizer_state_dict, save_dir,my_local_dir):
    print('Saving a checkpoint...' + my_local_dir)
    torch.save({
        "iter": iter_num,
        "running_loss": running_loss,
        "state_dict": model_state_dict,
        "optimizer": optimizer_state_dict
    }, os.path.join(my_local_dir, "checkpoint_best.pt"))

    trial = 0
    while trial < 10:
        try:
            print('Copying a checkpoint...from ' + my_local_dir + ' to ' + save_dir)
            shutil.copy(os.path.join(my_local_dir, "checkpoint_best.pt"), os.path.join(save_dir, "checkpoint_best.pt"))
            trial = 100
        except Exception as e:
            print(e)
            os.makedirs(save_dir, exist_ok=True)
            trial += 1

def load_checkpoint(checkpoint_file, model, model_opt):
    """
    Loads a checkpoint including model state and running loss for continued training
    """
    if checkpoint_file is not None:
        checkpoint = torch.load(checkpoint_file)
        state_dict = checkpoint["state_dict"]
        start_iter = checkpoint['iter']
        running_loss = checkpoint['running_loss']
        opt_state_dict = checkpoint['optimizer']
        model_opt.load_state_dict(opt_state_dict)
        for state in model_opt.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.cuda()
        #for key in list(state_dict.keys()):
        #    state_dict[key[7:]] = state_dict[key]
        #    del state_dict[key]
        #pos_emb_mask = torch.zeros(1, 1, vocab)
        #pos_emb_mask[:, :, -n_ctx] = -1e12
        model.load_state_dict(state_dict)
    else:
        start_iter = 1
        running_loss = 0
    return start_iter, running_loss


def evaluate(val_loader, train_log_interval, model, text_encoder, device, beam, gen_len, k,p, decoding_strategy, compute_loss_fct,splitlosses, auxloss=False, min_len=10):
    hyps, refs = [], []
    val_loss = 0
    if splitlosses:
        val_loss = {}
    for j, args in enumerate(val_loader):
        with torch.no_grad():
            '''if j == train_log_interval:
                break'''
            if j <= 5:
                model.eval()
                # Generating outputs for evaluation
                src_strs, new_refs, new_hyps = generate_paragraph(model, args, text_encoder, device, beam, gen_len, k,p, decoding_strategy, min_len=min_len)
                hyps.extend(new_hyps)
                refs.extend(new_refs)
            # Calculating loss
            l = run_batch(model, args, device, compute_loss_fct, splitlosses=splitlosses, auxloss=auxloss)
            if splitlosses:
                for idx in range(len(l)):
                    val_loss[idx] = val_loss.get(idx,0) + float(l[idx].mean().item())
            else:
                val_loss += float(l.item())
    try:
        print('Hypothesis: {}'.format(hyps[0]))
        print("Reference: {}".format(refs[0]))
    except:
        pass
    scores = get_average_scores(hyps, refs, maxlen=gen_len)
#     scores = None
    return val_loss, scores

def get_loss_value(num, denom):
        """Log a scalar variable."""
        if isinstance(num, (dict,list)):
            v1,v2,v3,v4 = 0.,0.,0.,0
            try:
                v1 = num[0]/denom
                v2 = num[1]/denom
                v3 = num[2]/denom
                v4 = num[3]/denom
            except:
                return 5.
                pass
            return v1+v2+v3+v4
        else:
            value = num/denom
            return value

def run_epoch(bestloss, start_iter, running_loss, model, compute_loss_fct, model_opt, train_loader, val_loader, train_log_interval, val_log_interval, device, beam, gen_len, k,p, decoding_strategy, accum_iter, desc_str, save_dir, logger, text_encoder, show_progress=False, summary_loss=None, planstart=0, auxloss=False, my_local_dir='checkpoints_local'):
    
    if show_progress:
        train_bar = tqdm(iterable=train_loader, desc=desc_str)
    else:
        train_bar = train_loader
    
    #val_loss, _ = evaluate(val_loader, train_log_interval, model, text_encoder, device, beam, gen_len, k, p, decoding_strategy, compute_loss_fct, splitlosses=True, auxloss=auxloss, min_len=args.min_len)
    #lv = get_loss_value(val_loss, len(val_loader))
    #print(lv)
    for i, batchargs in enumerate(train_bar, start_iter):
        num_updates = i // accum_iter
        model.train()
        loss = run_batch(model, batchargs, device, compute_loss_fct, splitlosses=False, auxloss=auxloss)
        loss.backward()

        running_loss += float(loss.item())
        if show_progress:
            train_bar.set_postfix(loss=running_loss / ((train_log_interval * accum_iter) if num_updates % train_log_interval == 0 and num_updates != 0 else i % (train_log_interval * accum_iter)))

        if i % accum_iter == 0:
            model_opt.step()
            model_opt.zero_grad()
            torch.cuda.empty_cache()
        if num_updates % train_log_interval == 0 and i % accum_iter == 0:
            logger.scalar_summary("Training", num=running_loss, denom=(train_log_interval * accum_iter), step=num_updates)
            print("training loss %.2f" % (running_loss/float(train_log_interval * accum_iter)))
            running_loss = 0
        # if True:
        if num_updates % 1000 == 0 and i % accum_iter == 0:
            val_loss, scores = evaluate(val_loader, train_log_interval, model, text_encoder, device, beam, gen_len, k, p, decoding_strategy, compute_loss_fct, splitlosses=True, auxloss=auxloss, min_len=args.min_len)
            # for key, value in scores.items():
            #     for key2, value2 in value.items():
            #         logger.rouge_summary("{}/{}".format(key, key2), value2, num_updates)
            # print("Validation rouge: " + str(scores.items()))
            #print("Validation loss: " + str(val_loss[0]/len(val_loader)))
            logger.scalar_summary("Validation", num=val_loss, denom=len(val_loader), step=num_updates)
            # if sum(val_loss) < bestloss or bestloss == -1:
            lv = get_loss_value(val_loss, len(val_loader))
            if (not math.isnan(lv)) and (bestloss == -1 or lv < bestloss):
                bestloss = lv
                save_checkpoint(i + 1, running_loss, model.state_dict(), model_opt.state_dict(), save_dir, my_local_dir)


    val_loss, scores = evaluate(val_loader, train_log_interval, model, text_encoder, device, beam, gen_len, k, p, decoding_strategy, compute_loss_fct, splitlosses=True, auxloss=auxloss, min_len=args.min_len)
    for key, value in scores.items():
        for key2, value2 in value.items():
            logger.rouge_summary("{}/{}".format(key, key2), value2, num_updates)
    print("Validation rouge: " + str(scores.items()))
    #print("Validation loss: " + str(val_loss[0]/len(val_loader)))
    logger.scalar_summary("Validation", num=val_loss, denom=len(val_loader), step=num_updates)
    #cumloss = sum(val_loss)/len(val_loader)
    #if sum(val_loss) < bestloss or bestloss == -1:
    lv = get_loss_value(val_loss, len(val_loader))
    if (not math.isnan(lv)) and (bestloss == -1 or lv < bestloss):
        bestloss = lv
        save_checkpoint(i + 1, running_loss, model.state_dict(), model_opt.state_dict(), save_dir,my_local_dir)


    torch.cuda.empty_cache()
    return i + 1, running_loss, bestloss, num_updates, lv

def init(args):
    print("Creating directories")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, args.experiment_name), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, args.experiment_name), exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

def main(args):
    init(args)
    #Args setup:
    save_dir = os.path.join(args.output_dir, args.experiment_name, "checkpoints")
    save_dir_local = "checkpoints_local"
    desc = args.desc
    data_dir = args.data_dir
    log_dir = os.path.join(args.output_dir, args.experiment_name, "logs")
    # if not os.path.exists(log_dir):
    #     os.makedirs(log_dir)
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    #
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_dir_local, exist_ok=True)

    train_log_interval = args.train_log_interval
    val_log_interval = args.val_log_interval
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
    logger = Logger(log_dir)

    #Text Encoder
    if args.use_offline_gpt2:
        text_encoder = GPT2Tokenizer.from_pretrained('./gpt2model')
    elif args.debug_mode:
        text_encoder = GPT2Tokenizer.from_pretrained('gpt2')
    else:
        text_encoder = GPT2Tokenizer.from_pretrained('gpt2-medium')

    text_encoder.add_special_tokens({'bos_token':'_start_',
                                     'cls_token':'_classify_',
                                     'eos_token':'_end_',
                                     'additional_special_tokens': ['_kw_','_endkw_', '_t_', '_i_', '_b_', '_c_']
                                    })

    vocab = len(text_encoder)

    print("Loading dataset...")
    if args.use_model == "full":
        train_loader = get_paragraph_input_loader(os.path.join(data_dir, "train_encoded.jsonl"), args.n_batch, text_encoder, 
                                                    num_workers=3, shuffle=True, gen_len=gen_len, n_ctx=n_ctx, include_discourse_type=True, 
                                                    include_neigh= args.use_neighbor_feat, include_curr= args.use_aux_losses, max_size=args.max_ex,
                                                    include_kw = not args.exclude_kw, dim = args.n_embd, debug_mode=args.debug_mode)

        val_loader = get_paragraph_input_loader(os.path.join(data_dir, "val_encoded.jsonl"), n_gpu, text_encoder, 
                                                    num_workers=0, shuffle=False, gen_len=gen_len, n_ctx=n_ctx, include_discourse_type=True,
                                                    include_neigh= args.use_neighbor_feat, include_curr= args.use_aux_losses, max_size=args.num_val_examples,
                                                    include_kw = not args.exclude_kw, dim = args.n_embd, debug_mode=args.debug_mode)

        print("Train length: {}, Validation length: {}".format(len(train_loader), len(val_loader)))

    elif args.use_model == "memory":
        #asli
        train_loader = get_paragraph_history_input_loader(os.path.join(data_dir, "train_encoded.jsonl"), args.n_batch, text_encoder, 
                                                    num_workers=3, shuffle=True, gen_len=gen_len, n_ctx=n_ctx, include_discourse_type=True,
                                                    include_neigh= args.use_neighbor_feat, include_curr= args.use_aux_losses, max_size = args.max_ex,
                                                    include_kw = not args.exclude_kw, memsize=args.memstatesize, dim = args.n_embd, use_kwmem=args.use_kwmem, debug_mode=args.debug_mode)

        val_loader = get_paragraph_history_input_loader(os.path.join(data_dir, "val_encoded.jsonl"), n_gpu, text_encoder, 
                                                    num_workers=0, shuffle=False, gen_len=gen_len, n_ctx=n_ctx, include_discourse_type=True,
                                                    include_neigh= args.use_neighbor_feat, include_curr= args.use_aux_losses, max_size = args.num_val_examples,
                                                    include_kw = not args.exclude_kw, memsize=args.memstatesize, dim = args.n_embd, use_kwmem=args.use_kwmem, debug_mode=args.debug_mode)

        print("Train length: {}, Validation length: {}".format(len(train_loader), len(val_loader)))
    elif  args.use_model == "vanilla":
        train_loader = get_paragraph_input_loader(os.path.join(data_dir, "train_encoded.jsonl"), args.n_batch, text_encoder,
                                                    num_workers=3, shuffle=True, gen_len=gen_len, n_ctx=n_ctx, max_size=args.max_ex, 
                                                    include_kw = not args.exclude_kw, include_discourse_type=False, dim = args.n_embd, debug_mode=args.debug_mode)
        val_loader = get_paragraph_input_loader(os.path.join(data_dir, "val_encoded.jsonl"), n_gpu, text_encoder, 
                                                    num_workers=0, shuffle=False, gen_len=gen_len, n_ctx=n_ctx,max_size=args.num_val_examples,
                                                     include_kw = not args.exclude_kw, include_discourse_type=False, dim = args.n_embd, debug_mode=args.debug_mode)
        print("Train length: {}, Validation length: {}".format(len(train_loader), len(val_loader)))

    n_updates_total = (len(train_loader) // args.accum_iter) * (args.num_epochs)


    if args.use_model == "memory":
        doc_model = DocumentMemoryDecoderModel(args, vocab=vocab, n_ctx=n_ctx, gen_len=gen_len, lastidx=text_encoder.eos_token_id, includeprev=args.use_neighbor_feat, use_offline_gpt2 = args.use_offline_gpt2)
    else:
        doc_model = DocumentDecoderModel(args, vocab=vocab, n_ctx=n_ctx, gen_len=gen_len, lastidx=text_encoder.eos_token_id, includeprev=args.use_neighbor_feat, use_offline_gpt2 = args.use_offline_gpt2)
    fm = open(log_dir+"/modeldescr.txt","w")
    fm.write(str(doc_model))
    fm.close()
    criterion = nn.CrossEntropyLoss(reduction="none")

    print(sum(p.numel() for p in doc_model.parameters() if p.requires_grad))
    model_opt = AdamW(filter(lambda p : p.requires_grad, doc_model.parameters()),
                           lr=args.lr,
                           betas=(args.b1,args.b2),
                           eps=args.e)
                           #l2=args.l2,
                           #vector_l2=args.vector_l2,
                           #max_grad_norm=args.max_grad_norm)
                           #
                           #schedule=args.lr_schedule,
                           #warmup=args.lr_warmup,
                           #t_total=n_updates_total,

    if args.use_aux_losses:
        lm_loss = ParagraphAndAuxLoss(criterion, opt=None, n_ctx=n_ctx, gen_len=gen_len)
        auxloss = True
    else:
        lm_loss = ParagraphLoss(criterion, n_ctx=n_ctx, gen_len=gen_len)
        auxloss=False

    print("Loading Model")
    ###load_openai_pretrained_model(doc_model.decoder, n_ctx=n_ctx+gen_len, n_special=n_special, path="model/", path_names="model/")
    doc_model.to(device)
    if n_gpu > 1:
        doc_model = DataParallelModel(doc_model)
        lm_loss = DataParallelCriterion(lm_loss)
    print("Parallelized")

    bestloss = -1
    start_iter, running_loss = 1,0
    prevloss = 1000

    start_iter, running_loss = load_checkpoint(args.checkpoint, doc_model, model_opt)
    for i in range(args.num_epochs):
        start_iter, running_loss, bestloss, updates, val_loss1 = run_epoch(bestloss, start_iter, running_loss, doc_model, lm_loss, model_opt, train_loader, val_loader, train_log_interval, val_log_interval, device, beam, gen_len, k, p, decoding_strategy, accum_iter, "FT Training Epoch [{}/{}]".format(i + 1, args.num_epochs), save_dir, logger, text_encoder, show_progress=args.show_progress, auxloss=auxloss, my_local_dir=save_dir_local)
        print("VAL LOSS: ", str(val_loss1))
        if val_loss1 > prevloss or math.isnan(val_loss1):
            break
        prevloss = val_loss1
        #if len(prevloss) >= 3 and cumloss >= prevloss[-3] and prevloss[-1] >= prevloss[-3] and prevloss[-2] >= prevloss[-3]:
        #    break

    bestcheck = os.path.join(save_dir,"checkpoint_best.pt")
    checkpoint = torch.load(bestcheck, map_location='cpu')
    state_dict = checkpoint["state_dict"]
    if state_dict.get('module.pos_emb_mask') is None and  doc_model.state_dict().get('module.pos_emb_mask') is not None:
        state_dict['module.pos_emb_mask'] = doc_model.state_dict().get('module.pos_emb_mask') 
    doc_model.load_state_dict(state_dict)
    #val_loader = get_paragraph_input_loader(os.path.join(data_dir, "val_encoded.jsonl"), n_gpu, text_encoder, num_workers=0, shuffle=False, gen_len=gen_len, n_ctx=n_ctx, include_neigh= args.use_neighbor_feat)
    evaluate_doc_model(doc_model, val_loader, text_encoder, device, beam, gen_len, k, p, args.decoding_strategy, os.path.join(save_dir,'valeval.log'), 'gen','tgt', gen_len, [], args)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--desc', type=str, help="Description")

    parser.add_argument('--output_hidden_states', action='store_true')
    parser.add_argument('--output_attentions', action='store_true')
    parser.add_argument('--output_past', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
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
    # Custom
    parser.add_argument('--output_dir', type=str, default=os.getenv('PT_OUTPUT_DIR', '/tmp'))
    parser.add_argument('--experiment_name', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='data')

    parser.add_argument('--train_log_interval', type=int, default=100)
    parser.add_argument('--val_log_interval', type=int, default=2000)
    parser.add_argument('--num_val_examples', type=int, default=None)
    parser.add_argument('--beam', type=int, default=0)
    parser.add_argument('--k', type=int, default=0)
    parser.add_argument('--p', type=int, default=0)
    parser.add_argument('--decoding_strategy', type=int, default=0)
    parser.add_argument('--accum_iter', type=int, default=2)
    parser.add_argument('--gen_len', type=int, default=922)
    parser.add_argument('--n_ctx', type=int, default=102)
    parser.add_argument('--show_progress', action='store_true')
    parser.add_argument('--use_neighbor_feat', action='store_true')
    parser.add_argument('--use_aux_losses', action='store_true')
    parser.add_argument('--exclude_kw', action='store_true')
    parser.add_argument('--max_ex', type=int, default=None)
    parser.add_argument('--min_len', type=int, default=100)
    parser.add_argument('--repeattheta', type=float, default=1.5)
    parser.add_argument('--memstatesize', type=int, default=10)
    parser.add_argument('--use_model', type=str, choices=['vanilla', 'full', 'memory'])

    parser.add_argument('--use_kwmem', action='store_true')
    parser.add_argument('--use_dual_att', action='store_true')
    parser.add_argument('--use_offline_gpt2', action='store_true')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--debug_mode', action='store_true')

    args = parser.parse_args()
    print(torch.__version__)
    print(args)
    main(args)
