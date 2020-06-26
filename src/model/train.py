import shutil
import argparse
import os
import random
import numpy as np
import rouge
import torch
from torch import nn
from tqdm import tqdm
import math

from data_loader import get_paragraph_input_loader, get_paragraph_memory_input_loader
from eval_utils import format_text, evaluate_doc_model
from generate_utils import generate_paragraph
from model import GPT2BaseModel, PlotMachinesModel
from logger import Logger
from loss import ParagraphLoss
from parallel import DataParallelModel, DataParallelCriterion
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

def run_batch(model, args, device, compute_loss_fct):
    for arg in args:
        if arg is not None:
            arg = arg.to(device)

    output = model(*args)
    allloss = compute_loss_fct(output, args[0], args[1])
    
    return allloss.mean()

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
        model.load_state_dict(state_dict)
    else:
        start_iter = 1
        running_loss = 0
    return start_iter, running_loss


def evaluate(val_loader, train_log_interval, model, text_encoder, device, beam, gen_len, k,p, decoding_strategy, compute_loss_fct, min_len=10):
    hyps, refs = [], []
    val_loss = 0
    for j, args in enumerate(val_loader):
        with torch.no_grad():
            if j <= 5:
                #evaluate Rouge on a very small subset of dev examples just to double check that training is working
                model.eval()
                # Generating outputs for evaluation
                src_strs, new_refs, new_hyps = generate_paragraph(model, args, text_encoder, device, beam, gen_len, k,p, decoding_strategy, min_len=min_len)
                hyps.extend(new_hyps)
                refs.extend(new_refs)
            # Calculating loss
            l = run_batch(model, args, device, compute_loss_fct)
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
    value = num/denom
    return value


'''Run a single training epoch:
@params-
bestloss: the best loss over any evaluation on the dev set
start_iter: the batch in the epoch to start with 
running_loss: the total loss since the last checkpoint update
model: the model being trained
compute_loss_fct: a loss function (from loss.py)
model_opt: the argparse options
train_loader, val_loader: training and validation data loaders
train_log_interval,val_log_interval: how often to log training and validation losses
device: cuda or cpu
beam, gen_len, k, p, decoding_strategy: decoding parameters 
accum_iter: how often to run backprop
desc_str: string for showing progress, 
save_dir: where to save checkpoints, 
logger: class for logging progress (mostly for debugging), 
text_encoder: the tokenizer, 
show_progress=False: whether to log progress to the command line 
summary_loss=None: not in use anymore
my_local_dir='checkpoints_local': a local checkpoint storage if running on servers
'''
def run_epoch(bestloss, start_iter, running_loss, model, compute_loss_fct, model_opt, train_loader, val_loader, train_log_interval, val_log_interval, device, beam, gen_len, k,p, decoding_strategy, accum_iter, desc_str, save_dir, logger, text_encoder, show_progress=False, summary_loss=None, my_local_dir='checkpoints_local'):
    '''
    Run a single epoch, log results, and save best checkpoint
    '''
    if show_progress:
        train_bar = tqdm(iterable=train_loader, desc=desc_str)
    else:
        train_bar = train_loader

    for i, batchargs in enumerate(train_bar, start_iter):
        num_updates = i // accum_iter
        model.train()
        loss = run_batch(model, batchargs, device, compute_loss_fct)
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

        if num_updates % 1000 == 0 and i % accum_iter == 0:
            val_loss, scores = evaluate(val_loader, train_log_interval, model, text_encoder, device, beam, gen_len, k, p, decoding_strategy, compute_loss_fct, min_len=args.min_len)

            logger.scalar_summary("Validation", num=val_loss, denom=len(val_loader), step=num_updates)
            # if sum(val_loss) < bestloss or bestloss == -1:
            lv = get_loss_value(val_loss, len(val_loader))
            if (not math.isnan(lv)) and (bestloss == -1 or lv < bestloss):
                bestloss = lv
                save_checkpoint(i + 1, running_loss, model.state_dict(), model_opt.state_dict(), save_dir, my_local_dir)


    val_loss, scores = evaluate(val_loader, train_log_interval, model, text_encoder, device, beam, gen_len, k, p, decoding_strategy, compute_loss_fct, min_len=args.min_len)
    for key, value in scores.items():
        for key2, value2 in value.items():
            logger.rouge_summary("{}/{}".format(key, key2), value2, num_updates)
    print("Validation rouge: " + str(scores.items()))
    logger.scalar_summary("Validation", num=val_loss, denom=len(val_loader), step=num_updates)
    lv = get_loss_value(val_loss, len(val_loader))
    if (not math.isnan(lv)) and (bestloss == -1 or lv < bestloss):
        bestloss = lv
        save_checkpoint(i + 1, running_loss, model.state_dict(), model_opt.state_dict(), save_dir, my_local_dir)


    torch.cuda.empty_cache()
    return i + 1, running_loss, bestloss, num_updates, lv

def print_model_params(log_dir, doc_model):
    fm = open(log_dir+"/modeldescr.txt","w")
    fm.write(str(doc_model))
    fm.close()
    print(sum(p.numel() for p in doc_model.parameters() if p.requires_grad))


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
    if args.use_model == "base":
        train_loader = get_paragraph_input_loader(os.path.join(data_dir, "train_encoded.csv"), args.n_batch, text_encoder, 
                                                    num_workers=3, shuffle=True, gen_len=gen_len, n_ctx=n_ctx, include_discourse_type=args.use_discourse, 
                                                    include_neigh= args.use_neighbor_feat, max_size=args.max_ex,
                                                    include_kw = not args.exclude_kw, dim = args.n_embd, debug_mode=args.debug_mode)

        val_loader = get_paragraph_input_loader(os.path.join(data_dir, "val_encoded.csv"), n_gpu, text_encoder, 
                                                    num_workers=0, shuffle=False, gen_len=gen_len, n_ctx=n_ctx, include_discourse_type=args.use_discourse,
                                                    include_neigh= args.use_neighbor_feat, max_size=args.num_val_examples,
                                                    include_kw = not args.exclude_kw, dim = args.n_embd, debug_mode=args.debug_mode)

        print("Train length: {}, Validation length: {}".format(len(train_loader), len(val_loader)))
        doc_model = GPT2BaseModel(args, vocab=vocab, n_ctx=n_ctx, gen_len=gen_len, lastidx=text_encoder.eos_token_id, includeprev=args.use_neighbor_feat, use_offline_gpt2 = args.use_offline_gpt2)

    elif args.use_model == "plotmachines":
        #asli
        train_loader = get_paragraph_memory_input_loader(os.path.join(data_dir, "train_encoded.csv"), args.n_batch, text_encoder, 
                                                    num_workers=3, shuffle=True, gen_len=gen_len, n_ctx=n_ctx, include_discourse_type=args.use_discourse,
                                                    include_neigh= args.use_neighbor_feat, max_size = args.max_ex,
                                                    include_kw = not args.exclude_kw, memsize=args.memstatesize, dim = args.n_embd, use_kwmem=True, debug_mode=args.debug_mode)

        val_loader = get_paragraph_memory_input_loader(os.path.join(data_dir, "val_encoded.csv"), n_gpu, text_encoder, 
                                                    num_workers=0, shuffle=False, gen_len=gen_len, n_ctx=n_ctx, include_discourse_type=args.use_discourse,
                                                    include_neigh= args.use_neighbor_feat, max_size = args.num_val_examples,
                                                    include_kw = not args.exclude_kw, memsize=args.memstatesize, dim = args.n_embd, use_kwmem=True, debug_mode=args.debug_mode)

        print("Train length: {}, Validation length: {}".format(len(train_loader), len(val_loader)))
        doc_model = PlotMachinesModel(args, vocab=vocab, n_ctx=n_ctx, gen_len=gen_len, lastidx=text_encoder.eos_token_id, includeprev=args.use_neighbor_feat, use_offline_gpt2 = args.use_offline_gpt2)
    


    n_updates_total = (len(train_loader) // args.accum_iter) * (args.num_epochs)

    if args.debug_mode:
        print_model_params(log_dir, doc_model)

    criterion = nn.CrossEntropyLoss(reduction="none")

    model_opt = AdamW(filter(lambda p : p.requires_grad, doc_model.parameters()),
                           lr=args.lr,
                           betas=(args.b1,args.b2),
                           eps=args.e)

    lm_loss = ParagraphLoss(criterion, n_ctx=n_ctx, gen_len=gen_len)

    print("Loading Model")
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
        start_iter, running_loss, bestloss, updates, val_loss1 = run_epoch(bestloss, start_iter, running_loss, doc_model, lm_loss, model_opt, train_loader, val_loader, train_log_interval, val_log_interval, device, beam, gen_len, k, p, decoding_strategy, accum_iter, "FT Training Epoch [{}/{}]".format(i + 1, args.num_epochs), save_dir, logger, text_encoder, show_progress=args.show_progress, my_local_dir=save_dir_local)
        print("VAL LOSS: ", str(val_loss1))
        if val_loss1 > prevloss or math.isnan(val_loss1):
            break
        prevloss = val_loss1


    print('Done training...')
    print('Evaluating on validation with best checkpoint...')

    bestcheck = os.path.join(save_dir,"checkpoint_best.pt")
    checkpoint = torch.load(bestcheck, map_location='cpu')
    state_dict = checkpoint["state_dict"]
    if state_dict.get('module.pos_emb_mask') is None and  doc_model.state_dict().get('module.pos_emb_mask') is not None:
        state_dict['module.pos_emb_mask'] = doc_model.state_dict().get('module.pos_emb_mask') 
    doc_model.load_state_dict(state_dict)
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
    parser.add_argument('--n_transfer', type=int, default=12)
    parser.add_argument('--lm_coef', type=float, default=0.5)
    parser.add_argument('--b1', type=float, default=0.9)
    parser.add_argument('--b2', type=float, default=0.999)
    parser.add_argument('--e', type=float, default=1e-8)
    # Custom
    parser.add_argument('--output_dir', type=str, default=os.getenv('PT_OUTPUT_DIR', '/tmp'), help='directory to save logs and checkpoints to')
    parser.add_argument('--experiment_name', type=str, required=True, help='name of this experiment will be included in output')
    parser.add_argument('--data_dir', type=str, default='data', help='directory with train, dev, test files')
    parser.add_argument('--train_log_interval', type=int, default=100, help='number of train steps before logging training progress')
    parser.add_argument('--val_log_interval', type=int, default=2000, help='number of train steps before logging validation progress')
    parser.add_argument('--num_val_examples', type=int, default=None, help='max number of validation examples, or None:use all data')
    parser.add_argument('--beam', type=int, default=0, help='beam size for beam search - not in use')
    parser.add_argument('--k', type=int, default=0, help='k for TopK sampling')
    parser.add_argument('--p', type=int, default=0, help='p for Nucleus sampling')
    parser.add_argument('--decoding_strategy', type=int, default=0, help='not in use')
    parser.add_argument('--accum_iter', type=int, default=2, help='number of batches to accumulate gradiencts before doing backprop')
    parser.add_argument('--gen_len', type=int, default=922, help='max generation length + 1 for end token')
    parser.add_argument('--n_ctx', type=int, default=102, help='max outline length + 2 for delimiters')
    parser.add_argument('--show_progress', action='store_true')
    parser.add_argument('--exclude_kw', action='store_true', help='unconditional baseline')
    parser.add_argument('--max_ex', type=int, default=None, help='max number of train examples, or None:use all training data')
    parser.add_argument('--min_len', type=int, default=100, help='minimum generation length')
    parser.add_argument('--repeattheta', type=float, default=1.5, help='how much to penalize repitition (1 is not at all, > 1 is more penalty)')
    parser.add_argument('--memstatesize', type=int, default=100, help='size of global document state portion of memory (default:100)')
    parser.add_argument('--use_model', type=str, choices=['base', 'plotmachines'], help='full plotmachines (w/ memory) vs base gpt (no memory)')
    parser.add_argument('--use_neighbor_feat', action='store_true', help='use neighboring (previous) paragraph encoding as extra input')
    parser.add_argument('--use_discourse', action='store_true', help='use discouse tokens as extra input')
    parser.add_argument('--use_offline_gpt2', action='store_true')
    parser.add_argument('--checkpoint', type=str, default=None, help='location of a previous checkpoint')
    parser.add_argument('--debug_mode', action='store_true')

    args = parser.parse_args()
    print(torch.__version__)
    print(args)
    main(args)
