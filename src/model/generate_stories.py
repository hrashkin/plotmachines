import argparse
import os
import random
import numpy as np
import rouge
import torch
from torch import nn
from tqdm import tqdm

from eval_utils import format_text
from data_loader import get_paragraph_input_loader, get_fullstory_loader
from model import GPT2BaseModel, PlotMachinesModel
from generate_utils import toks_to_str
from parallel import DataParallelModel, DataParallelCriterion
from transformers import *

def tfmclassifier(textlines, model, tokenizer, gen_len):
    '''Create encoding of the previous paragraph (textlines) using the model and tokenizer'''
    clf = []
    nb = len(textlines)
    #if nb < 8:
    wds = torch.zeros(nb, gen_len, dtype=torch.long).cuda()
    mask = torch.zeros(nb, gen_len, dtype=torch.long).cuda()
    for j in range(nb):
          
        temp = torch.tensor(tokenizer.encode(textlines[j], add_special_tokens=False)[:gen_len])
        wds[j,:len(temp)] = temp.cuda()
        mask[j,:len(temp)] = torch.ones(len(temp), dtype=torch.long).cuda()
    model.eval()
    outputs = model(wds)
    total = (mask.unsqueeze(2).type_as(outputs[0]) * outputs[0]).sum(dim=1) / mask.type_as(outputs[0]).sum(dim=1).unsqueeze(1)
    return total

'''Generate a single paragraph'''
def generate_paragraph(model, args, text_encoder, device, beam, gen_len, k, p, decoding_strategy, ids, tagnum, min_len=None, returnnewmem=False):
    src_strs, tgt_strs, gen_strs, genraw, gentok = [], [], [], [], []
    n_gpu = torch.cuda.device_count()
    
    outputs = model(*args, text_encoder=text_encoder, device=device, beam=beam, gen_len=gen_len, k=k, p=p, decoding_strategy=decoding_strategy, generate=True, min_len=min_len)
    if n_gpu == 1:
        outputs = [outputs]
    i = 0

    seenout = []
    if len(outputs[0]) > 3:
        for generated_toks, input_toks, target_toks, seenuni in outputs: ##outputs[0][i],outputs[1][i],outputs[2][i]
            for idx in range(generated_toks.size(0)):
                gentok.append(generated_toks[idx].view(1,-1))
                seenout.append(seenuni[idx])
                #print(toks_to_str(input_toks[idx], text_encoder, is_input=True))
                gen_str = toks_to_str(generated_toks[idx], text_encoder).replace("\n", " ")
                genraw.append(gen_str)
                gen_strs.append(str(ids[2][i].item()) + "\t"+str(ids[0][i])+"\t"+ str(ids[1][i]) +"\t"+str(tagnum)+"\t"+gen_str)
                i+=1
        return gen_strs, genraw, torch.cat(gentok, dim = 0), torch.stack(seenout, dim=0)

    else:
        for generated_toks, input_toks, target_toks in outputs: ##outputs[0][i],outputs[1][i],outputs[2][i]
            for idx in range(generated_toks.size(0)):
                gentok.append(generated_toks[idx].view(1,-1))
                #print(toks_to_str(input_toks[idx], text_encoder, is_input=True))
                gen_str = toks_to_str(generated_toks[idx], text_encoder).replace("\n", " ")
                genraw.append(gen_str)
                gen_strs.append(str(ids[2][i].item()) + "\t"+str(ids[0][i])+"\t"+ str(ids[1][i]) +"\t"+str(tagnum)+"\t"+gen_str)
                i+=1
        return gen_strs, genraw, torch.cat(gentok, dim = 0)


'''Generate full stories'''
def generatedocs(model, gptmodel, gpttok, val_loader, text_encoder, device, beam, gen_len, k, p, decoding_strategy, save_file, gen_dir="gen", tgt_dir="tgt", max_len=110, stop_words=[], args=None, tags=['_i_','_b_','_b_','_b_','_c_'], dim=768, localfile=None, save_dir=None):
    def dump_to_file(jsf, data):
        for i in range(len(data)):
            try:
                jsf.write(data[i] + "\n")
            except:
                jsf.write('error on line ' + str(i) + "\n")
                pass

    data = {'gen':[]}
    srcs, hyps, refs = [], [], []
    model.eval()
    gptmodel.eval()
    iter = 0

    try:
        if os._exists(save_file):
            os.remove(save_file)
    except:
        print("Error while deleting file ", save_file)
    jsf = open(localfile, "w")

    for pad_seq, mask_seq, docids in tqdm(val_loader):
        with torch.no_grad():
            # Generating outputs for evaluation
            prev= ['NA']*pad_seq.size(0)

            kwsize = args.n_ctx-2
            mem = torch.torch.empty(pad_seq.size(0), kwsize + args.memstatesize, args.n_embd).normal_(std=0.02)
            mmask = torch.zeros(mask_seq.size(0), kwsize + args.memstatesize)#.long()
            mmask[:, :kwsize] = mask_seq[:, 1:args.n_ctx-1]
            mmask[:, -args.memstatesize:] = torch.ones(mmask.size(0), args.memstatesize)#.long()

            ph = torch.zeros(pad_seq.size(0), 10, 1, dim)#.long()
            pmask = torch.zeros(pad_seq.size(0), 10, 1)#.long()
            seenunigrams =  torch.ones(pad_seq.size(0), len(text_encoder)) #[{} for _ in range(pad_seq.size(0))]
            idces = torch.arange(pad_seq.size(0))


            for tnum in range(len(tags)):
                tag=  tags[tnum]
                if args.use_model =="plotmachines":
                    if args.use_neighbor_feat:
                        prevprc = tfmclassifier(prev, gptmodel, gpttok, gen_len)
                    if args.use_discourse:
                        pad_seq [:,args.n_ctx-1] = text_encoder.added_tokens_encoder[tag] #add discourse marker
                    modelargs = (pad_seq, mask_seq, mem, mmask, ph, pmask, prevprc, seenunigrams, idces)
                    gen_strs, genraw, gentok, seenunigrams = generate_paragraph(model, modelargs, text_encoder, device, beam, gen_len, k, p, decoding_strategy, docids, tnum, min_len=args.min_len)
                    prevprc = tfmclassifier(genraw, gptmodel, gpttok,gen_len)
                    ph[:, tnum, 0, :] = prevprc
                    pmask[:, tnum, 0] = 1

                else:
                    prevprc = None
                    if args.use_neighbor_feat:
                        prevprc = tfmclassifier(prev, gptmodel, gpttok, gen_len)
                    if args.use_discourse:
                        pad_seq[:,args.n_ctx-1] =  text_encoder.added_tokens_encoder[tag] # add discourse marker
                    modelargs = (pad_seq, mask_seq, prevprc, seenunigrams, idces)
                    gen_strs, genraw, gentok, seenunigrams  = generate_paragraph(model, modelargs, text_encoder, device, beam, gen_len, k, p, decoding_strategy, docids, tnum, min_len=args.min_len)
                data["gen"].extend(gen_strs)
                prev = genraw

            if iter %100 == 0:
                dump_to_file(jsf, data["gen"])
                data = {'gen': []}
            iter+=1

    dump_to_file(jsf, data["gen"])

    import shutil
    trial = 0
    while trial < 10:
        try:
            print('Copying the generated file from ' + localfile + ' to ' + save_file)
            shutil.move(localfile, save_file)
            trial = 100
        except Exception as e:
            print(e)
            os.makedirs(save_dir, exist_ok=True)
            trial += 1

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
    val_loader = get_fullstory_loader(datafile, args.n_batch, text_encoder, num_workers=0, shuffle=False, gen_len=gen_len, n_ctx=n_ctx, include_kw = not args.exclude_kw, max_size=args.max_ex)
    print(len(val_loader))

    if args.use_model == "plotmachines":
        doc_model = PlotMachinesModel(args, vocab=vocab, n_ctx=n_ctx, gen_len=gen_len, lastidx=text_encoder.eos_token_id, includeprev=args.use_neighbor_feat)
    else:
        doc_model = GPT2BaseModel(args, vocab=vocab, n_ctx=n_ctx, gen_len=gen_len, lastidx=text_encoder.eos_token_id, includeprev=args.use_neighbor_feat)

    doc_model.to(device)
    if n_gpu > 1:
        doc_model = DataParallelModel(doc_model)


    if args.debug_mode:
        gptclf = GPT2Model.from_pretrained('gpt2')
        gptclf.eval()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        gptclf.to(device)
        #gpttok = gptTokenizer.from_pretrained('openai-gpt')
        gpttok = GPT2Tokenizer.from_pretrained('gpt2')

    else:
        gptclf = GPT2Model.from_pretrained('gpt2-medium')
        gptclf.eval()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        gptclf.to(device)
        #gpttok = gptTokenizer.from_pretrained('openai-gpt')
        gpttok = GPT2Tokenizer.from_pretrained('gpt2-medium')

    prevloss = []
    upd = []
    start_iter, running_loss = 1,0
    load_dir = args.load_dir
    bestcheck = os.path.join(load_dir,"checkpoint_best.pt")
    checkpoint = torch.load(bestcheck, map_location='cpu')
    state_dict = checkpoint["state_dict"]
    if n_gpu ==1:
      if state_dict.get('module.pos_emb_mask') is None and doc_model.state_dict().get('pos_emb_mask') is not None:
        state_dict['module.pos_emb_mask'] = doc_model.state_dict().get('pos_emb_mask')
      for k in list(state_dict.keys()):
        state_dict[k[7:]] = state_dict[k]
        del state_dict[k]
    else:
      if state_dict.get('module.pos_emb_mask') is None and  doc_model.state_dict().get('module.pos_emb_mask') is not None:
        state_dict['module.pos_emb_mask'] = doc_model.state_dict().get('module.pos_emb_mask')
    doc_model.load_state_dict(state_dict)

    print("Parallelized")
    tagset = ['_i_'] + args.bodynum* ['_b_'] + ['_c_']
    vort = 'test' if args.testset else 'val'
    generatedocs(doc_model, gptclf, gpttok, val_loader, text_encoder, device, beam, gen_len, k, p, args.decoding_strategy, os.path.join(args.save_dir,vort+'.gens.tsv'),
                 'gen','tgt', gen_len, [], args, tags = tagset, dim=args.n_embd, save_dir=args.save_dir, localfile=os.path.join('/tmp',vort+'.gens.tsv'))

    print('done decoding....')


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
    parser.add_argument('--n_transfer', type=int, default=12)
    parser.add_argument('--lm_coef', type=float, default=0.5)
    parser.add_argument('--b1', type=float, default=0.9)
    parser.add_argument('--b2', type=float, default=0.999)
    parser.add_argument('--e', type=float, default=1e-8)
    parser.add_argument('--load_dir', type=str, default="output", help='directory containing checkpoint_best.pt')
    parser.add_argument('--save_dir', type=str, default="output", help='directory to save generations to')
    parser.add_argument('--data_dir', type=str, default='data', help='directory containing dev/test inputs')
    parser.add_argument('--max_ex', type=int, default=None, help='maximum number of inputs to use, or None for using whole dataset')
    parser.add_argument('--beam', type=int, default=0)
    parser.add_argument('--k', type=int, default=0)
    parser.add_argument('--p', type=int, default=0)
    parser.add_argument('--decoding_strategy', type=int, default=0)
    parser.add_argument('--accum_iter', type=int, default=2)
    parser.add_argument('--gen_len', type=int, default=922)
    parser.add_argument('--n_ctx', type=int, default=102)
    parser.add_argument('--min_len', type=int, default=100)
    parser.add_argument('--repeattheta', type=float, default=1.5)
    parser.add_argument('--show_progress', action='store_true')
    parser.add_argument('--exclude_kw', action='store_true')
    parser.add_argument('--testset', action='store_true', help='if true will generate from test set, if false will generate from dev set')
    parser.add_argument('--memstatesize', type=int, default=100)
    parser.add_argument('--use_model', type=str, choices=['base', 'plotmachines'])
    parser.add_argument('--use_neighbor_feat', action='store_true')
    parser.add_argument('--use_discourse', action='store_true')
    parser.add_argument('--debug_mode', action='store_true')
        #--bodynum determines format of discourse template for output 
        #(for five paragraph format, use 3, because intro and conclusion will be added automatically)
    parser.add_argument('--bodynum', type=int, default=3, help='The number of body pargraphs to use in generation')


    args = parser.parse_args()
    print(torch.__version__)
    print(args)
    main(args)
