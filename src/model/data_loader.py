import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import json
from pytorch_transformers import *
import pickle


class ParagraphDataset(Dataset):
    def __init__(self, data_file, encoder, max_size=None, n_ctx=102, n_gen=401, include_neigh=False, include_curr=False,
                 include_discourse_type=True, include_kw=True, dim=0 ,debug_mode=False):
        with open(data_file, "rb") as f:
            self.data = f.readlines()

        if include_neigh:
            self.prev = []
            fn = ".".join(data_file.split(".")[:-1]) + "_gpt2.pkl"
            if debug_mode:
                fn = ".".join(data_file.split(".")[:-1]) + "_gpt.pkl"
            with open(fn, 'rb') as fp:
                for k in range(len(self.data)):
                    temp = pickle.load(fp)
                    assert temp[0] == k and temp[1] == self.data[k].decode('utf-8', 'ignore').split("\t")[-1].replace(
                        "<o>", "").strip()
                    self.prev.append(temp[2])
        else:
            self.prev = None

        self.dids = []
        for d in range(1, len(self.data)):
            t = self.data[d].decode("utf-8", "ignore").strip().split('\t')
            if len(t) == 7 and t[5].replace("<o>", "").strip() != "":
                try:
                    x, y = int(t[0].split("_")[-1]), int(t[4])
                    self.dids.append(d)
                except:
                    pass

        if max_size is not None:
            self.dids = self.dids[:max_size]
        self.encoder = encoder
        self.ctx = n_ctx - 2
        self.gen = n_gen - 1
        self.dim = dim
        self.len = len(self.data)
        self.include_neigh = include_neigh
        self.include_curr = include_curr
        self.include_discourse_type = include_discourse_type
        self.include_kw = include_kw

    '''def bertrep(self,textline):
        nb = 1
        #if nb < 8:
        wds = torch.zeros(1, 512, dtype=torch.long) + self.tokenizer.convert_tokens_to_ids(['[PAD]'])[0]
        temp = torch.tensor(self.tokenizer.convert_tokens_to_ids(["[CLS]"]+ self.tokenizer.tokenize(textline)[:500] + ["[SEP]"]))
        wds[0,:len(temp)] = temp #temp.cuda()
        self.model.eval()
        outputs = self.model(wds)
        clfone = outputs[1].detach() #.cpu()
        return clfone.squeeze()'''

    def __getitem__(self, index):
        idx = self.dids[index]
        csv_data = self.data[idx].decode("utf-8", "ignore").strip().split('\t')
        kws = csv_data[2].split("[SEP]")
        # print(self.encoder.encode(csv_data[5]))
        tgt_phrase = self.encoder.encode(csv_data[5].replace("<o>", ""), add_prefix_space=True, add_special_tokens=False)[:self.gen]
        start = torch.LongTensor([self.encoder.bos_token_id])
        clstok = torch.LongTensor([self.encoder.cls_token_id])
        end = torch.LongTensor([self.encoder.eos_token_id])
        tstart = torch.LongTensor([self.encoder.convert_tokens_to_ids('_t_')])
        istart = torch.LongTensor([self.encoder.convert_tokens_to_ids('_i_')])
        bstart = torch.LongTensor([self.encoder.convert_tokens_to_ids('_b_')])
        cstart = torch.LongTensor([self.encoder.convert_tokens_to_ids('_c_')])
        keytok = torch.LongTensor([self.encoder.convert_tokens_to_ids('_kw_')])
        endkeytok = torch.LongTensor([self.encoder.convert_tokens_to_ids('_endkw_')])
        getstart = {"I": istart,
                    "B": bstart,
                    "C": cstart,
                    "T": tstart,
                    }

        if self.include_discourse_type:
            starttyptok = bstart
            if int(csv_data[0].split("_")[-1]) == 0:
                starttyptok = istart
            elif int(csv_data[0].split("_")[-1]) == int(csv_data[4]) - 1:
                starttyptok = cstart
        else:
            starttyptok = clstok

        pad_output = torch.zeros(self.ctx + self.gen + 3).long()
        mask_output = torch.zeros(self.ctx + self.gen + 3).long()

        # print(tgt_phrase)
        # Tokens
        pad_output[0] = start

        if self.include_kw:
            i = 1
            for k in kws:
                if i - 1 >= self.ctx:
                    break
                enck = self.encoder.encode(k.strip(), add_prefix_space=True, add_special_tokens=False)[:self.ctx - i]
                # print(enck, i)
                pad_output[i:i + len(enck)] = torch.LongTensor(enck)
                pad_output[i + len(enck)] = keytok
                i += len(enck) + 1
            pad_output[i - 1] = endkeytok
            mask_output[0:i] = torch.ones(i).long()

        pad_output[self.ctx + 1] = starttyptok if self.include_discourse_type else clstok
        pad_output[self.ctx + 1 + 1:self.ctx + 1 + 1 + len(tgt_phrase)] = torch.LongTensor(tgt_phrase)
        pad_output[self.ctx + 1 + 1 + len(tgt_phrase)] = end

        # Mask
        mask_output[self.ctx + 1:self.ctx + 1 + len(tgt_phrase) + 2] = torch.ones(len(tgt_phrase) + 2).long()

        if self.include_neigh:
            # n = self.bertrep(csv_data[-1].replace("<o>",""))
            n = torch.FloatTensor(self.prev[idx].flatten())
        else:
            n = torch.zeros(self.dim, dtype=torch.float64)
        if self.include_curr:
            c = torch.FloatTensor(self.curr[idx].flatten())
        else:
            c = torch.zeros(self.dim, dtype=torch.float64)
        return pad_output, mask_output, n, c

    def __len__(self):
        return len(self.dids)


def get_paragraph_input_loader(data_file, batch_size, encoder, shuffle=True, num_workers=0, max_size=None, n_ctx=102,
                               gen_len=401, include_neigh=False, include_discourse_type=True, include_kw=True,
                               include_curr=False, dim=768, debug_mode=False):
    dataset = ParagraphDataset(data_file, encoder, max_size=max_size, n_ctx=n_ctx, n_gen=gen_len,
                               include_neigh=include_neigh, include_curr=include_curr,
                               include_discourse_type=include_discourse_type, include_kw=include_kw, dim=dim,debug_mode=debug_mode)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=True)


class DocumentDataset(Dataset):
    def __init__(self, data_file, encoder, max_size=None, n_ctx=102, n_gen=401, include_kw=True):
        self.data = []

        with open(data_file, "rb") as f:
            data = f.readlines()

        self.encoder = encoder
        self.ctx = n_ctx - 2
        self.gen = n_gen - 1
        self.dids = []

        for d in range(1, len(data)):
            t = data[d].decode("utf-8", "ignore").strip().split('\t')
            newinput = t[0].split("_")[0] + "\t" + t[2]
            if not (newinput in self.dids) and len(t) == 7:
                self.dids.append(newinput)

        if max_size is not None:
            self.dids = self.dids[:max_size]
        self.len = len(self.dids)
        self.include_kw = include_kw

    def __getitem__(self, index):
        csv_data = self.dids[index].split('\t')
        kws = csv_data[1].split("[SEP]")

        tgt_phrase = []
        start = torch.LongTensor([self.encoder.bos_token_id])
        clstok = torch.LongTensor([self.encoder.cls_token_id])
        end = torch.LongTensor([self.encoder.eos_token_id])
        tstart = torch.LongTensor([self.encoder.convert_tokens_to_ids('_t_')])
        istart = torch.LongTensor([self.encoder.convert_tokens_to_ids('_i_')])
        bstart = torch.LongTensor([self.encoder.convert_tokens_to_ids('_b_')])
        cstart = torch.LongTensor([self.encoder.convert_tokens_to_ids('_c_')])
        keytok = torch.LongTensor([self.encoder.convert_tokens_to_ids('_kw_')])
        endkeytok = torch.LongTensor([self.encoder.convert_tokens_to_ids('_endkw_')])

        pad_output = torch.zeros(self.ctx + self.gen + 3).long()
        mask_output = torch.zeros(self.ctx + self.gen + 3).long()

        # print(tgt_phrase)
        # Tokens
        pad_output[0] = start

        if self.include_kw:
            i = 1
            for k in kws:
                if i - 1 >= self.ctx:
                    break
                enck = self.encoder.encode(k.strip(), add_prefix_space=True, add_special_tokens=False)[:self.ctx - i]
                # print(enck, i)
                pad_output[i:i + len(enck)] = torch.LongTensor(enck)
                pad_output[i + len(enck)] = keytok
                i += len(enck) + 1
            pad_output[i - 1] = endkeytok
            mask_output[0:i] = torch.ones(i).long()

        pad_output[self.ctx + 1] = clstok
        pad_output[self.ctx + 1 + 1:self.ctx + 1 + 1 + len(tgt_phrase)] = torch.LongTensor(tgt_phrase)
        pad_output[self.ctx + 1 + 1 + len(tgt_phrase)] = end

        # Mask
        mask_output[self.ctx + 1:self.ctx + 1 + len(tgt_phrase) + 2] = torch.ones(len(tgt_phrase) + 2).long()

        ids = csv_data + [index]
        return pad_output, mask_output, ids

    def __len__(self):
        return len(self.dids)


def get_document_full_loader(data_file, batch_size, encoder, shuffle=True, num_workers=0, max_size=None, n_ctx=102,
                             gen_len=401, include_kw=True):
    dataset = DocumentDataset(data_file, encoder, max_size=max_size, n_ctx=n_ctx, n_gen=gen_len, include_kw=include_kw)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=True)


class ParagraphWithHistoryDataset(Dataset):
    def __init__(self, data_file, encoder, max_size=None, n_ctx=102, n_gen=401, include_discourse_type=True,
                 include_kw=True, memsize=10, dim=768, use_kwmem=False, debug_mode=False):

        def isClean(line):
            chunks = line.decode('utf-8', 'ignore').strip().split('\t')
            if len(chunks) < 7:
                return False

            keys = chunks[2]
            par = chunks[5]
            par_prev = chunks[6].strip()

            if len(keys) < 5 or len(par) < 5 or (par_prev != 'NA' and len(par_prev) < 5):  # or len(par_prev):
                return False
            return True

        with open(data_file, "rb") as f:
            self.data = f.readlines()
        temp_data = []

        self.prevmat = []

        fn = ".".join(data_file.split(".")[:-1]) + "_gpt2.pkl"
        if debug_mode:
            fn = ".".join(data_file.split(".")[:-1]) + "_gpt.pkl"
        with open(fn, 'rb') as fp:
            for k in range(len(self.data)):
                temp = pickle.load(fp)
                if k == 0:
                    continue

                if not isClean(self.data[k]):
                    continue

                if temp[0] != k or temp[1] != self.data[k].decode('utf-8', 'ignore').split("\t")[-1].replace("<o>","").strip():
                    print(str(temp[0]))
                    print(str(k))
                    print(temp[1])
                    print(self.data[k].decode('utf-8', 'ignore').split("\t")[-1].replace("<o>","").strip())
                    continue

                temp_data.append(self.data[k])
                # if len(self.data[k].decode('utf-8', 'ignore').split("\t"))==0 or len(temp)<2:
                #     continue
                # assert temp[0] == k and temp[1] == self.data[k].decode('utf-8', 'ignore').split("\t")[-1].replace("<o>","").strip()
                self.prevmat.append(temp[2])

        self.data = temp_data
        temp_data = []

        print('i read so many of ... ' + str(len(self.data)))
        assert len(self.prevmat) == len(self.data)

        self.dids = []
        self.history = dict()
        self.h = 10
        for d in range(1, len(self.data)):
            t = self.data[d].decode("utf-8", "ignore").strip().split('\t')
            docid = t[0].split("_")[0]
            if len(t) == 7 and t[5].replace("<o>", "").strip() != "":
                try:
                    x, y = int(t[0].split("_")[-1]), int(t[4])
                except:
                    continue
                self.dids.append(d)
                if docid not in self.history:
                    self.history[docid] = dict()
                self.history[docid][x] = self.prevmat[d]  ##t[5].replace("<o>","")
        # print(len(self.history))

        if max_size is not None:
            self.dids = self.dids[:max_size]
        self.encoder = encoder
        self.ctx = n_ctx - 2
        self.gen = n_gen - 1
        self.memsize = memsize
        self.len = len(self.data)
        # self.include_neigh= include_neigh
        # self.include_curr= include_curr
        self.include_discourse_type = include_discourse_type
        self.include_kw = include_kw
        self.h = 10
        self.dim = dim
        # asli
        self.use_kwmem = use_kwmem

    def __getitem__(self, index):
        idx = self.dids[index]
        csv_data = self.data[idx].decode("utf-8", "ignore").strip().split('\t')
        kws = csv_data[2].split("[SEP]")

        tgt_phrase = self.encoder.encode(csv_data[5].replace("<o>", ""), add_prefix_space=True, add_special_tokens=False)[:self.gen]
        start = torch.LongTensor([self.encoder.bos_token_id])
        clstok = torch.LongTensor([self.encoder.cls_token_id])
        end = torch.LongTensor([self.encoder.eos_token_id])
        tstart = torch.LongTensor([self.encoder.convert_tokens_to_ids('_t_')])
        istart = torch.LongTensor([self.encoder.convert_tokens_to_ids('_i_')])
        bstart = torch.LongTensor([self.encoder.convert_tokens_to_ids('_b_')])
        cstart = torch.LongTensor([self.encoder.convert_tokens_to_ids('_c_')])
        keytok = torch.LongTensor([self.encoder.convert_tokens_to_ids('_kw_')])
        endkeytok = torch.LongTensor([self.encoder.convert_tokens_to_ids('_endkw_')])
        getstart = {"I": istart,
                    "B": bstart,
                    "C": cstart,
                    "T": tstart,
                    }

        if self.include_discourse_type:
            starttyptok = bstart
            if int(csv_data[0].split("_")[-1]) == 0:
                starttyptok = istart
            elif int(csv_data[0].split("_")[-1]) == int(csv_data[4]) - 1:
                starttyptok = cstart
        else:
            starttyptok = clstok

        pad_output = torch.zeros(self.ctx + self.gen + 3).long()
        mask_output = torch.zeros(self.ctx + self.gen + 3).long()

        if self.use_kwmem:
            mem = torch.torch.empty(self.ctx + self.memsize, self.dim).normal_(std=.02)
            mmask = torch.zeros(self.ctx + self.memsize).long()
        else:
            mem = torch.torch.empty(self.memsize, self.dim).normal_(std=.02)
            mmask = torch.zeros(self.memsize).long()

        pad_output[0] = start

        if self.include_kw:
        ##if self.use_kwmem:
            i = 1
            for k in kws:
                if i - 1 >= self.ctx:
                    break
                enck = self.encoder.encode(k.strip(), add_prefix_space=True, add_special_tokens=False)[:self.ctx - i]
                # print(enck, i)
                pad_output[i:i + len(enck)] = torch.LongTensor(enck)
                pad_output[i + len(enck)] = keytok
                i += len(enck) + 1
            pad_output[i - 1] = endkeytok
            mask_output[0:i] = torch.ones(i).long()

            # mem[0:i-1,0] = pad_output[1:i, 0]
            if self.use_kwmem:
                mmask[0:i - 1] = torch.ones(i - 1).long()
            mmask[-self.memsize:] = torch.ones(self.memsize).long()

        pad_output[self.ctx + 1] = starttyptok if self.include_discourse_type else clstok
        pad_output[self.ctx + 1 + 1:self.ctx + 1 + 1 + len(tgt_phrase)] = torch.LongTensor(tgt_phrase)
        pad_output[self.ctx + 1 + 1 + len(tgt_phrase)] = end

        # Mask
        mask_output[self.ctx + 1:self.ctx + 1 + len(tgt_phrase) + 2] = torch.ones(len(tgt_phrase) + 2).long()

        prev = torch.zeros(self.h, 1, self.dim).float()  # .long()
        pmask = torch.zeros(self.h, 1).long()
        docid = csv_data[0].split("_")[0]
        pid = int(csv_data[0].split("_")[-1])

        for p in range(1, min(pid + 1, self.h + 1)):
            # p = 1 --> pid+1
            if self.history[docid].get(p) is None:
                continue
            try:
                prev[p - 1, 0, :] = torch.LongTensor(self.history[docid][p])
                pmask[p - 1, 0] = torch.ones(1).long()
            except:
                continue
        return pad_output, mask_output, mem, mmask, prev, pmask, torch.FloatTensor(self.prevmat[idx].flatten())

    def __len__(self):
        return len(self.dids)


def get_paragraph_history_input_loader(data_file, batch_size, encoder, shuffle=True, num_workers=0, max_size=None,
                                       n_ctx=102, gen_len=401, include_neigh=False, include_discourse_type=True,
                                       include_kw=True, include_curr=False, memsize=10, dim=768, use_kwmem=False, debug_mode=False):
    dataset = ParagraphWithHistoryDataset(data_file, encoder, max_size=max_size, n_ctx=n_ctx, n_gen=gen_len,
                                          include_discourse_type=include_discourse_type, include_kw=include_kw,
                                          memsize=memsize, dim=dim, use_kwmem=use_kwmem, debug_mode=debug_mode)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=True)
