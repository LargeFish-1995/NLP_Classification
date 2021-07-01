'''
Each mini-batch returns the followings:
words: list of input sents. ["The 26-year-old ...", ...]
x: encoded input sents. [N, T]. int64.
is_heads: list of head markers. [[1, 1, 0, ...], [...]]
tags: list of tags.['O O B-MISC ...', '...']
y: encoded tags. [N, T]. int64
seqlens: list of seqlens. [45, 49, 10, 50, ...]
'''
import numpy as np
import torch
from torch.utils import data
from pathlib2 import Path
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
# there are 146 tags in total
VOCAB = ('Nx', 'Ec', 'Va', 'ql', 'Vo', 'ln', 'Nn', 'Ac', 'v', 'ud', 'nr', 'c', 'tt', 'Rc', 'nz', 'Ny', 'ns', 'qr', 'ui', 'dc',
'h', 'Ub', 't', 'Vp', 'lv', 'd', 'mq', 'vd', 'Ug', 'Vx', 'la', 'Vi', 'Sy', 'ryw', 'Ad', 'w', 'lb', 'vx', 'Nt', 'l', 'Nl', 'Dc',
'ww', 'As', 'Ng', 'qj', 'wm', 'Fc', 'vi', 'uo', 'wky', 'wt', 'wyz', 'Dg', 'jb', 'ia', 'rr', 'wyy', 'vn', 'ld', 'Mc', 'rzw', 'rz',
 'df', 'ue', 'nt', 'Zc', 'Vc', 'Ag', 'vl', 'Rg', 'i', 'jd', 'j', 'Vg', 'vq', 'id', 'ul', 'Ux', 'qb', 'wu', 'wf', 'Nz', 'nrg',
 'in', 'Ua', 'vu', 'No', 'us', 'Uf', 'Nh', 'm', 'wj', 'z', 'Ns', 'r', 's', 'Qg', 'u', 'Mo', 'Vn', 'ib', 'Ax', 'a', 'An', 'Cc',
 'Qc', 'qe', 'ad', 'jn', 'ws', 'Bg', 'Nc', 'Oc', 'an', 'ry', 'y', 'jv', 'o', 'nrf', 'k', 'Aa', 'Vt', 'qt', 'qc', 'n', 'wd', 'p',
 'Tg', 'nx', 'Vu', 'Pc', 'e', 'wkz', 'iv', 'f', 'qv', 'Us', 'Vd', 'wp', 'b', 'qd', 'Mg', 'q', 'uz', 'qz')

tag2idx = {tag: idx for idx, tag in enumerate(VOCAB)}
idx2tag = {idx: tag for idx, tag in enumerate(VOCAB)}

class NerDataset(data.Dataset):
    def __init__(self, fold_path, train_d = True):
        """
        fpath: [train|valid|test].txt
        """
        sentences, tags_list = [], []
        for f_p in Path(fold_path).iterdir():
            sents, tags_li = [], []
            print(f"handling file: {f_p}")
            with open(f_p,  encoding="utf16") as f:
                all_lines = list(f.readlines())
                words = []
                tags = []
                for index, line in  enumerate(all_lines):
                    if line == '\n' or line == '':
                        continue
                    lines_parts = line.rstrip('\r\n').split('  ')
                    for e_part in lines_parts:
                        if len(e_part) < 2:
                            continue
                        words.append(e_part.split('/')[0])
                        tags.append(e_part.split('/')[1])

                    words_arr = [i for i in words]
                    tags_arr = [i for i in tags]
                    sents.append(["[CLS]"] + words_arr + ["[SEP]"])
                    tags_li.append(["<PAD>"] + tags_arr + ["<PAD>"])
                    words = []
                    tags = []
                    
            if train_d:
                len_d = len(sents) - round(len(sents) * 0.3)
                sents = sents[:len_d]
                tags_li = tags_li[:len_d]
            else:
                len_d = len(sents) - round(len(sents) * 0.3)
                sents = sents[len_d:]
                tags_li = tags_li[len_d:]
            sentences.extend(sents)
            tags_list.extend(tags_li)
            sents, tags_li = [], []

        self.sents, self.tags_li = sentences, tags_list

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, idx):
        words, tags = self.sents[idx], self.tags_li[idx] # words, tags: string list

        # We give credits only to the first piece.
        x, y = [], [] # list of ids
        is_heads = [] # list. 1: the token is the first piece of a word
        for w, t in zip(words, tags):
            tokens = tokenizer.tokenize(w) if w not in ("[CLS]", "[SEP]") else [w]
            xx = tokenizer.convert_tokens_to_ids(tokens)

            is_head = [1] + [0]*(len(tokens) - 1)

            t = [t] + ["<PAD>"] * (len(tokens) - 1)  # <PAD>: no decision
            yy = [tag2idx[each] for each in t]  # (T,)

            x.extend(xx)
            is_heads.extend(is_head)
            y.extend(yy)

        assert len(x)==len(y)==len(is_heads), f"len(x)={len(x)}, len(y)={len(y)}, len(is_heads)={len(is_heads)}"

        # seqlen
        seqlen = len(y)

        # to string
        words = " ".join(words)
        tags = " ".join(tags)
        return words, x, is_heads, tags, y, seqlen


def pad(batch):
    '''Pads to the longest sample'''
    f = lambda x: [sample[x] for sample in batch]
    words = f(0)
    is_heads = f(2)
    tags = f(3)
    seqlens = f(-1)
    maxlen = np.array(seqlens).max()

    f = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch] # 0: <pad>
    x = f(1, maxlen)
    y = f(-2, maxlen)


    f = torch.LongTensor

    return words, f(x), is_heads, tags, f(y), seqlens
