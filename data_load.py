'''
An entry or sent looks like ...

SOCCER NN B-NP O
- : O O
JAPAN NNP B-NP B-LOC
GET VB B-VP O
LUCKY NNP B-NP O
WIN NNP I-NP O
, , O O
CHINA NNP B-NP B-PER
IN IN B-PP O
SURPRISE DT B-NP O
DEFEAT NN I-NP O
. . O O

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

# tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# VOCAB = ('<PAD>', 'O', 'I-LOC', 'B-PER', 'I-PER', 'I-ORG', 'I-MISC', 'B-MISC', 'B-LOC', 'B-ORG')
VOCAB = ('<PAD>', 'O', 'B-Qualifier', 'I-Term-frag', 'B-Ordered-Term', 'I-Definition', 'I-Alias-Term', 'I-Ordered-Term',
        'B-Alias-Term', 'I-Referential-Definition', 'I-Term', 'B-Alias-Term-frag', 'B-Ordered-Definition', 'B-Referential-Definition',
        'B-Definition', 'I-Qualifier', 'I-Definition-frag', 'B-Term', 'I-Secondary-Definition', 'I-Referential-Term', 'B-Definition-frag',
        'I-Ordered-Definition', 'B-Secondary-Definition', 'B-Term-frag', 'B-Referential-Term')
tag2idx = {tag: idx for idx, tag in enumerate(VOCAB)}
idx2tag = {idx: tag for idx, tag in enumerate(VOCAB)}

class NerDataset(data.Dataset):
    def __init__(self, fold_path):
        """
        fpath: [train|valid|test].txt
        """
        # entries = open(fpath, 'r').read().strip().split("\n\n")
        # sents, tags_li = [], [] # list of lists
        # for entry in entries:
        #     words = [line.split()[0] for line in entry.splitlines()]
        #     tags = ([line.split()[-1] for line in entry.splitlines()])
        #     sents.append(["[CLS]"] + words + ["[SEP]"])
        #     tags_li.append(["<PAD>"] + tags + ["<PAD>"])
        sents, tags_li = [], []
        for f_p in Path(fold_path).iterdir():
            if f_p.suffix == '.deft':
                with open(f_p) as f:
                    all_lines = list(f.readlines())
                    words = []
                    tags = []
                    for index, line in enumerate(all_lines):
                        if line == '\n' or index == len(all_lines) - 1:
                            words_arr = [i for i in words]
                            tags_arr = [i for i in tags]
                            sents.append(["[CLS]"] + words_arr + ["[SEP]"])
                            tags_li.append(["<PAD>"] + tags_arr + ["<PAD>"])
                            words = []
                            tags = []
                            continue
                        line_parts = line.split('\t')
                        words.append(line_parts[0].strip(' '))
                        tags.append(line_parts[4].strip(' '))

        self.sents, self.tags_li = sents, tags_li

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
