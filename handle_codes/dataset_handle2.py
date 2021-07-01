import sys
import pandas as pd
import os
import csv
from pathlib2 import Path

fold_path = "/mnt/lustre/yankun/temp/homework/data/"


sents, tags_li = [], []

all_tags = set()

for f_p in Path(fold_path).iterdir():
    print(f"handling file: {f_p}")
    with open(f_p,  encoding="utf16") as f:
        all_lines = list(f.readlines())
        words = []
        tags = []
        for index, line in  enumerate(all_lines):
            if line == '\n' or index == len(all_lines) - 1:
                words_arr = [i for i in words]
                tags_arr = [i for i in tags]
                sents.append(["[CLS]"] + words_arr + ["[SEP]"])
                tags_li.append(["<PAD>"] + tags_arr + ["<PAD>"])
                words = []
                tags = []
                continue
            lines_parts = line.rstrip('\r\n').split(' ')
            for e_part in lines_parts:
                if len(e_part) < 2:
                    continue
                words.append(e_part.split('/')[0])
                tags.append(e_part.split('/')[1])
                all_tags.add(e_part.split('/')[1])

print(f'{len(all_tags)} tags all')
print(all_tags)

import pdb; pdb.set_trace()
