import sys
import pandas as pd
import os
import csv
from pathlib2 import Path

fold_path = "/mnt/lustre/yankun/temp/homework/deft_corpus/data/deft_files/train"
# fold_path = "/mnt/lustre/yankun/temp/homework/deft_corpus/data/deft_files/dev"

sents, tags_li = [], []
all_tags = set()
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
                all_tags.add(line_parts[4].strip(' '))
