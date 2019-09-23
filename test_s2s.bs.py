#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import division, unicode_literals

import argparse
import glob
import os
import sys
import random
import io
from datetime import datetime
import numpy as np

import torch
import torch.nn as nn
from torch import cuda

import generator.Models
from generator import ModelConstructor
import generator.opts as opts
from tool.vocab import Vocab 
from modules.utils import use_gpu
from generator.Trainer import Statistics, DoubleTrainer
from tool.datasetbase import DataSet 

from mybeam.beam import batch_bs
from mybeam.beam_penalize_sibling import batch_bs_pena_sibling
from mybeam.beam_group_diverse import batch_bs_group_diverse
from mybeam.greedy import greedy_decoding
 
############## 
utf_fn = lambda x: x.encode('utf-8') 

def id2text(idxs, idx2word):
    out = [idx2word.get(idx) for idx in idxs]
    return [t for t in out if t is not None]

def text2id(line, word2index, unk_id, max_len=None, pre_trunc=True):
    words = line.strip().split()
    wids = [word2index.get(w, unk_id) for w in words]
    if max_len is None:
        return len(wids), wids
    else:
        if pre_trunc:
            wids = wids[-max_len:]
        else:
            wids = wids[:max_len]
        return len(wids), wids
############

# parse arguments
parser = argparse.ArgumentParser(
    description='train.py',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

opts.model_opts(parser)
opts.train_opts(parser)
opts.test_opts(parser)

opt = parser.parse_args()

opt.brnn = (opt.encoder_type == "brnn")
if opt.seed > 0:
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)

if torch.cuda.is_available() and not opt.gpuid:
    print("WARNING: You have a CUDA device, should run with -gpuid 0")

if opt.gpuid:
    cuda.set_device(opt.gpuid[0])
    if opt.seed > 0:
        torch.cuda.manual_seed(opt.seed)

# modify vocab size according to the real vocab
vocab = torch.load(opt.vocab_path)
# set padding_word number 
opt.padding_idx = 0
opt.numwords = len(vocab.word2idx)
opt.enc_numwords = opt.numwords
opt.dec_numwords = opt.numwords
try:
    unk_ids = vocab.word2idx['<unk>']
except:
    unk_ids = 0

# several automatic evaluation approach
def _report_score(name, score_total, words_total):
    print("%s AVG SCORE: %.4f, %s PPL: %.4f" % (
        name, score_total / words_total,
        name, math.exp(-score_total / words_total)))


def _report_bleu():
    import subprocess
    print()
    res = subprocess.check_output(
        "perl tools/multi-bleu.perl %s < %s" % (opt.tgt, opt.output),
        shell=True).decode("utf-8")
    print(">> " + res.strip())


def _report_rouge():
    import subprocess
    res = subprocess.check_output(
        "python tools/test_rouge.py -r %s -c %s" % (opt.tgt, opt.output),
        shell=True).decode("utf-8")
    print(res.strip())


def build_test_dataset(test_query_path):
    """
       This function is solely to handle with small datasets.
    """
    examples = []
    with io.open(test_query_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            txt_len, txt = text2id(line, vocab.word2idx, unk_ids, 
                    max_len=opt.seq_len, pre_trunc=True)
            examples.append([txt_len, txt])
    return DataSet(examples)

def main():
    # Load the model.
    model = ModelConstructor.load_test_model(opt)

    # File to write sentences to.
    out_file = io.open(opt.test_output, 'w', encoding="utf-8")

    # test file info
    test_query_path = opt.test_corpus_path[0]
    if len(opt.test_corpus_path) > 2:
        test_response_path = opt.test_corpus_path[1]
    else:
        test_response_path = None 
    # 
    test_dataset = build_test_dataset(test_query_path)
    test_dataset.set_property(batch_size=opt.batch_size, 
                              with_label=False, rank=False)

    for batch in test_dataset:
        query, query_lens = batch
        query, query_lens = ( query.cuda(), query_lens.cuda() )
        if opt.beam_type == "general":
            cands, scores = batch_bs(model, query, query_lens, beam_k=opt.beam_size, max_len=opt.max_length)
        elif opt.beam_type == "greedy":
            cands, scores = greedy_decoding(model, query, query_lens, start_width=opt.beam_size, max_len=opt.max_length)
        elif opt.beam_type == "group":
            cands, scores = batch_bs_pena_sibling(model, query, query_lens, beam_k=opt.beam_size, max_len=opt.mat_length)
        else:
            raise ValueError("Beam Type {} does not exist".format(opt.beam_type))
        query, query_lens = batch 
        query = query.data.transpose(1,0).numpy()
        sample_num, _ = query.shape
        for i in range(sample_num):
            q_ids = query[i].tolist()
            try:
                end_index = q_ids.index(2)
                q_ids = q_ids[:end_index]
            except:
                pass
            out_file.write("query: ".decode('utf-8'))
            out_file.write(' '.join( id2text(q_ids, vocab.idx2word)))
            out_file.write('\n'.decode('utf-8'))
            for r_ids in cands[i]:
                out_file.write("response: ".decode('utf-8'))
                out_file.write(' '.join( id2text(r_ids, vocab.idx2word)))
                out_file.write('\n'.decode('utf-8'))
    out_file.close()
    # 
if __name__ == "__main__":
    main()
