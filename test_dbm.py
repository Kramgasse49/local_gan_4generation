#!/usr/bin/env python

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

import gan_rbm.Models
from gan_rbm import ModelConstructor
import gan_rbm.opts as opts
from tool.vocab import Vocab 
from modules.utils import use_gpu
from gan_rbm.Trainer import Statistics, GANTrainer
from tool.datasetbase import DataSet 
from beam.Beam import GNMTGlobalScorer
from beam.Translator import Translator

############## 
utf_fn = lambda x: x.encode('utf-8') 

def id2text(idxs, idx2word):
    return [idx2word.get(idx) for idx in idxs]

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
    # build beam generator
    scorer = GNMTGlobalScorer(0, 0)

    translator = Translator(model, vocab,
                            beam_size=opt.beam_size,
                            n_best=opt.n_best,
                            global_scorer=scorer,
                            max_length=opt.max_length,
                            cuda=use_gpu(opt))
    # translate batch 
    for batch in test_dataset:
        batch_result = translator.translate_batch(batch)
        predictions = batch_result['predictions']
        query, query_lens = batch 
        query = query.data.transpose(1,0).numpy()
        sample_num = len(predictions)
        for i in range(sample_num):
            q_ids = query[i].tolist()
            #print("Query:")
            #print(' '.join( map(utf_fn, id2text(q_ids, vocab.idx2word))))
            #out_file.write("query: {}\n".format(' '.join( id2text(q_ids, vocab.idx2word))))
            out_file.write("query: ".decode('utf-8'))
            out_file.write(' '.join( id2text(q_ids, vocab.idx2word)))
            out_file.write('\n'.decode('utf-8'))
            #print("Response:")
            for r_ids in predictions[i]:
                #print(' '.join( map(utf_fn, id2text(r_ids, vocab.idx2word))))
                #out_file.write("response: {}\n".format(' '.join( id2text(r_ids, vocab.idx2word))))
                out_file.write("response: ".decode('utf-8'))
                out_file.write(' '.join( id2text(r_ids, vocab.idx2word)))
                out_file.write('\n'.decode('utf-8'))
        # print(batch_result['scores'])
        # print(batch_result['attention'])
    out_file.close()
    # 
if __name__ == "__main__":
    main()
