# -*- coding:utf-8 -*- 
"""
Test a pre-trained text-rbm on a text classification task

"""
from __future__ import unicode_literals, print_function, division

import os, sys, time, logging
import pickle
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from torch.nn.init import xavier_uniform

from dbm.w2v_load import load_glove_txt
from dbm.data import load_vocab, batcher, build_vocab
import dbm.utils as utils

from normalization.scale_data import MinMaxScaler
from normalization.scale_data import StandardScaler 

parser = argparse.ArgumentParser('pretrain')
utils.common_opt(parser)
utils.data_opt(parser)
args = parser.parse_args()

# load vocab
vocab_src = torch.load(args.vocab_file)
vocab, rev_vocab = vocab_src.word2idx, vocab_src.idx2word
args.vocab_size = len(vocab)

# load embedding
embeddings = nn.Embedding(args.vocab_size, args.embedding_dim,
                          padding_idx=utils.PAD_ID)

print("############################Loading Embedding##########################")
# rev_vocab, 
not_in_glove = 0
word_embeddings = None

if args.embedfile is not None:
    word_embedding = np.load(args.embedfile)
    embeddings.weight.data.copy_(torch.from_numpy(word_embedding))
if args.use_cuda:
    embeddings = embeddings.cuda()

def query_generator():
    train_batcher = batcher(args.batch_size, args.train_query_file, \
                            args.train_response_file)
    while True:
        try:
            batch = train_batcher.next()
        except StopIteration:
            break
        (query_src, query_len, reply_src, reply_len, _, _) = utils.get_variables( 
                    batch, vocab, args.dec_max_len, use_cuda=args.use_cuda)
        q_avg, r_avg = get_avg_embedding(query_src, query_len, reply_src, reply_len)
        yield q_avg

def reply_generator():
    train_batcher = batcher(args.batch_size, args.train_query_file,
                            args.train_response_file)
    while True:
        try:
            batch = train_batcher.next()
        except StopIteration:
            break
        (query_src, query_len, reply_src, reply_len, _, _) = utils.get_variables(
                    batch, vocab, args.dec_max_len, use_cuda=args.use_cuda)
        q_avg, r_avg = get_avg_embedding(query_src, query_len, reply_src, reply_len)
        yield r_avg

def qreply_generator():
    train_batcher = batcher(args.batch_size, args.train_query_file,
                            args.train_response_file)
    while True:
        try:
            batch = train_batcher.next()
        except StopIteration:
            break
        (query_src, query_len, reply_src, reply_len, _, _) = utils.get_variables(
                    batch, vocab, args.dec_max_len, use_cuda=args.use_cuda)
        q_avg, r_avg = get_avg_embedding(query_src, query_len, reply_src, reply_len)
        yield q_avg
        yield r_avg

def train_norm():#(cls, optimizer, criterion, args):
    """
    """
    print("MinMax Normalizations  ........................................................")
    q_minmax_scaler = MinMaxScaler()
    r_minmax_scaler = MinMaxScaler()
    #qr_minmax_scaler = MinMaxScaler()

    start_time = time.time()
    q_generator = query_generator()
    q_minmax_scaler.fit(q_generator)
    print("Query MinMax", time.time() - start_time)
    start_time = time.time()
    r_generator = reply_generator()
    r_minmax_scaler.fit(r_generator)
    print("Reply MinMax", time.time() - start_time)
    # save
    torch.save(q_minmax_scaler, os.path.join(args.data_name, 'q_minmax.pt'))
    torch.save(r_minmax_scaler, os.path.join(args.data_name, 'r_minmax.pt'))

    keys = q_minmax_scaler.__dict__.keys()
    for key in keys:
        print("Q: ", key, getattr(q_minmax_scaler, key))
        print("R: ", key, getattr(r_minmax_scaler, key))

    print("Standard Normalizations  ........................................................")
    q_zscore_scaler = StandardScaler()
    r_zscore_scaler = StandardScaler()

    start_time = time.time()
    q_generator = query_generator()
    q_zscore_scaler.fit(q_generator)
    print("Query Zscore", time.time() - start_time)
    start_time = time.time()
    r_generator = reply_generator()
    r_zscore_scaler.fit(r_generator)
    print("Reply Zscore", time.time() - start_time)

    torch.save(q_zscore_scaler, os.path.join(args.data_name, 'q_zscore.pt'))
    torch.save(r_zscore_scaler, os.path.join(args.data_name, 'r_zscore.pt'))

    # display
    keys = q_zscore_scaler.__dict__.keys()
    for key in keys:
        print("Q: ", key, getattr(q_zscore_scaler, key))
        print("R: ", key, getattr(r_zscore_scaler, key))

    # return q_minmax_scaler, r_minmax_scaler, qr_minmax_scaler

def query_scale_generator(scaler):
    train_batcher = batcher(args.batch_size, args.train_query_file, \
                            args.train_response_file)
    while True:
        try:
            batch = train_batcher.next()
        except StopIteration:
            break
        (query_src, query_len, reply_src, reply_len, _, _) = utils.get_variables( 
                    batch, vocab, args.dec_max_len, use_cuda=args.use_cuda)
        q_avg, r_avg = get_avg_embedding(query_src, query_len, reply_src, reply_len)
        q_scaled = scaler.transform(q_avg)
        yield q_scaled
def reply_scale_generator(scaler):
    train_batcher = batcher(args.batch_size, args.train_query_file, \
                            args.train_response_file)
    while True:
        try:
            batch = train_batcher.next()
        except StopIteration:
            break
        (query_src, query_len, reply_src, reply_len, _, _) = utils.get_variables( 
                    batch, vocab, args.dec_max_len, use_cuda=args.use_cuda)
        q_avg, r_avg = get_avg_embedding(query_src, query_len, reply_src, reply_len)
        r_scaled = scaler.transform(r_avg)
        yield r_scaled

def reply_scale_generator(scaler):
    train_batcher = batcher(args.batch_size, args.train_query_file, \
                            args.train_response_file)
    while True:
        try:
            batch = train_batcher.next()
        except StopIteration:
            break
        (query_src, query_len, reply_src, reply_len, _, _) = utils.get_variables( 
                    batch, vocab, args.dec_max_len, use_cuda=args.use_cuda)
        q_avg, r_avg = get_avg_embedding(query_src, query_len, reply_src, reply_len)
        r_scaled = scaler.transform(r_avg)
        yield r_scaled


def test_zscore():#(cls, optimizer, criterion, args):
    """
    """
    print("Normalizations  ........................................................")
    q_zscore_scaler = torch.load(os.path.join(args.data_name, 'q_zscore.pt')) 
    r_zscore_scaler = torch.load(os.path.join(args.data_name, 'r_zscore.pt')) 
    q_minmax_scaler = MinMaxScaler()
    r_minmax_scaler = MinMaxScaler()

    # transform 
    q_scale_g = query_scale_generator(q_zscore_scaler)
    r_scale_g = reply_scale_generator(r_zscore_scaler)
    q_minmax_scaler.fit(q_scale_g)
    r_minmax_scaler.fit(r_scale_g)
 
    keys = q_minmax_scaler.__dict__.keys()
    for key in keys:
        print("Q: ", key, getattr(q_minmax_scaler, key))
        print("R: ", key, getattr(r_minmax_scaler, key))

    torch.save(q_zscore_scaler, os.path.join(args.data_name, 'q_zscore_minmax.pt'))
    torch.save(r_zscore_scaler, os.path.join(args.data_name, 'r_zscore_minmax.pt'))

def get_avg_embedding(query_src, query_len, reply_src, reply_len):         
    # get average embedding of both query and response
    query_embed = embeddings(query_src)
    reply_embed = embeddings(reply_src)
    query_len = torch.FloatTensor(query_len.view(-1, 1).numpy())
    reply_len = torch.FloatTensor(reply_len.view(-1, 1).numpy())
    if args.use_cuda:
        query_len, reply_len = ( query_len.cuda(), reply_len.cuda() )
    batch_size, _, embed_dim = query_embed.size()
    if args.use_cuda:
        query_avg_embed = torch.sum(query_embed, dim=1) / Variable(query_len.expand(batch_size, embed_dim))
        reply_avg_embed = torch.sum(reply_embed, dim=1) / Variable(reply_len.expand(batch_size, embed_dim))
    else:
        query_avg_embed = torch.sum(query_embed, dim=1) / query_len.expand(batch_size, embed_dim)
        reply_avg_embed = torch.sum(reply_embed, dim=1) / reply_len.expand(batch_size, embed_dim)
    # obtain latent representation, DBM hidden layer
    if args.use_cuda:
        return query_avg_embed.data.cpu().numpy(), reply_avg_embed.data.cpu().numpy()
    else:
        return query_avg_embed.data.numpy(), reply_avg_embed.data.numpy()
#######################################Training##################################

train_norm()
test_zscore()
