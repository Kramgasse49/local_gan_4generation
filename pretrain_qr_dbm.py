# -*- coding:utf-8 -*- 
"""Train q->r rbm
"""
from __future__ import unicode_literals, print_function, division

import os, sys, time, logging
import pickle
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
from dbm import tqdm_logging
from dbm.w2v_load import load_glove_txt
from dbm.data import load_vocab, batcher, build_vocab
from dbm.text_rbm import TRBM 
import dbm.utils as utils

logger = logging.getLogger("TextRBM")

# set up the logger

parser = argparse.ArgumentParser('pretrain')
utils.common_opt(parser)
utils.data_opt(parser)
args = parser.parse_args()

 # set up the output directory
exp_dir = os.path.join('exp/rbm', args.data_name,
                      args.mode, time.strftime("%Y-%m-%d-%H-%M-%S"))
os.makedirs(exp_dir)
args.exp_dir = exp_dir

tqdm_logging.config(logger, os.path.join(exp_dir, '%s.log' % args.mode),
                        mode='w', silent=False, debug=True)

# load vocab
#vocab, rev_vocab = load_vocab(args.vocab_file, max_vocab=args.vocab_size)
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

# build RBM model
if args.use_cuda:
    embeddings = embeddings.cuda()
q_size = args.embedding_dim
r_size = args.embedding_dim
h_size = 2*args.hidden_dim
rq_rbm = TRBM(q_size, r_size, h_size, use_cuda=args.use_cuda,
                       lr=0.0001, monument=0.5, weight_decay=1e-4) # r->q
qr_rbm = TRBM(r_size, q_size, h_size, use_cuda=args.use_cuda,
                       lr=0.0001, monument=0.5, weight_decay=1e-4) # q->r

for p in rq_rbm.parameters:
    print(type(p), p.size())
for p in qr_rbm.parameters:
    print(type(p), p.size())

################################# Loading Normalizer###################
print("Standard Normalization for RBM Training")
q_normalizer = torch.load(os.path.join(args.data_name, 'q_zscore.pt'))
r_normalizer = torch.load(os.path.join(args.data_name, 'r_zscore.pt'))
#print("MinMax Normalization for RBM Training")
#q_normalizer = torch.load(os.path.join(args.data_name, 'q_minmax.pt'))
#r_normalizer = torch.load(os.path.join(args.data_name, 'r_minmax.pt'))

def norm_data(posts_var, posts_len, reply_var, reply_len):
    # check batcher, and padding 
    post_embed = embeddings(posts_var)
    reply_embed = embeddings(reply_var)
     
    posts_len = torch.FloatTensor(posts_len.view(-1, 1).numpy())
    reply_len = torch.FloatTensor(reply_len.view(-1, 1).numpy())
    if args.use_cuda:
        posts_len = posts_len.cuda()
        reply_len = reply_len.cuda()
    # average embedding as utterance embedding
    batch_size, _, h_dim = post_embed.data.size()
    q_pos = torch.sum(post_embed.data, dim=1) / posts_len.expand(batch_size, h_dim) 
    r_pos = torch.sum(reply_embed.data, dim=1) / reply_len.expand(batch_size, h_dim) 
    # normalization
    q_pos_norm = torch.from_numpy(q_normalizer.transform(q_pos.cpu().numpy()))
    r_pos_norm = torch.from_numpy(r_normalizer.transform(r_pos.cpu().numpy()))
    if args.use_cuda:
        q_pos_norm, r_pos_norm = (q_pos_norm.cuda(), r_pos_norm.cuda())
    return q_pos_norm, r_pos_norm

# training 
def eval_model(epoch):
    rq_fe = []
    qr_fe = []
    total_case = 0
    cur_time = time.time()
    rq_recon = []
    qr_recon = []
    valid_batcher = batcher(args.batch_size, args.valid_query_file,
                        args.valid_response_file)
    while True:
        try:
            batch = valid_batcher.next()
        except:
            break
        posts_var, posts_len, response_var, response_len, _, _= utils.get_variables(batch,
                                      vocab, args.dec_max_len, use_cuda=args.use_cuda)
        # check batcher, and padding 
        q_pos, r_pos = norm_data(posts_var, posts_len, response_var, response_len)
        
        rq_recon_error = rq_rbm.valid_op(q_pos, r_pos)
        qr_recon_error = qr_rbm.valid_op(r_pos, q_pos)
        rq_recon.append(rq_recon_error)
        qr_recon.append(qr_recon_error)

        rq_fe.append( rq_rbm.hr_free_energy(q_pos, r_pos))
        qr_fe.append( qr_rbm.hr_free_energy(r_pos, q_pos))

        total_case += len(batch[0])
    rq_error = zip(*rq_recon)
    qr_error = zip(*qr_recon)
    batch_num = len(rq_recon)
    logger.info("Valid recon error: %.4f, %.4f, %.4f %.4f" % (sum(rq_error[1])/batch_num, 
        sum(rq_error[2])/batch_num, sum(qr_error[1])/batch_num, sum(qr_error[2])/batch_num))
    logger.info('Valid batch %d: average QR FE %.4f, RQ FE %.4f, QR, FEgap: %.4f, %4.f, RQ FEGap: %.4f, %.4f; (%.1f case/sec)' % (
                epoch, sum(qr_fe) / len(qr_fe), sum(rq_fe) / len(rq_fe),
                (sum(qr_fe)-sum(qr_error[4])) /len(qr_fe), (sum(qr_fe)-sum(qr_error[5]))/len(qr_fe),
                (sum(rq_fe)-sum(rq_error[4])) /len(rq_fe), (sum(rq_fe)-sum(rq_error[5]))/len(rq_fe),
                 total_case/(time.time()-cur_time)))
    logger.info('') 

eval_model(-1)
for e in range(args.num_epoch):
    logger.info('---------------------training--------------------------')
    logger.info("Epoch: %d/%d" % (e, args.num_epoch))
    print("Epoch: %d/%d" % (e, args.num_epoch))
    
    if e > 5:
        rq_rbm.monument = 0.8
        qr_rbm.monument = 0.8
    train_batcher = batcher(args.batch_size, args.train_query_file,
                            args.train_response_file)
    # train state 
    rq_fe = []
    qr_fe = []
    rq_recon = []
    qr_recon = []
    total_case = 0
    cur_time = time.time()
    step = 0

    while True:
        try:
            batch = train_batcher.next()
        except StopIteration:
            break
        # train_opt, 
        posts_var, posts_len, response_var, response_len, _, _= utils.get_variables(batch,
                                          vocab, args.dec_max_len, use_cuda=args.use_cuda)
        q_pos, r_pos = norm_data(posts_var, posts_len, response_var, response_len)
        # 
        rq_recon_error = rq_rbm.train_op(q_pos, r_pos)
        qr_recon_error = qr_rbm.train_op(r_pos, q_pos)
        rq_recon.append(rq_recon_error)
        qr_recon.append(qr_recon_error)
        rq_fe.append( rq_rbm.hr_free_energy(q_pos, r_pos))
        qr_fe.append( qr_rbm.hr_free_energy(r_pos, q_pos))
        total_case += len(batch[0])

        step += 1
        if step == 1:
            logger.info("QR FE %.4f, RQ FE %.4f " % (qr_fe[0],
                         rq_fe[0]) + 
                       "recon error %.4f, %.4f, %.4f, %.4f %4.f, %4.f" % rq_recon_error )
        if step % args.print_every == 0:
            if step > 0:
                logger.info('Train batch %d: average QR FE %.4f, RQ FE %.4f ' % (step, qr_fe[-1], rq_fe[-1])
                + 'recon_error  %.4f, %.4f, %.4f, %.4f, (%.1f case/sec)' % (rq_recon_error[1], rq_recon_error[2],
                qr_recon_error[1], qr_recon_error[2],
                total_case/(time.time()-cur_time)))
                total_case = 0
                cur_time = time.time()
        if step % 5000 == 0:
            eval_model(e)
        # 
    # save model
    avg_qr_fe = sum(qr_fe) / len(qr_fe) 
    avg_rq_fe = sum(rq_fe) / len(rq_fe) 
    logger.info("%.4f, %.4f" % (avg_qr_fe, avg_rq_fe))
    rq_error = zip(*rq_recon)
    qr_error = zip(*qr_recon)
    batch_num = len(rq_recon)
    logger.info("Train recon error: %.4f, %.4f, %.4f %.4f" % (sum(rq_error[1])/batch_num,
        sum(rq_error[2])/batch_num, sum(qr_error[1])/batch_num, sum(qr_error[2])/batch_num))
    logger.info('Epoch %d average QR FE %.4f, RQ FE %.4f, QR, FEgap: %.4f, %4.f, RQ FEGap: %.4f, %.4f;' % (
                e, sum(qr_fe) / len(qr_fe), sum(rq_fe) / len(rq_fe),
                (sum(qr_fe)-sum(qr_error[4])) /len(qr_fe), (sum(qr_fe)-sum(qr_error[5]))/len(qr_fe),
                (sum(rq_fe)-sum(rq_error[4])) /len(rq_fe), (sum(rq_fe)-sum(rq_error[5]))/len(rq_fe),
                ))
    rq_rbm.save_model(os.path.join(args.exp_dir, "rq_{}".format(e)))
    qr_rbm.save_model(os.path.join(args.exp_dir, "qr_{}".format(e)))
    
    # validate 
    eval_model(e) 
