#!/usr/bin/env python

from __future__ import division

import argparse
import glob
import os
import sys
import random
from datetime import datetime

import torch
import torch.nn as nn
from torch import cuda
# import torch.optim as optim

import generator.Models
from generator import ModelConstructor
import generator.opts as opts
from tool.vocab import Vocab 
from modules.utils import use_gpu
from generator.Trainer import Statistics, DoubleTrainer
from modules.Optim import Optim
from modules.utils import compute_recall_ks
from tool.datasetbase import DataSet 
import numpy as np

parser = argparse.ArgumentParser(
    description='train.py',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

opts.model_opts(parser)
opts.train_opts(parser)

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

if opt.tensorboard:
    from tensorboardX import SummaryWriter
    writer = SummaryWriter(
        opt.tensorboard_log_dir + datetime.now().strftime("/%b-%d_%H-%M-%S"),
        comment="Onmt")

vocab = torch.load(opt.vocab_path)
# set padding_word number 
opt.padding_idx = 0
opt.numwords = len(vocab.word2idx)
opt.enc_numwords = opt.numwords
opt.dec_numwords = opt.numwords

# loss function
criterion = nn.CrossEntropyLoss(ignore_index=opt.dec_padding_idx,
                                size_average=False) 

progress_step = 0

def report_func(epoch, batch, num_batches,
                progress_step,
                start_time, lr, report_stats):
    """
    This is the user-defined batch-level traing progress
    report function.

    Args:
        epoch(int): current epoch count.
        batch(int): current batch count.
        num_batches(int): total number of batches.
        progress_step(int): the progress step.
        start_time(float): last report time.
        lr(float): current learning rate.
        report_stats(Statistics): old Statistics instance.
    Returns:
        report_stats(Statistics): updated Statistics instance.
    """
    if batch % opt.report_every == -1 % opt.report_every:
        report_stats.output(epoch, batch + 1, num_batches, start_time)
        report_stats = Statistics()
    return report_stats

############################*****************************#########################

def train_model(model, optim, criterion, model_opt):
    trainer = DoubleTrainer(model, optim, criterion, True) 

    print('\nStart training...')
    print(' * number of epochs: %d, starting from Epoch %d' %
          (opt.epochs + 1 - opt.start_epoch, opt.start_epoch))
    print(' * batch size: %d' % opt.batch_size)
    valid_iter = lazily_load_dataset("valid")
    valid_stats = trainer.validate(valid_iter)
    print('Validation Loss: %g' % valid_stats.xent())
    print('Validation accuracy: %g' % valid_stats.accuracy())

    for epoch in range(opt.start_epoch, opt.epochs + 1):
        print('')

        # 1. Train for one epoch on the training set.
        train_iter = lazily_load_dataset("train")
        train_stats = trainer.train(train_iter, epoch, report_func)
        print('Train Loss: %g' % train_stats.xent())
        print('Train accuracy: %g' % train_stats.accuracy())

        # 2. Validate on the validation set.
        valid_iter = lazily_load_dataset("valid")
        valid_stats = trainer.validate(valid_iter)
        print('Validation Loss: %g' % valid_stats.xent())
        print('Validation accuracy: %g' % valid_stats.accuracy())

        # 3. Log to remote server.
        if opt.exp_host:
            train_stats.log("train", experiment, optim.lr)
            valid_stats.log("valid", experiment, optim.lr)
        if opt.tensorboard:
            train_stats.log_tensorboard("train", writer, optim.lr, epoch)
            train_stats.log_tensorboard("valid", writer, optim.lr, epoch)

        # 4. Update the learning rate
        #trainer.epoch_step(valid_stats.xent(), epoch)

        # 5. Drop a checkpoint if needed.
        if epoch >= opt.start_checkpoint_at:
            trainer.drop_checkpoint(model_opt, epoch, valid_stats)

def lazily_load_dataset(corpus_type):
    """
    Dataset generator. Don't do extra stuff here, like printing,
    because they will be postponed to the first loading time.

    Args:
        corpus_type: 'train' or 'valid'
    Returns:
        A list of dataset, the dataset(s) are lazily loaded.
    """
    assert corpus_type in ["train", "valid"]

    def lazy_dataset_loader(pt_file, corpus_type):
        dataset = torch.load(pt_file)
        print('Loading %s dataset from %s, number of examples: %d' %
              (corpus_type, pt_file, len(dataset.examples)))
        return dataset

    # Sort the glob output by file name (by increasing indexes).
    pts = sorted(glob.glob(opt.data + '.' + corpus_type + '.[0-9]*.pt'))
    if pts:
        for pt in pts:
            dataset = lazy_dataset_loader(pt, corpus_type)
            dataset.set_property(batch_size=opt.batch_size, 
                                 with_label=False, rank=False,
                                 eos_id=2, sos_id=1)
            yield dataset
    else:
        # Only one onmt.io.*Dataset, simple!
        pt = opt.data + '.' + corpus_type + '.pt'
        dataset = lazy_dataset_loader(pt, corpus_type)
        dataset.set_batch_size(opt.batch_size)
        yield dataset

######################********************#########################

def check_save_model_path():
    save_model_path = os.path.abspath(opt.save_model)
    model_dirname = os.path.dirname(save_model_path)
    if not os.path.exists(model_dirname):
        os.makedirs(model_dirname)

def tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    print('* number of parameters: %d' % n_params)
    enc = 0
    dec = 0
    for name, param in model.named_parameters():
        if 'encoder' in name:
            enc += param.nelement()
        else:
            dec += param.nelement()
    print('encoder: ', enc)
    print('project: ', dec)

def build_model(model_opt, opt, checkpoint):
    print('Building model...')
    model = ModelConstructor.make_base_model(model_opt,
                                             use_gpu(opt),
                                             checkpoint)
    print(model)
    return model

def build_optim(model, opt, checkpoint):
    if opt.train_from:
        print('Loading optimizer from checkpoint.')
        optim = checkpoint['optim']
        optim.optimizer.load_state_dict(
            checkpoint['optim'].optimizer.state_dict())
    else:
        print('Making optimizer for training.')
        optim = Optim(
            opt.optim, opt.learning_rate, opt.max_grad_norm,
            lr_decay=opt.learning_rate_decay,
            start_decay_at=opt.start_decay_at,
            beta1=opt.adam_beta1,
            beta2=opt.adam_beta2,
            adagrad_accum=opt.adagrad_accumulator_init,
            decay_method=opt.decay_method,
            warmup_steps=opt.warmup_steps,
            model_size=opt.rnn_size)

    optim.set_parameters(model.parameters())
    #for name, para in model.named_parameters():
    #    print(name)
    #paras = [para for name, para in model.named_parameters() if name.find('embeddings') == -1] 
    #optim.set_parameters(paras)

    return optim

def main():
    # Load checkpoint if we resume from a previous training.
    if opt.train_from:
        print('Loading checkpoint from %s' % opt.train_from)
        checkpoint = torch.load(opt.train_from,
                                map_location=lambda storage, loc: storage)
        model_opt = checkpoint['opt']
        # I don't like reassigning attributes of opt: it's not clear.
        opt.start_epoch = checkpoint['epoch'] + 1
    else:
        checkpoint = None
        model_opt = opt

    # Build model.
    model = build_model(model_opt, opt, checkpoint)
    tally_parameters(model)
    check_save_model_path()
    # ????? handling no sharing scenario
    if opt.pre_word_vecs is not None and os.path.exists(opt.pre_word_vecs):
        W = np.load(opt.pre_word_vecs)
        model.encoder.embeddings.embeddings.weight.data.copy_(torch.from_numpy(W))
        #W = torch.load(opt.pre_word_vecs)
        #model.encoder.embeddings.embeddings.weight.data.copy_(torch.from_numpy(W))

    # Build optimizer.
    optimizer = build_optim(model, opt, checkpoint)
    
    # Do training.
    train_model(model, optimizer, criterion, model_opt)

if __name__ == "__main__":
    main()
