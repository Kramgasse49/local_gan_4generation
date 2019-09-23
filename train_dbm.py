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

import gan_rbm.Models
from gan_rbm import ModelConstructor
import gan_rbm.opts as opts
from tool.vocab import Vocab 
from modules.utils import use_gpu
from gan_rbm.Trainer import Statistics, GANTrainer
from modules.Optim import Optim
from tool.datasetbase import DataSet 
import numpy as np
from tool.utils import lazily_load_dataset

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

# modify vocab size according to the real vocab
# ????????????????
vocab = torch.load(opt.vocab_path)
# set padding_word number 
opt.padding_idx = 0
opt.numwords = len(vocab.word2idx)
opt.enc_numwords = opt.numwords
opt.dec_numwords = opt.numwords

# loss function
s2s_criterion = nn.CrossEntropyLoss(ignore_index=opt.dec_padding_idx,
                                size_average=False) 
#discor_criterion = nn.BECLogistLoss(size_average=False)

progress_step = 0

def report_func(epoch, batch, num_batches,
                progress_step,
                start_time, lr, report_stats, mode=1):
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
        report_stats.output(epoch, batch + 1, num_batches, start_time, mode=mode)
        report_stats = Statistics(loss_s2s=0, num_word=0, n_word_correct=0,
            cls_racc=[], cls_facc=[], loss_d=[], loss_g=[], loss_delta=[])
    return report_stats

############################*****************************#########################

def train_model(model, s2s_optim, dec_optim, d_optim, g_optim, 
          s2s_criterion, model_opt):
    trainer = GANTrainer(model, s2s_optim, dec_optim, d_optim, g_optim, 
          s2s_criterion, True, opt.d_train_ratio, model_opt, opt) 

    print('\nStart training...')
    print(' * number of epochs: %d, starting from Epoch %d' %
          (opt.epochs + 1 - opt.start_epoch, opt.start_epoch))
    print(' * batch size: %d' % opt.batch_size)
    valid_iter = lazily_load_dataset("valid", opt)
    valid_stats = trainer.gan_ael_validate(valid_iter)
    print('Validation Loss: %g' % valid_stats.xent_s2s())
    print('Validation accuracy: %g' % valid_stats.word_accuracy())
    loss_d = sum(valid_stats.loss_d)/len(valid_stats.loss_d) if len(valid_stats.loss_d)>0 else 0. 
    loss_g = sum(valid_stats.loss_g)/len(valid_stats.loss_g) if len(valid_stats.loss_g)>0 else 0.
    cls_racc = sum(valid_stats.cls_racc)/len(valid_stats.cls_racc) if len(valid_stats.cls_racc)>0 else 0.
    cls_facc = sum(valid_stats.cls_facc)/len(valid_stats.cls_facc) if len(valid_stats.cls_facc)>0 else 0.
    delta = sum(valid_stats.loss_delta)/len(valid_stats.loss_delta) if len(valid_stats.loss_delta)>0 else 0.
    print('\t D_Loss: %g; G_loss: %g; Delat: %g' % (loss_d, loss_g, delta))
    print('\t Real_Acc: %g; Fake_Acc: %g' % (cls_racc, cls_facc))
    
    for epoch in range(opt.start_epoch, opt.epochs + 1):
        print('')

        # 1. Train for one epoch on the training set.
        train_iter = lazily_load_dataset("train", opt)
        train_stats = trainer.gan_ael_train(train_iter, epoch, report_func)
        print('Train Loss: %g' % train_stats.xent_s2s())
        print('Train accuracy: %g' % train_stats.word_accuracy())

        # 2. Validate on the validation set.
        valid_iter = lazily_load_dataset("valid", opt)
        valid_stats = trainer.gan_ael_validate(valid_iter)
        print('Validation Loss: %g' % valid_stats.xent_s2s())
        print('Validation accuracy: %g' % valid_stats.word_accuracy())
        loss_d = sum(valid_stats.loss_d)/len(valid_stats.loss_d) if len(valid_stats.loss_d)>0 else 0. 
        loss_g = sum(valid_stats.loss_g)/len(valid_stats.loss_g) if len(valid_stats.loss_g)>0 else 0.
        cls_racc = sum(valid_stats.cls_racc)/len(valid_stats.cls_racc) if len(valid_stats.cls_racc)>0 else 0.
        cls_facc = sum(valid_stats.cls_facc)/len(valid_stats.cls_facc) if len(valid_stats.cls_facc)>0 else 0.
        delta = sum(valid_stats.loss_delta)/len(valid_stats.loss_delta) if len(valid_stats.loss_delta)>0 else 0.
        print('\t D_Loss: %g; G_loss: %g; Delat: %g' % (loss_d, loss_g, delta))
        print('\t Real_Acc: %g; Fake_Acc: %g' % (cls_racc, cls_facc))

        # 4. Update the learning rate
        #trainer.epoch_step(valid_stats.xent(), epoch)

        # 5. Drop a checkpoint if needed.
        if epoch >= opt.start_checkpoint_at:
            trainer.drop_checkpoint(epoch, None, valid_stats)

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

def build_optim(sub_model_name, opt, checkpoint, lr=0.001):
    if opt.train_from and 'ale' in checkpoint: # ??????
        print('Loading optimizer from checkpoint.')
        optim = checkpoint['%s_optim'%sub_model_name]
        optim.optimizer.load_state_dict(
            checkpoint['%s_optim'%sub_model_name].optimizer.state_dict())
    else:
        print('Making {} optimizer for training.'.format(sub_model_name))
        optim = Optim(
            opt.optim, lr, opt.max_grad_norm,
            lr_decay=opt.learning_rate_decay,
            start_decay_at=opt.start_decay_at,
            beta1=opt.adam_beta1,
            beta2=opt.adam_beta2,
            adagrad_accum=opt.adagrad_accumulator_init,
            decay_method=opt.decay_method,
            warmup_steps=opt.warmup_steps,
            model_size=opt.rnn_size) # ?????
    return optim 

def build_optim_gan(model, opt, checkpoint):
    if opt.train_from and 'ale' in checkpoint: #???
        # waitting for modification ??? to load optimizer respectively
        print('Loading optimizer from checkpoint.')
        optim = checkpoint['optim']
        optim.optimizer.load_state_dict(
            checkpoint['optim'].optimizer.state_dict())
    else:
        print('Making optimizer for training.')
        d_optim = build_optim('discrim', opt, checkpoint, lr=opt.d_lr) # model.disor, 
        d_optim.set_parameters(model.disor.parameters())
        dec_optim = build_optim('dec', opt, checkpoint, lr=opt.dec_lr)
        generator_optim = build_optim('generator', opt, checkpoint, lr=opt.dec_lr)
        generator_optim.set_parameters(model.generator.parameters())
        s2s_optim = build_optim('s2s', opt, checkpoint, lr=opt.s2s_lr)# ???
        dec_parameters = list(model.decoder.parameters())[1:]
        enc_parameters = list(model.encoder.parameters())[1:]
        dec_optim.set_parameters(dec_parameters)
        s2s_optim.set_parameters(dec_parameters + 
                                 enc_parameters +
                                 list(model.generator.parameters()))
    return d_optim, dec_optim, generator_optim, s2s_optim 

def main():
    # Load checkpoint if we resume from a previous training.
    if opt.train_from:
        print('Loading checkpoint from %s' % opt.train_from)
        checkpoint = torch.load(opt.train_from,
                                map_location=lambda storage, loc: storage)
        if 'ael' in checkpoint:
            model_opt = checkpoint['opt']
        else:
            model_opt = opt
        # I don't like reassigning attributes of opt: it's not clear.
        opt.start_epoch = checkpoint['epoch'] + 1
    else:
        checkpoint = None
        model_opt = opt

    # Build model.
    model = build_model(model_opt, opt, checkpoint)
    # load s2s to initialize the model 
    tally_parameters(model)
    check_save_model_path()
    # handling no sharing scenario
    if opt.pre_word_vecs is not None and os.path.exists(opt.pre_word_vecs):
        W = np.load(opt.pre_word_vecs)
        model.encoder.embeddings.embeddings.weight.data.copy_(torch.from_numpy(W))
    if use_gpu(opt):
        model = model.cuda()
    # Build optimizer. ?? current progress 
    d_optim, dec_optim, generator_optim, s2s_optim = build_optim_gan(model, opt, checkpoint)
    
    # Do training.
    train_model(model, s2s_optim, dec_optim, d_optim, generator_optim, 
          s2s_criterion, model_opt)

if __name__ == "__main__":
    main()
