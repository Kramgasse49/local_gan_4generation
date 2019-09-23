#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
"""
import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform

from modules.Encoder import MeanEncoder, RNNEncoder, CNNEncoder
from modules.Decoder import InputFeedRNNDecoder, StdRNNDecoder
from Models import GANRBM
from modules.Embeddings import Embeddings 
from modules.utils import use_gpu 
from ael_layer import ApproxEmbedding 
from discriminator_rbm import Discriminator

from normalization.scale_torch import StandardScalerTorch

import os 

#from text_qr_rbm import QRRBM
from text_rbm import TRBM

def make_embeddings(opt, num_word, padding_idx):
    """
    """
    return Embeddings(opt.word_vec_size, num_word, padding_idx,
            dropout=opt.dropout,sparse=opt.optim=="sparseadam")
def make_encoder(opt, embeddings):
    """
    """
    if opt.encoder_type == "rnn":
        return RNNEncoder(opt.rnn_type, opt.brnn, opt.enc_layers,
            opt.rnn_size, opt.dropout, embeddings=embeddings)
    elif opt.encoder_type == "cnn":
        return CNNEncoder(opt.hidden_size, num_layers=opt.enc_layers,
            filter_num=opt.filter_num, filter_sizes=[1,2,3,4],
            dropout=opt.dropout, embeddings=embeddings)
    elif opt.encoder_type == "mean":
        return MeanEncoder(opt.enc_layers, embeddings)
    else:
        raise ValueError("""Unsupported Encoder type:{}""".format(opt.encoder_type))

def make_decoder(opt, embeddings):
    """
    """
    if opt.decoder_type == "input_feed":
        return InputFeedRNNDecoder(opt.rnn_type, opt.brnn, opt.dec_layers,
            opt.rnn_size, attn_type=opt.attn_type, dropout=opt.dropout,
            embeddings=embeddings)
    else:
       return StdRNNDecoder(opt.rnn_type, opt.brnn, opt.dec_layers, opt.rnn_size,
            attn_type=opt.attn_type, dropout=opt.dropout, embeddings=embeddings)


def make_dbm_discriminator(opt):
    """
    """
    rq_path = os.path.join(opt.rbm_path, opt.rbm_rq_prefix)
    qr_path = os.path.join(opt.rbm_path, opt.rbm_qr_prefix)
    rq_rbm = TRBM(opt.word_vec_size, opt.word_vec_size, 
                   2*opt.hidden_size, use_cuda=use_gpu(opt)) # r->q
    #rq_rbm.load_model(rq_path)
    qr_rbm = TRBM(opt.word_vec_size, opt.word_vec_size, 
                   2*opt.hidden_size, use_cuda=use_gpu(opt)) # q->r
    qr_rbm.load_model(qr_path)
    
    discor = Discriminator( opt.word_vec_size, rq_rbm, qr_rbm, 
        opt.hidden_size//2, class_num=1)
    return discor

def make_qr_norm(opt):
    """
    """
    q_scaler = torch.load(opt.query_norm_path)
    r_scaler = torch.load(opt.reply_norm_path)
    q_scaler_torch = StandardScalerTorch()
    q_scaler_torch.load_weight(q_scaler, True)
    r_scaler_torch = StandardScalerTorch()
    r_scaler_torch.load_weight(r_scaler, True)
    return q_scaler_torch, r_scaler_torch

def make_base_model(model_opt, gpu, checkpoint=None):
    """
    """
    ################# The canical seq2seq ###############################
    embeddings = make_embeddings(model_opt, model_opt.enc_numwords,
                          model_opt.enc_padding_idx)
    encoder = make_encoder(model_opt, embeddings)
    # 
    if model_opt.share_embeddings:
        decoder = make_decoder(model_opt, embddings)
    else:
        tgt_embedding = make_embeddings(model_opt, model_opt.dec_numwords,
                    model_opt.dec_padding_idx)
        decoder = make_decoder(model_opt, tgt_embedding)
    ################## Discriminator ####################################
    discor = make_dbm_discriminator(model_opt)
    # Discriminator(model_opt.word_vec_size, filter_num=32, filter_sizes=[1,2], 
    #hidden_size=model_opt.hidden_size, class_num=1)
    ################## Generator (Projection Layer) #####################
    generator = nn.Sequential(
        nn.Linear(model_opt.rnn_size, model_opt.dec_numwords)
        )
    # AEL 
    ael = ApproxEmbedding(decoder.embeddings)
    # normalizer 
    q_norm, r_norm = make_qr_norm(model_opt)
    # final model 
    model = GANRBM(encoder, decoder, ael, generator, discor,
        dec_max_len=model_opt.dec_max_len, type_loss=model_opt.gan_loss_type,
        q_normalizer=q_norm, r_normalizer=r_norm ) 
    # Load the model states from checkpoint or initialize them.
    # remove rbm part 
    if model_opt.param_init != 0.0:
        print('Intializing model parameters.')
        for p in model.parameters():
            p.data.uniform_(-model_opt.param_init, model_opt.param_init)
        for p in generator.parameters():
            p.data.uniform_(-model_opt.param_init, model_opt.param_init)
    else:
        for p in model.parameters():
            if p.dim() > 1:
                xavier_uniform(p)
        for p in generator.parameters():
            if p.dim() > 1:
                xavier_uniform(p)
    # special intialization for DBM 
    if model_opt.rbm_path is not None:
        print("Initial rmb", model_opt.rbm_path)
        model.disor.rq_rbm.load_model(os.path.join(model_opt.rbm_path,
                                 model_opt.rbm_rq_prefix))
        model.disor.qr_rbm.load_model(os.path.join(model_opt.rbm_path,
                                 model_opt.rbm_qr_prefix))
    # 
    if checkpoint is not None:
        print('Loading model parameters.')
        load_temp = ['encoder', 'decoder', 'generator']
        model.encoder.load_state_dict(checkpoint['encoder'])
        model.decoder.load_state_dict(checkpoint['decoder'])
        model.generator.load_state_dict(checkpoint['generator'])
        if 'ael' in checkpoint:
            model.ael.load_state_dict(checkpoint['ael'])
            load_temp.append('ael')
        if 'disor' in checkpoint:
            model.disor.load_state_dict(checkpoint['disor'])
            load_temp.append('disor')
        print("Load", load_temp)
    # DBM Initializatio
    return model 
    # end of 

def load_test_model(opt):
    checkpoint = torch.load(opt.model_path,
                            map_location=lambda storage, loc: storage)

    model_opt = checkpoint['opt']
    model = make_base_model(model_opt, use_gpu(opt), checkpoint)
    if use_gpu(opt):
        model = model.cuda()
    model.eval()
    model.generator.eval()

    return model
