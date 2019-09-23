#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This file is for models creation, which consults options
and creates each encoder and decoder accordingly.
"""
import torch
import torch.nn as nn

from modules.Encoder import MeanEncoder, RNNEncoder, CNNEncoder
from modules.Decoder import InputFeedRNNDecoder, StdRNNDecoder
from Models import Seq2SeqModel
from torch.nn.init import xavier_uniform

from modules.Embeddings import Embeddings
from modules.utils import use_gpu

def make_embeddings(opt, num_word, padding_idx):
    """
    Make an Embeddings instance.
    Args:
        opt: the option in current environment.
        num_word, 
        padding_idx,
        for_encoder(bool): make Embeddings for encoder or decoder?
    """
    embedding_dim = opt.word_vec_size
    word_padding_idx = padding_idx
    num_word_embeddings = num_word
    return Embeddings(embedding_dim,
                      num_word_embeddings,
                      word_padding_idx,
                      dropout=opt.dropout,
                      sparse=opt.optim == "sparseadam")

def make_encoder(opt, embeddings):
    """
    Various encoder dispatcher function.
    Args:
        opt: the option in current environment.
        embeddings (Embeddings): vocab embeddings for this encoder.
    """
    # "rnn" or "brnn" 
    if opt.encoder_type == 'rnn':
        return RNNEncoder(opt.rnn_type, opt.brnn, opt.enc_layers,
                      opt.rnn_size, opt.dropout, embeddings=embeddings)
    elif opt.encoder_type == 'cnn':
        return CNNEncoder(opt.hidden_size, num_layers=opt.enc_layers, 
                    filter_num=opt.filter_num, filter_sizes=[1, 2, 3, 4], 
                    dropout=opt.dropout, embeddings=embeddings)
    elif opt.encoder_type == "mean":
        return MeanEncoder(opt.enc_layers, embeddings)
    else:
        raise ValueError("""Unsupported Encoder type
        	             : {0}""".format(opt.encoder_type))

def make_decoder(opt, embeddings):
    """
    Various decoder dispatcher function.
    Args:
        opt: the option in current environment.
        embeddings (Embeddings): vocab embeddings for this decoder.
    """
    # revise this partition refering to rnn_factory
    if opt.decoder_type == "input_feed":
        return InputFeedRNNDecoder(opt.rnn_type, opt.brnn,
                                   opt.dec_layers, opt.rnn_size,
                                   attn_type=opt.attn_type,
                                   dropout=opt.dropout,
                                   embeddings=embeddings)
    else:
        return StdRNNDecoder(opt.rnn_type, opt.brnn,
                                   opt.dec_layers, opt.rnn_size,
                                   attn_type=opt.attn_type,
                                   dropout=opt.dropout,
                                   embeddings=embeddings)
    # 

def make_base_model(model_opt, gpu, checkpoint=None):
    """
    Args:
        model_opt: the option loaded from checkpoint.
        gpu(bool): whether to use gpu.
        checkpoint: the model gnerated by train phase, or a resumed snapshot
                    model from a stopped training.
    Returns:
        the Classifier.
    """

    # Make encoder,
    embeddings = make_embeddings(model_opt, 
                                 model_opt.enc_numwords,
                                 model_opt.enc_padding_idx )
    encoder = make_encoder(model_opt, embeddings)

    # make decoder, currently share embedding 
    if model_opt.share_embeddings:
        decoder = make_decoder(model_opt, embeddings)
    else:
        tgt_embeddings = make_embeddings(model_opt, 
                                 model_opt.dec_numwords,
                                 model_opt.dec_padding_idx)
        decoder = make_decoder(model_opt, tgt_embeddings)
    # 
    model = Seq2SeqModel(encoder, decoder)

    # make generator, mapping the outputs of the decoder
    # to word distributions.
    generator = nn.Sequential(
            nn.Linear(model_opt.rnn_size, model_opt.dec_numwords)
            )
            #nn.LogSoftmax())

    # Load the model states from checkpoint or initialize them.
    if checkpoint is not None:
        print('Loading model parameters.')
        #model.load_state_dict(checkpoint['model'])
        model.encoder.load_state_dict(checkpoint['encoder'])
        model.decoder.load_state_dict(checkpoint['decoder'])
        generator.load_state_dict(checkpoint['generator'])
    else:
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
        # end of parameter initialization

    # Add generator to model (this registers it as parameter of model).
    model.generator = generator

    if gpu:
        model.cuda()
    else:
        model.cpu()

    return model

def load_test_model(opt):
    checkpoint = torch.load(opt.model_path,
                            map_location=lambda storage, loc: storage)

    model_opt = checkpoint['opt']
    model = make_base_model(model_opt, use_gpu(opt), checkpoint)
    model.eval()
    model.generator.eval()

    return model
