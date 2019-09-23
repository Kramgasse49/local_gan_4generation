# -*- coding:utf-8 -*-
"""
    Q-R & R-Q RBM
    
    ref:
        https://github.com/wiseodd/generative-models
        https://github.com/monsta-hd/boltzmann-machines
        https://github.com/GabrielBianconi/pytorch-rbm
        https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/neural_network/rbm.py
        https://github.com/GabrielBianconi/pytorch-rbm/blob/master/rbm.py
        https://github.com/meownoid/tensorfow-rbm

    Problem,
        (1) how to get v_q and v_r based on Embedding Matrix.
            a, embedding average, there still exist other problems, 
            such as in the generated responses have max_length word_embedding,
            while the real ones just contains word embedding in itself length.

            b, Convolutional Layer, 
            This layer need to be trained.

            c, max-pooling directly

        (2) training on GPU

        (3) as an additional component of Seq2Seq model

        hidden=128, 92%
        hidden=256, 94.72%
        hidden=512, 96%
"""

from __future__ import unicode_literals, print_function, division

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np 
from collections import OrderedDict

def sample_gaussian(x, sigma=1.0):
    v_bias = torch.Tensor(x.size()).normal_(mean=0., std=sigma)
    if x.is_cuda:
        v_bias = v_bias.cuda()
    return x + v_bias 

def sample_bernoulli(probs):
    return torch.bernoulli(probs)

class TRBM(object):
    """
    """
    def __init__(self, q_size, r_size, h_size, use_cuda=False,
                 lr=0.001, monument=0.5, weight_decay=0., sigma=1.0):#1e-5):
        """
        """
        self.q_size = q_size
        self.r_size = r_size
        self.h_size = h_size
        self.lr = lr
        self.use_cuda = use_cuda
        self.monument = monument
        self.weight_decay = weight_decay
        self.sigma = sigma

        self.build()

    def build(self):
        """
        """
        self.W_rr, self.W_rq = ( torch.randn(self.r_size, self.h_size) * 0.01,
                                 torch.randn(self.q_size, self.h_size) * 0.01 )
        self.b_q, self.b_r, self.c_r = ( torch.ones(self.q_size) * 0.001,
                                         torch.ones(self.r_size) * 0.001,
                                         torch.ones(self.h_size) * 0.001) 

        nn.init.xavier_normal(self.W_rr)
        nn.init.xavier_normal(self.W_rq)

        if self.use_cuda:
            self.W_rr, self.W_rq = (self.W_rr.cuda(), self.W_rq.cuda())
            self.b_q, self.b_r, self.c_r = (self.b_q.cuda(), self.b_r.cuda(), 
                                            self.c_r.cuda())
        # 
        self.W_rr_momentum, self.W_rq_momentum = ( torch.zeros(self.r_size, self.h_size),
                                                   torch.zeros(self.h_size, self.q_size) )
        (self.b_q_momentum, self.b_r_momentum, 
         self.c_r_momentum) = ( torch.zeros(self.q_size),
                                torch.zeros(self.r_size), torch.zeros(self.h_size) )
        if self.use_cuda:
            self.W_rr_momentum, self.W_rq_momentum = ( self.W_rr_momentum.cuda(), self.W_rq_momentum.cuda())
            (self.b_q_momentum, self.b_r_momentum, 
             self.c_r_momentum) = ( self.b_q_momentum.cuda(), self.b_r_momentum.cuda(),
                                    self.c_r_momentum.cuda() )  
        # 
    def rq_sample_hr(self, q_pos, r_pos):
        return torch.sigmoid( F.linear(q_pos, self.W_rq.transpose(0,1), self.c_r) 
                            + F.linear(r_pos, self.W_rr.transpose(0,1)))
    def r_sample_hr(self, r_pos):
        return torch.sigmoid( F.linear(r_pos, self.W_rr.transpose(0,1), self.c_r))

    def q_sample_hr(self, q_pos):
        return torch.sigmoid( F.linear(q_pos, self.W_rq.transpose(0,1), self.c_r))

    def hr_sample_q(self, hr):
        return F.linear(hr, self.W_rq, self.b_q)

    def hr_sample_r(self, hr):
        return F.linear(hr, self.W_rr, self.b_r)
    
    def hr_free_energy(self, q_pos, r_pos):
        '''
        '''
        batch_size = q_pos.size(0)
        #hr = self.rq_sample_hr(q_pos, r_pos)
        wx_b = F.linear(q_pos, self.W_rq.transpose(0,1), self.c_r) + F.linear(r_pos, self.W_rr.transpose(0,1))
        x = torch.cat((q_pos, r_pos), dim=1)
        bx = torch.cat((self.b_q, self.b_r))
        vbias_term = 0.5*torch.sum( (x - bx)**2, dim=1 )
        #hidden_term = torch.sum( torch.log(1. + torch.exp(wx_b)), dim=1 )
        hidden_term = torch.sum( F.logsigmoid(Variable(-wx_b)).data, dim=1)
        # check inf
        v_avg = torch.sum(vbias_term) / batch_size
        h_avg = torch.sum(hidden_term) / batch_size
        if v_avg == float('inf'):
            print('v_term', v_avg)
        if h_avg == -float('inf'):
            print('h_term', h_avg) 
        return torch.sum(hidden_term + vbias_term)/batch_size

    def hr_free_energy_one(self, q_pos, r_pos):
        '''
        '''
        batch_size = q_pos.size(0)
        #hr = self.rq_sample_hr(q_pos, r_pos)
        wx_b = F.linear(q_pos, self.W_rq.transpose(0,1), self.c_r) + F.linear(r_pos, self.W_rr.transpose(0,1))
        x = torch.cat((q_pos, r_pos), dim=1)
        bx = torch.cat((self.b_q, self.b_r))
        vbias_term = 0.5*torch.sum( (x - bx)**2, dim=1 )
        #hidden_term = torch.sum( torch.log(1. + torch.exp(wx_b)), dim=1 )
        hidden_term = torch.sum( F.logsigmoid(Variable(-wx_b)).data, dim=1)
        # check inf
        #return -hidden_term + vbias_term
        return hidden_term + vbias_term

    def train_op(self, q_pos, r_pos):
        """training dule RBM
        """
        assert q_pos.size(0) == r_pos.size(0)
        batch_size = q_pos.size(0)

        ############################ [r, q]-->hr Success ##########################
        # postive process
        hr_pos = self.rq_sample_hr(q_pos, r_pos)

        # negative process
        hr_sampled = sample_bernoulli(hr_pos)
        #q_neg, r_neg = ( sample_gaussian( self.hr_sample_q(hr_sampled), sigma=self.sigma ),
        #                 sample_gaussian( self.hr_sample_r(hr_sampled), sigma=self.sigma )) 
        q_neg, r_neg = ( self.hr_sample_q(hr_sampled),
                         self.hr_sample_r(hr_sampled) ) 
        
        hr_neg = self.rq_sample_hr(q_neg, r_neg)

        ############################ r-->q ##########################
        hr_r_pos, hr_q_pos = ( self.r_sample_hr(r_pos),
                               self.q_sample_hr(q_pos) )
        hr_r_sampled = sample_bernoulli(hr_r_pos)
        r_q_neg = self.hr_sample_q( hr_r_sampled ) #
        r_r_neg = self.hr_sample_r( hr_r_sampled ) #

        hr_q_neg = ( self.q_sample_hr(r_q_neg) )
        hr_r_neg = ( self.r_sample_hr(r_r_neg) )

        hr_rq_neg = self.rq_sample_hr(r_q_neg, r_r_neg)

        # postive phase # check this part 
        rr_pos_phase, rq_pos_phase = ( torch.matmul(torch.transpose(r_pos, 0, 1), hr_pos),
                                       torch.matmul(torch.transpose(q_pos, 0, 1), hr_pos) )

        rq_hr_pos_phase = torch.matmul(torch.transpose(q_pos, 0, 1), hr_r_pos) # hr_r_pos
        rr_hr_pos_phase = torch.matmul(torch.transpose(r_pos, 0, 1), hr_r_pos)

        # negative phase
        rr_neg_phase, rq_neg_phase = ( torch.matmul(torch.transpose(r_neg, 0, 1), hr_neg),
                                       torch.matmul(torch.transpose(q_neg, 0, 1), hr_neg) )
        
        rq_hr_neg_phase = torch.matmul(torch.transpose(r_q_neg, 0, 1), hr_rq_neg) #hr_q_neg)
        rr_hr_neg_phase = torch.matmul(torch.transpose(r_r_neg, 0, 1), hr_rq_neg) #hr_r_neg)

        #delta_rr, delta_rq = ( (rr_pos_phase - rr_neg_phase) / batch_size, # + (rr_hr_pos_phase - rr_hr_neg_phase),
        delta_rr, delta_rq = ( ((rr_pos_phase - rr_neg_phase) + (rr_hr_pos_phase - rr_hr_neg_phase))/batch_size,
                               ((rq_pos_phase - rq_neg_phase) + (rq_hr_pos_phase - rq_hr_neg_phase))/batch_size )

        delta_bq, delta_br, delta_cr = ( torch.sum((q_pos - q_neg) + (q_pos - r_q_neg), dim=0)/batch_size, 
                                         #torch.sum((r_pos - r_neg), dim=0)/batch_size, # + (r_pos - r_r_neg),
                                         torch.sum((r_pos - r_neg) + (r_pos - r_r_neg), dim=0)/batch_size,  
                                         torch.sum(hr_pos - hr_neg, dim=0)/batch_size
                                       )
        
        # lack of monument
        self.W_rr_momentum, self.W_rq_momentum = ( self.W_rr_momentum * self.monument, 
                                                   self.W_rq_momentum * self.monument )
        self.W_rr_momentum, self.W_rq_momentum = ( self.W_rr_momentum + (1. - self.monument)*delta_rr, 
							   self.W_rq_momentum + (1. - self.monument)*delta_rq )
        
        (self.b_q_momentum, self.b_r_momentum,
         self.c_r_momentum) = ( self.b_q_momentum * self.monument, 
                                self.b_r_momentum * self.monument, 
                                self.c_r_momentum * self.monument 
                               )
        (self.b_q_momentum, self.b_r_momentum,
                            self.c_r_momentum) = ( self.b_q_momentum + (1. - self.monument)*delta_bq, #/2., 
                                                   self.b_r_momentum + (1. - self.monument)*delta_br, #/2., 
                                                   self.c_r_momentum + (1. - self.monument)*delta_cr 
                                                  )
        self.b_q, self.b_r, self.c_r = ( self.b_q + (self.lr * self.b_q_momentum), #/batch_size,
                                                   self.b_r + (self.lr * self.b_r_momentum), #/batch_size,
                                                   self.c_r + (self.lr * self.c_r_momentum) #/batch_size
                                                  )
        # 
        self.W_rr, self.W_rq = ( self.W_rr + (self.lr * self.W_rr_momentum), #/batch_size,
                                 self.W_rq + (self.lr * self.W_rq_momentum) )

        # L2 weight decay, self.weights -= self.weights * self.weight_decay 
        self.W_rr, self.W_rq = ( self.W_rr - self.W_rr * self.weight_decay, 
                                 self.W_rq - self.W_rq * self.weight_decay)
        #
        q_recon_error = torch.mean( (q_pos - q_neg)**2 )
        r_recon_error = torch.mean( (r_pos - r_neg)**2 ) 
        r_q_recon_error = torch.mean( (q_pos - r_q_neg)**2 )
        r_r_recon_error = torch.mean( (r_pos - r_r_neg)**2 )
        neg_fe = self.hr_free_energy(q_neg, r_neg)
        r_neg_fe = self.hr_free_energy(r_q_neg, r_r_neg)
        return (q_recon_error, r_q_recon_error, r_recon_error, r_r_recon_error, neg_fe, r_neg_fe)

    def valid_op(self, q_pos, r_pos):
        """training dule RBM
        """
        assert q_pos.size(0) == r_pos.size(0)
        batch_size = q_pos.size(0)
        # postive process
        hr_pos = self.rq_sample_hr(q_pos, r_pos)

        # negative process
        hr_sampled = sample_bernoulli(hr_pos)
        #q_neg, r_neg = ( sample_gaussian( self.hr_sample_q(hr_sampled), sigma=self.sigma ),
        #                 sample_gaussian( self.hr_sample_r(hr_sampled), sigma=self.sigma )) 
        q_neg, r_neg = ( self.hr_sample_q(hr_sampled),
                         self.hr_sample_r(hr_sampled) ) 

        ############################ r-->q ##########################
        hr_r_pos, hr_q_pos = ( self.r_sample_hr(r_pos),
                               self.q_sample_hr(q_pos) )
        hr_r_sampled = sample_bernoulli(hr_r_pos)
        r_q_neg = self.hr_sample_q( hr_r_sampled ) 
        r_r_neg = self.hr_sample_r( hr_r_sampled ) 


        # reconstruction error 
        q_recon_error = torch.mean( (q_pos - q_neg)**2 )
        r_recon_error = torch.mean( (r_pos - r_neg)**2 )    
        r_q_recon_error = torch.mean( (q_pos - r_q_neg)**2 )
        r_r_recon_error = torch.mean( (r_pos - r_r_neg)**2 ) 

        neg_fe = self.hr_free_energy(q_neg, r_neg)
        r_neg_fe = self.hr_free_energy(r_q_neg, r_r_neg)
        return (q_recon_error, r_q_recon_error, r_recon_error, r_r_recon_error, neg_fe, r_neg_fe)
    
    @property
    def parameters(self):
        return [self.W_rr, self.W_rq, self.b_q, self.b_r, self.c_r]

    def save_model(self, outdir):
        """
        """
        parameters = [self.W_rr, self.W_rq, self.b_q, self.b_r, self.c_r]
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        for idx, p in enumerate(parameters):
            if self.use_cuda:
                np.save(os.path.join(outdir, '{}.npy'.format(idx)), p.cpu().numpy())
            else:
                np.save(os.path.join(outdir, '{}.npy'.format(idx)), p.numpy())
        # 
    def load_model(self, indir):
        """
        """
        parameters = [self.W_rr, self.W_rq, self.b_q, self.b_r, self.c_r]
        for idx, p in enumerate(parameters):
            W = np.load( os.path.join(indir, '{}.npy'.format(idx)) )
            p.copy_(torch.from_numpy(W))
