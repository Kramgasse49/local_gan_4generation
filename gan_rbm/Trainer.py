#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable

import numpy as np
import time 
import sys 
import math 

from tool.utils import lazily_load_dataset

class Statistics(object):
    """
      Accumulator for loss statistics,

      * accuracy,  
          ** word_acc, 
          ** cls_acc,
      * perplexity 
      * elapsed time 
      * delta_loss
      * discriminator_loss 
    """
    def __init__(self, loss_s2s=0, num_word=0, n_word_correct=0,
        cls_racc=[], cls_facc=[], loss_d=[], loss_g=[], loss_delta=[]):
        """
        """
        self.loss_s2s = loss_s2s
        self.loss_d = loss_d 
        self.loss_g = loss_g 
        self.loss_delta = loss_delta
        self.num_word = num_word 
        self.n_word_correct = n_word_correct 
        self.cls_racc=cls_racc
        self.cls_facc=cls_facc
        self.start_time = time.time() 

    def update(self, stat):
        # 
        self.loss_s2s += stat.loss_s2s 
        self.loss_d += stat.loss_d
        self.loss_g += stat.loss_g
        self.loss_delta += stat.loss_delta
        self.num_word += stat.num_word 
        self.n_word_correct += stat.n_word_correct 
        self.cls_racc += stat.cls_racc 
        self.cls_facc += stat.cls_facc 

    def word_accuracy(self):
        return 100 * (self.n_word_correct / self.num_word)
    
    def xent_s2s(self):
        return self.loss_s2s / self.num_word 

    def elapsed_time(self):
       return time.time() - self.start_time 

    def output(self, epoch, batch, n_batches, start, mode=1):
        """
          mode, 1,the canical S2S training 
               2,GAN-AEL
        """
        t = self.elapsed_time()
        if mode == 1:
            print(("Epoch %2d, %5d/%5d; word_acc:%6.2f; xnet_s2s: %6.4f;" +  
            "%3.0f sample / %6.0f s elapsed ") % (epoch, batch, n_batches,
            self.word_accuracy(), self.xent_s2s(), self.num_word / t, t))
        elif mode == 2:
            loss_d = sum(self.loss_d)/len(self.loss_d) if len(self.loss_d)>0 else 0. 
            loss_g = sum(self.loss_g)/len(self.loss_g) if len(self.loss_g)>0 else 0. 
            cls_racc = sum(self.cls_racc)/len(self.cls_racc) if len(self.cls_racc)>0 else 0.
            cls_facc = sum(self.cls_facc)/len(self.cls_facc) if len(self.cls_facc)>0 else 0.
            print(("Epoch %2d, %5d/%5d; word_acc:%6.2f; xnet_s2s: %6.4f;" +  
                "loss_d: %6.4f; loss_g: %6.4f; cls_racc: %2.4f; cls_facc: %2.4f;" +
                "%6.0f sample / %6.0f s elapsed ") % (epoch, batch, n_batches, 
                self.word_accuracy(), self.xent_s2s(), 
                loss_d, loss_g, cls_racc, cls_facc, self.num_word/(t + 1e-5), t))
        elif mode == 3: 
            loss_d = sum(self.loss_d)/len(self.loss_d) if len(self.loss_d)>0 else 0. 
            loss_g = sum(self.loss_g)/len(self.loss_g) if len(self.loss_g)>0 else 0. 
            cls_racc = sum(self.cls_racc)/len(self.cls_racc) if len(self.cls_racc)>0 else 0.
            cls_facc = sum(self.cls_facc)/len(self.cls_facc) if len(self.cls_facc)>0 else 0.
            loss_delta = sum(self.loss_delta) / len(self.loss_delta) if len(self.loss_delta)>0 else 0. 
            print(("Epoch %2d, %5d/%5d; word_acc:%6.2f; xnet_s2s: %6.4f;" +  
                " loss_d: %6.4f;loss_g: %6.4f; cls_racc: %2.4f; cls_facc: %2.4f;" +
                " delta: %6.4f; %6.0f sample / %6.0f s elapsed ") % (epoch, batch, 
                 n_batches, self.word_accuracy(), self.xent_s2s(), 
                 loss_d, loss_g, cls_racc, cls_facc,
                 loss_delta, self.num_word/(t + 1e-5), t))
        else:
            raise ValueError("The statistical mode {} is invalid".format(mode))
        sys.stdout.flush()
    # 

class GANTrainer(object):
    """
    """
    def __init__(self, model,
        s2s_opt, dec_opt, discrim_opt, generator_opt,
        s2s_criterion, use_cuda, d_train_ratio, model_opt, opt):
        """
          s2s_opt: parameters of Encoder-Decoder
          dec_opt: parameters of Decoder, + project layer ?
          discrim_opt: parameters of the discriminator 
          generator_opt: ??

        """
        self.model = model
        self.s2s_opt = s2s_opt
        self.dec_opt = dec_opt 
        self.discrim_opt = discrim_opt 
        self.generator_opt = generator_opt 
        self.s2s_criterion = s2s_criterion  
        self.use_cuda = use_cuda 
        self.d_train_ratio = d_train_ratio
        self.model_opt = model_opt
        self.opt = opt 

    def discrim_pretrain(self, train_iter, epoch, report_func=None):
        """
        """
        self.model.train()
        loss, real_acc, fake_acc, batch_num = ([], [], [], 0)
        
        for sub_dataset in train_iter:
            for batch in sub_dataset:
                q_src, q_lens, r_src, r_lens = batch 
                _, batch_size = q_src.size()
                if self.use_cuda:
                    q_src, q_lens, r_src, r_lens = (q_src.cuda(), q_lens.cuda(),
                        r_src.cuda(), r_lens.cuda())
                loss_D, loss_G, prob_real, prob_fake = self.model.forward_gan(q_src, 
                        q_lens, r_src, r_lens) 
                self.discrim_opt.optimizer.zero_grad()
                loss_D.div(batch_size).backward()
                torch.nn.utils.clip_grad_norm(self.model.discor.parameters(), 10.)
                self.discrim_opt.step()
                for p in self.model.discor.parameters():
                    p.data.clamp_(-0.02, 0.02)
                loss.append(loss_D.data.cpu().numpy()[0])
                real_acc.append(prob_real.data.cpu().numpy().mean())
                fake_acc.append(prob_fake.data.cpu().numpy().mean())
                batch_num += 1
                if batch_num % 1000 == 0:
                    logger.info("""Pretraining D, Epoch:%2d, batch: %6d, D Loss: %6.2f;
                    Real Acc:%4.2f; Fake Acc:%4.2f; """ % (epoch, batch_num,  sum(loss)/len(loss), 
                    sum(real_acc)/len(real_acc), sum(fake_acc)/len(fake_acc)))
                    loss, real_acc, fake_acc = ([], [], [])
        # end epoch 
    def s2s_pretrain(self, train_iter, epoch, report_func=None):
        """
        """
        self.model.train()
        total_stats = Statistics(loss_s2s=0, num_word=0, n_word_correct=0)
        report_stats = Statistics(loss_s2s=0, num_word=0, n_word_correct=0)
        # train
        for sub_dataset in train_iter:
            batch_num = 0 
            num_batch = len(sub_dataset.examples) / sub_dataset.batch_size 
            for batch in sub_dataset:
                q_src, q_lens, r_src, r_lens = batch 
                _, batch_size = q_src.size()
                if self.use_cuda:
                    q_src, q_lens, r_src, r_lens = (q_src.cuda(), q_lens.cuda(),
                    r_src.cuda(), r_lens.cuda())
               # fprward s2s
                _, batch_size = q_src.size()
                batch_stat, loss, _ = self.s2s_forward(q_src, q_len, r_src, r_lens)
                total_stats.update(batch_stat)
                report_stats.update(batch_stat)
                #
                self.s2s_opt.optimizer.zero_grad()
                loss.div(batch_size).backward()
                self.s2s_opt.step()
                batch_num += 1
                if report_func is not None:
                    report_stats = report_func(epoch, batch_num, num_batch,
                        total_stats.start_time, report_stats, mode=1)
            #
        #
        return total_stats 

    def eval_s2s(self, q_src, q_lens, r_src, r_lens):
        """
        """
        self.model.eval()
        batch_stat, _, _ = self.s2s_forward(q_src, q_lens, r_src, r_lens)
        
        self.model.train()
        return batch_stat.loss_s2s, batch_stat.num_word, batch_stat.n_word_correct

    def s2s_forward(self, q_src, q_lens, r_src, r_lens):
        decoder_outputs, _, dec_state = self.model.forward_s2s(q_src, q_lens,
                    r_src, r_lens)
        size = decoder_outputs.size()
        # check the size of output, and r_src 
        output = self.model.generator(decoder_outputs.view(size[0]*size[1], -1))
        loss = self.s2s_criterion(output, r_src[1:].view(size[0]*size[1]))#
        loss_data = loss.data.clone()
        batch_stat = self._s2s_stats(loss_data, output.data,
                r_src[1:].view(size[0]*size[1]).data)
        return batch_stat, loss, output

    def _s2s_stats(self, loss, scores, target):
        mask = (target > 0)
        pred = scores.max(1)[1]
        num_correct = (pred.eq(target)*mask).sum()
        num_word = mask.sum()
        return Statistics(loss_s2s=loss[0], num_word=num_word,
            n_word_correct=num_correct, cls_racc=[], cls_facc=[],
            loss_d=[], loss_g=[], loss_delta=[])

    def _gan_ael_stats(self, loss_s2s, num_word, n_word_correct,
        loss_D, loss_G, loss_delta, real_prob, fake_prob, batch_size):
        """
        """
        racc = real_prob.cpu().numpy().mean()
        facc = fake_prob.cpu().numpy().mean()
        d_loss = loss_D[0] #/ batch_size
        g_loss = loss_G[0] #/ batch_size
        delta_loss = loss_delta[0]
        batch_stat = Statistics(loss_s2s=loss_s2s, num_word=num_word, 
             n_word_correct=n_word_correct, cls_racc=[racc], cls_facc=[facc],
             loss_d=[d_loss], loss_g=[g_loss], loss_delta=[delta_loss])
        return batch_stat

    def gan_ael_train(self, train_iter, epoch, report_func=None):
        """
        """
        self.model.train()
        total_stats = Statistics(loss_s2s=0, num_word=0, n_word_correct=0,
            cls_racc=[], cls_facc=[], loss_d=[], loss_g=[], loss_delta=[])
        report_stats = Statistics(loss_s2s=0, num_word=0, n_word_correct=0,
            cls_racc=[], cls_facc=[], loss_d=[], loss_g=[], loss_delta=[])
        # 
        t_batch_num = 0
        for sub_dataset in train_iter:
            batch_num = 0 
            num_batch = len(sub_dataset.examples) / sub_dataset.batch_size 
            for batch in sub_dataset:
                q_src, q_lens, r_src, r_lens = batch 
                if self.use_cuda:
                    q_src, q_lens, r_src, r_lens = (q_src.cuda(), q_lens.cuda(),
                    r_src.cuda(), r_lens.cuda())
                # forward 
                self.model.train()
                _, batch_size = q_src.size()
                loss_D, loss_G, delta, batch_stat = self.gan_dbm_forward(q_src, q_lens,
                      r_src, r_lens)
                # update state
                total_stats.update(batch_stat)
                report_stats.update(batch_stat)
                #
                batch_num += 1
                t_batch_num += 1
                if batch_num % self.d_train_ratio == 0:
                    self.discrim_opt.optimizer.zero_grad()
                    loss_D.backward(retain_graph=True)
                    torch.nn.utils.clip_grad_norm(self.model.disor.parameters(), 10.)
                    self.discrim_opt.step()
                self.dec_opt.optimizer.zero_grad()
                total_g_loss = loss_G #+ delta
                total_g_loss.backward()
                torch.nn.utils.clip_grad_norm(self.model.decoder.parameters(), 10.)
                self.dec_opt.step()

                if report_func is not None:
                    report_stats = report_func(epoch, batch_num, num_batch,
                        t_batch_num, total_stats.start_time, 0.001,
                        report_stats, mode=3)
                if t_batch_num % 1000 == 0:
                    # validate
                    valid_stats = self.validation(epoch, t_batch_num)
                if t_batch_num % 1000 == 0:
                    self.drop_checkpoint(epoch, t_batch_num, valid_stats)
            # end for 
        return total_stats 
    
    def validation(self, epoch, t_batch_num):
        valid_iter = lazily_load_dataset("valid", self.opt)
        valid_stats = self.gan_ael_validate(valid_iter)
        print('Epoch: %d, batch_num: %d; Validation Loss: %g; Validation accuracy: %g' % (epoch, 
             t_batch_num, valid_stats.xent_s2s(), valid_stats.word_accuracy()))
        loss_d = sum(valid_stats.loss_d)/len(valid_stats.loss_d) if len(valid_stats.loss_d)>0 else 0. 
        loss_g = sum(valid_stats.loss_g)/len(valid_stats.loss_g) if len(valid_stats.loss_g)>0 else 0.
        cls_racc = sum(valid_stats.cls_racc)/len(valid_stats.cls_racc) if len(valid_stats.cls_racc)>0 else 0.
        cls_facc = sum(valid_stats.cls_facc)/len(valid_stats.cls_facc) if len(valid_stats.cls_facc)>0 else 0.
        delta = sum(valid_stats.loss_delta)/len(valid_stats.loss_delta) if len(valid_stats.loss_delta)>0 else 0.
        print('\t D_Loss: %g; G_loss: %g; \t Real_Acc: %g; Fake_Acc: %g, Delta: %g' % (loss_d, loss_g, cls_racc, cls_facc, delta))

        return valid_stats

    def gan_dbm_forward(self, q_src, q_lens, r_src, r_lens):
        _, batch_size = q_src.size()
        loss_D, loss_G, delta, real_prob, feak_prob = self.model.forward_gan(
                       q_src, q_lens, r_src, r_lens) 
        loss_s2s, num_word, n_word_correct = self.eval_s2s(q_src, q_lens,
                                r_src, r_lens)
        batch_stat = self._gan_ael_stats(loss_s2s, num_word, n_word_correct,
                     loss_D.data.clone(), loss_G.data.clone(), delta.data.clone(),
                     real_prob.data.clone(), feak_prob.data.clone(), batch_size)
        return loss_D, loss_G, delta, batch_stat

    def gan_ael_validate(self, valid_iter):
        """
        """
        self.model.eval()
        total_stats = Statistics(loss_s2s=0, num_word=0, n_word_correct=0,
            cls_racc=[], cls_facc=[], loss_d=[], loss_g=[], loss_delta=[])
        report_stats = Statistics(loss_s2s=0, num_word=0, n_word_correct=0,
            cls_racc=[], cls_facc=[], loss_d=[], loss_g=[], loss_delta=[])
        # 
        for sub_dataset in valid_iter:
            batch_num = 0 
            num_batch = len(sub_dataset.examples) / sub_dataset.batch_size 
            for batch in sub_dataset:
                q_src, q_lens, r_src, r_lens = batch 
                if self.use_cuda:
                    q_src, q_lens, r_src, r_lens = (q_src.cuda(), q_lens.cuda(),
                    r_src.cuda(), r_lens.cuda())
                # forward 
                self.model.eval()
                _, batch_size = q_src.size()
                _, _, _, batch_stat = self.gan_dbm_forward(q_src, q_lens,
                      r_src, r_lens)
                # update state
                total_stats.update(batch_stat)
                report_stats.update(batch_stat)
                #
                batch_num += 1
            # end for 
        return total_stats

    def drop_checkpoint(self, epoch, batch_num, valid_stats):
        """ Save a resumable checkpoint.

        Args:
            opt (dict): option object
            epoch (int): epoch number
            ### fields (dict): fields and vocabulary
            valid_stats : statistics of last validation run
        """
        encoder = (self.model.encoder.module 
                     if isinstance(self.model.encoder, nn.DataParallel)
                     else self.model.encoder)
        decoder = (self.model.decoder.module 
                     if isinstance(self.model.decoder, nn.DataParallel)
                     else self.model.decoder)
        ael = (self.model.ael.module 
                     if isinstance(self.model.ael, nn.DataParallel)
                     else self.model.ael)
        generator = (self.model.generator.module 
                     if isinstance(self.model.generator, nn.DataParallel)
                     else self.model.generator)
        disor = (self.model.disor.module 
                     if isinstance(self.model.disor, nn.DataParallel)
                     else self.model.disor)
        encoder_dict, decoder_dict, ael_dict = ( encoder.state_dict(),
                   decoder.state_dict(), ael.state_dict() )
        generator_dict, disor_dict = ( generator.state_dict(), disor.state_dict() )

        checkpoint = {
            'encoder': encoder_dict,
            'decoder': decoder_dict,
            'ael': ael_dict,
            'generator': generator_dict,
            'discor': disor_dict,
            'opt': self.model_opt,
            'epoch': epoch,
            's2s_optim': self.s2s_opt,
            'dec_optim': self.dec_opt,
            'discrim_optim': self.discrim_opt,
            'generator_optim': self.generator_opt,
        }
        if batch_num is None:
            torch.save(checkpoint, '%s_acc_%.2f_ppl_%.2f_e%d.pt'
                 % (self.opt.save_model, valid_stats.word_accuracy(), valid_stats.xent_s2s(), epoch))
        else:
            torch.save(checkpoint, '%s_acc_%.2f_ppl_%.2f_e%d-%d.pt'
                 % (self.opt.save_model, valid_stats.word_accuracy(), 
                 valid_stats.xent_s2s(), epoch, batch_num))
        # end of drop_checkpoint
