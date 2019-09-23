from __future__ import unicode_literals, print_function, division
"""
This is the loadable seq2seq trainer library that is
in charge of training details, loss compute, and statistics.
See train.py for a use case of this library.

Note!!! To make this a general library, we implement *only*
mechanism things here(i.e. what to do), and leave the strategy
things to users(i.e. how to do it). Also see train.py(one of the
users of this library) for the strategy things we do.
"""
import time
import sys
import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class Statistics(object):
    """
    Accumulator for loss statistics.
    Currently calculates:

    * accuracy
    * perplexity
    * elapsed time
    """
    def __init__(self, loss=0, n_sample=0, n_correct=0):
        self.loss = loss
        self.n_sample = n_sample
        self.n_correct = n_correct
        self.start_time = time.time() 

    def update(self, stat):
        self.loss += stat.loss
        self.n_sample += stat.n_sample
        self.n_correct += stat.n_correct

    def accuracy(self):
        return 100 * (self.n_correct / self.n_sample)

    def xent(self):
        return self.loss / self.n_sample

    def elapsed_time(self):
        return time.time() - self.start_time

    def output(self, epoch, batch, n_batches, start):
        """Write out statistics to stdout.

        Args:
           epoch (int): current epoch
           batch (int): current batch
           n_batch (int): total batches
           start (int): start time of epoch.
        """
        t = self.elapsed_time()
        print(("Epoch %2d, %5d/%5d; acc: %6.2f; xent: %6.4f; " +
               "%3.0f sample /s; %6.0f s elapsed ") %
              (epoch, batch, n_batches,
               self.accuracy(),
               self.xent(),
               self.n_sample / (t + 1e-5),
               time.time() - start))
        sys.stdout.flush()

    def log(self, prefix, experiment, lr):
        t = self.elapsed_time()
        experiment.add_scalar_value(prefix + "_accuracy", self.accuracy())
        experiment.add_scalar_value(prefix + "_tgtper",  self.n_sample / t)
        experiment.add_scalar_value(prefix + "_lr", lr)
     
    def log_tensorboard(self, prefix, writer, lr, step):
        t = self.elapsed_time()
        writer.add_scalar(prefix + "/xent", self.xent(), step)
        writer.add_scalar(prefix + "/accuracy", self.accuracy(), step)
        writer.add_scalar(prefix + "/tgtper",  self.n_sample / t, step)
        writer.add_scalar(prefix + "/lr", lr, step)

############################### Seq2Seq Input Trainer

class DoubleTrainer(object):
    """
    Class that controls the training process.

    Args:
            model(:py:class:`onmt.Model.NMTModel`): translation model to train

            optimizer(:obj:`onmt.Optim.Optim`):
               the optimizer responsible for update
            norm_method(string): normalization methods: [sents|tokens]
    """

    def __init__(self, model, optimizer, criterion, use_cuda, norm_method="sents"):
        # Basic attributes.
        self.model = model
        self.optimizer = optimizer
        self.norm_method = norm_method
        self.progress_step = 0
        self.use_cuda = use_cuda
        self.criterion = criterion

    def train(self, train_iter, epoch, report_func=None):
        """ Train next epoch.
        Args:
            train_iter: training data iterator
            epoch(int): the epoch number
            report_func(fn): function for logging

        Returns:
            stats (:obj:`onmt.Statistics`): epoch loss statistics
        """
        # Set model in training mode.
        self.model.train()
        total_stats = Statistics(loss=0, n_sample=0, n_correct=0)
        report_stats = Statistics(loss=0, n_sample=0, n_correct=0)

        for sub_dataset in train_iter:
            batch_num = 0
            num_batch = len(sub_dataset.examples) / sub_dataset.batch_size
            for batch in sub_dataset:
                q_src, q_src_lengths, r_src, r_src_lengths = batch 
                if self.use_cuda:
                    q_src, q_src_lengths, r_src, r_src_lengths = ( q_src.cuda(),
                                              q_src_lengths.cuda(), r_src.cuda(),
                                              r_src_lengths.cuda() ) 
                self.optimizer.optimizer.zero_grad()
                _, batch_size = q_src.size()
                batch_stats, loss, _ = self.forward_step(q_src, q_src_lengths, 
                                        r_src, r_src_lengths)
                # 4. Update the parameters and statistics.
                loss.div(batch_size).backward()
                self.optimizer.step()
                self.progress_step += 1

                total_stats.update(batch_stats)
                report_stats.update(batch_stats)
                batch_num += 1
                if report_func is not None:
                    report_stats = report_func(
                        epoch, batch_num, num_batch,
                        self.progress_step,
                        total_stats.start_time, 0.001,
                        report_stats)
                # 
        return total_stats

    def epoch_step(self, ppl, epoch):
        return self.optimizer.update_learning_rate(ppl, epoch)

    def _bottle(self, v):
        return v.view(-1, v.size(2))

    def _stats(self, loss, scores, target):
        """
        Args:
            loss (:obj:`FloatTensor`): the loss computed by the loss criterion.
            scores (:obj:`FloatTensor`): a score for each possible output
            target (:obj:`FloatTensor`): true targets

        Returns:
            :obj:`Statistics` : statistics for this batch.
        """
        mask = (target > 0)
        pred = scores.max(1)[1]
        num_correct = (pred.eq(target)*mask).sum()
        n_sample = mask.sum()
        return Statistics(loss=loss[0], n_sample=n_sample,
               n_correct=num_correct)

    def validate(self, valid_iter):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`onmt.Statistics`: validation loss statistics
        """
        # Set model in validating mode.
        self.model.eval()

        valid_stats = Statistics(loss=0., n_sample=0, n_correct=0)

        for sub_dataset in valid_iter:
            for batch in sub_dataset:
                q_src, q_src_lengths, r_src, r_src_lengths = batch 
                if self.use_cuda:
                    q_src, q_src_lengths, r_src, r_src_lengths = (q_src.cuda(),
                                              q_src_lengths.cuda(), r_src.cuda(),
                                              r_src_lengths.cuda()) 
                _, batch_size = q_src.size()
                batch_stats, _, _ = self.forward_step(q_src, q_src_lengths,
                                    r_src, r_src_lengths)
                # Update statistics.
                valid_stats.update(batch_stats)
        # Set model back to training mode.
        self.model.train()

        return valid_stats

    def drop_checkpoint(self, opt, epoch, valid_stats):
        """ Save a resumable checkpoint.

        Args:
            opt (dict): option object
            epoch (int): epoch number
            ### fields (dict): fields and vocabulary
            valid_stats : statistics of last validation run
        """ # waiting 
        # real_model = (self.model.module
                      # if isinstance(self.model, nn.DataParallel)
                      # else self.model)
        # real_generator = (real_model.generator.module
                          # if isinstance(real_model.generator, nn.DataParallel)
                          # else real_model.generator)

        # model_state_dict = real_model.state_dict()
        # model_state_dict = {k: v for k, v in model_state_dict.items()
                            # if 'generator' not in k}
        # generator_state_dict = real_generator.state_dict()

        # checkpoint = {
            # 'model': model_state_dict,
            # 'generator': generator_state_dict,
            # 'opt': opt,
            # 'epoch': epoch,
            # 'optim': self.optimizer,
        # }
        encoder = (self.model.encoder.module 
                     if isinstance(self.model.encoder, nn.DataParallel)
                     else self.model.encoder)
        decoder = (self.model.decoder.module 
                     if isinstance(self.model.decoder, nn.DataParallel)
                     else self.model.decoder)
        generator = (self.model.generator.module 
                     if isinstance(self.model.generator, nn.DataParallel)
                     else self.model.generator)
        encoder_dict, decoder_dict, generator_dict = ( encoder.state_dict(),
                   decoder.state_dict(), generator.state_dict())
        checkpoint = {
            'encoder': encoder_dict,
            'decoder': decoder_dict,
            'generator': generator_dict,
            'opt': opt,
            'epoch': epoch,
            'optim': self.optimizer,
        }
        torch.save(checkpoint,
                   '%s_acc_%.2f_ppl_%.2f_e%d.pt'
                   % (opt.save_model, valid_stats.accuracy(),
                      valid_stats.xent(), epoch))

    def forward_step(self, q_src, q_src_lengths, tgt, tgt_lengths):
        # 2. F-prop all but generator.
        outputs, _, dec_final = \
                self.model(q_src, tgt, q_src_lengths) ## ????
        size = outputs.size()[:2] # _bottle
        word_dist = self.model.generator(outputs.view(size[0]*size[1], -1))
        # 3. Compute loss 
        loss = self.criterion(word_dist, tgt[1:].view(size[0]*size[1]))

        loss_data = loss.data.clone()
        batch_stats = self._stats(loss_data, word_dist.data, 
                                  tgt[1:].view(size[0]*size[1]).data)
        # 
        return batch_stats, loss, word_dist 

