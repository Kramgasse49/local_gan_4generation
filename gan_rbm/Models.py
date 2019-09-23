#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
from modules.utils import sequence_mask

class GANRBM(nn.Module):
    """GAN with an embedding approximating layer,
      for sequence generation tasks. 
    """
    def __init__(self, encoder, decoder, ael, generator, discriminator,
            dec_max_len=10, type_loss='General', q_normalizer=None, r_normalizer=None):
        """
        """
        super(GANRBM, self).__init__()
        self.encoder = encoder
        self.decoder = decoder 
        self.ael = ael 
        self.generator = generator 
        self.disor = discriminator 
        self.dec_max_len = dec_max_len 
        self.type_loss = type_loss  
        self.q_normalizer = q_normalizer
        self.r_normalizer = r_normalizer
        # parameter for embedding approximation
        # lack of parameters of the noise generator 
    def encode_fn(self, query, query_lens):
        query_state, memory_blank = self.encoder(query, query_lens)
        enc_state = self.decoder.init_decoder_state(query, memory_blank, query_state)
        return enc_state, memory_blank 

    def forward_s2s(self, query, query_lens, reply, reply_lens, dec_state=None):
        """
         query, reply, [seq_len, batch]
          query_lens, reply_lens, [seq_len]
        """
        #
        tgt = reply[:-1]
        #query_state, memory_blank = self.encoder(query, query_lens)
        #enc_state = self.decoder.init_decoder_state(query, memory_blank, query_state)
        enc_state, memory_blank = self.encode_fn(query, query_lens)
        decoder_outputs, dec_state, attns = self.decoder(tgt, memory_blank,
                enc_state if dec_state is None else dec_state,
                memory_lengths=query_lens)
        return decoder_outputs, attns, dec_state  

    def generate_fake(self, go_inp, state=None, memory_blank=None):
        """generate responses using AEL layer or greedy search.
          go_inp, [1, batch]
          state, (h_n, [c_n])
          Returns:
          fake reply embedding,  [
        """
        dec_inp = go_inp #  [1, batch]
        next_w_embed = self.decoder.embeddings(dec_inp) # [1, batch, emb_dim] 
        fake_reply_embed = []
        reply_word_dist = []

        for i in range(self.dec_max_len):
            output, state = self.decoder.rnn(next_w_embed, state)
            if self.decoder.attn_type is not None:
                output, p_attn = self.decoder.attn(output,
                        memory_blank.transpose(0,1),
                        memory_lengths=query_lens)
            word_dist = self.generator(output.squeeze(0)).unsqueeze(0)
            reply_word_dist.append(word_dist)
            if self.ael is not None:
                next_w_embed = self.ael(word_dist)
            else:
                dec_inp = torch.max(output.squeeze(1), dim=1, keepdim=True)[1]
                next_w_embed = self.decoder.embeddings(dec_inp)
            fake_reply_embed.append(next_w_embed)
        fake_reply_embed = torch.cat(fake_reply_embed, dim=0) 
        # check, batch first or length first
        reply_word_dist = torch.cat(reply_word_dist, dim=0)#??
        return fake_reply_embed, reply_word_dist 
        # 
    def forward_gan(self, query, query_lens, reply, reply_lens, dec_state=None):
        """
        """
        # get state of the encoder 
        enc_state, memory_blank = self.encode_fn(query, query_lens)
        if isinstance(self.decoder.rnn, nn.LSTM):
            state = enc_state.hidden
        else:
            state = enc_state.hidden[0]
        # prepare  decoder  
        fake_reply_embed, fake_word_dist = self.generate_fake(reply[0].unsqueeze(0), 
            state=state, memory_blank=memory_blank)
        real_reply_embed = self.decoder.embeddings(reply[1:])
        query_embed = self.encoder.embeddings(query) # (seq_len, batch, dim)
        # 
        fake_embed = fake_reply_embed.transpose(0, 1)
        real_embed = real_reply_embed.transpose(0, 1)
        query_embed = query_embed.transpose(0, 1)
        # mask fake 
        _, max_word_idx = fake_word_dist.topk(1, dim=-1)
        max_word_idx = max_word_idx.squeeze().transpose(0, 1)
        max_word_idx[:, -1] = 2
        fake_length = [(tt==2).nonzero()[0] for tt in max_word_idx]
        fake_length = torch.cat(fake_length).data + 1

        #fake_mask = sequence_mask(reply_lens, max_len=self.dec_max_len)
        fake_mask = sequence_mask(fake_length, max_len=self.dec_max_len) # fake_len
        fake_embed = fake_embed * Variable((fake_mask.unsqueeze(-1).expand_as(fake_embed)).float(), requires_grad=False)
        # 
        batch_size, _, embed_dim = query_embed.size()
        query_lens, reply_lens = ( query_lens.float(), reply_lens.float())
        rlens = Variable(reply_lens.expand(embed_dim, batch_size).transpose(0, 1))
        qlens = Variable(query_lens.expand(embed_dim, batch_size).transpose(0, 1))
        flens = Variable(fake_length.float().expand(embed_dim, batch_size).transpose(0, 1))

        query_embed_dis = torch.sum(query_embed, dim=1) / qlens
        real_embed_dis = torch.sum(real_embed, dim=1) / rlens 
        fake_embed_dis = torch.sum(fake_embed, dim=1) / flens
        #fake_embed_dis = torch.sum(fake_embed, dim=1) / rlens
        # normalization  
        if self.q_normalizer is not None and self.r_normalizer is not None:
            query_embed_dis = self.q_normalizer.transform(query_embed_dis)
            real_embed_dis = self.r_normalizer.transform(real_embed_dis)
            fake_embed_dis = self.r_normalizer.transform(fake_embed_dis)

        # judge which one is fake 
        real_logist = self.disor(query_embed_dis, real_embed_dis)
        fake_logist = self.disor(query_embed_dis, fake_embed_dis)
        # calculate loss 
        #TODO add length penalty
        length_ratio = F.relu(Variable(reply_lens) - Variable(fake_length.float())) / Variable(reply_lens)
        loss_penalty = 1.0 - length_ratio
        # 
        real_prob = F.sigmoid(real_logist)
        fake_prob = F.sigmoid(fake_logist)
        if self.type_loss == "LOG":
            loss_D = -torch.mean(torch.log(real_prob)
                    + torch.log(1-fake_prob)) / 2. 
            #loss_G = -torch.mean(torch.log(fake_prob))
            #loss_G = -torch.mean(torch.log(fake_prob)*loss_penalty)
            loss_G = -torch.mean(torch.log(fake_prob) / loss_penalty)
        elif self.type_loss == "Sqrt":
            loss_D = -torch.mean(torch.log(real_prob)
                    + torch.log(1-fake_prob)) / 2. 
            loss_G = torch.mean((real_logist - fake_logist)**2)
        else:
            loss_D = torch.mean(fake_logist - real_logist)
            #loss_G = torch.mean(-fake_logist*loss_penalty)
            loss_G = torch.mean(-fake_logist / loss_penalty)
        # local distribution variance, delta
        # normalization all the avg_embeddings.
        rc = self.disor.qr_rbm.hr_sample_r( self.disor.qr_rbm.q_sample_hr(query_embed_dis) )
        free_rc = self.disor.qr_rbm.hr_free_energy_one(rc, query_embed_dis) + self.disor.rq_rbm.hr_free_energy_one(query_embed_dis, rc)
        free_r_pos = self.disor.qr_rbm.hr_free_energy_one(real_embed_dis, query_embed_dis) + self.disor.rq_rbm.hr_free_energy_one(query_embed_dis, real_embed_dis)
        free_f_pos = self.disor.qr_rbm.hr_free_energy_one(fake_embed_dis, query_embed_dis) + self.disor.rq_rbm.hr_free_energy_one(query_embed_dis, fake_embed_dis)
        # 
        delta = torch.mean(F.relu((free_r_pos - free_f_pos)/free_rc))

        loss_G = loss_G + 10*delta
        return  loss_D, loss_G, 10*delta, real_prob, fake_prob 
