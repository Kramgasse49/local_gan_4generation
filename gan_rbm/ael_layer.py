#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch.nn as nn 
import torch.nn.functional as F 
import torch 

class ApproxEmbedding(nn.Module):
    """ approximating the word embedding using the word distribution 
        and the embedding matrix. 
    """
    def __init__(self, embedding):
        """
        embedding _layers, map word_id to vectors 
        """
        super(ApproxEmbedding, self).__init__()
        self.embed_W = embedding 
    def forward(self, inputs):
        """
          inputs,  word distributions,  shaped [1, batch_size, vocab_size]
            Returns:
             the approximated vectors of words
              [1, batch, embed_dim]
        """
        assert inputs.size(0) == 1
        inputs = inputs.squeeze(0)
        approxi_embed = torch.mm(F.softmax(inputs,dim=1), 
                         self.embed_W.embeddings.weight)
        return approxi_embed.unsqueeze(0)

