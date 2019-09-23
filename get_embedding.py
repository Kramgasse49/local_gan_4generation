#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys 

import torch 
import numpy as np 

model_path = sys.argv[1]
out_path = sys.argv[2]

params_dict = torch.load(model_path)

encoder_dict = params_dict['encoder']

embed_weight = encoder_dict['embeddings.embeddings.weight']

embedding = embed_weight.cpu().numpy()

np.save(out_path, embedding)
