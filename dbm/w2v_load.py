# -*- coding:utf-8 -*-
"""
    Word Embedding Types:
        (1) Glove, txt format
        (2) Google Bin
        (3) Gensim
"""
import numpy as np

def load_glove_txt(fpath):
    """
    """
    # 
    word2id_dict = {}
    embedding_lists = []
    
    # load
    with open(fpath, 'r') as f:
        line_no = 0
        for line in f:
            items = line.strip().split()
            word2id_dict[items[0].decode('utf-8')] = len(word2id_dict)
            embedding_lists.append( map(float, items[1:]) )
            line_no += 1
        print("Read {} line".format(line_no))
    # 
    return word2id_dict, embedding_lists # np.asarray(embedding_lists) 
"""
n [1]: from w2v_load import *

In [2]: w2id_dict, embeddings = load_glove_txt('glove.twitter.27B.50d.txt')
Read 1193514 line

In [3]: len(w2id_dict)
Out[3]: 1193514

In [4]: embeddings.shape()

In [5]: embeddings.shape
Out[5]: (1193514, 50)

"""
