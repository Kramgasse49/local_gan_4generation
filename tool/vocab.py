# -*- coding: utf-8 -*-
"""
    Manage Vocabulary,
        1), Build Vocabulary
        2), save and load vocabulary
        3), 
"""
import os, sys, json
import logging
import codecs
import io

logger = logging.getLogger(__name__)

PAD = u'<pad>'
SOS = u'<sos>'
EOS = u'<eos>'
UNK = u'<unk>'

class Vocab(object):
    '''
    '''
    def __init__(self):
        self.init_vocab()
    
    def init_vocab(self):
        self.word2idx = {}
        self.idx2word = {}
        self.word2idx[u'<pad>'] = 0
        self.word2idx[u'<sos>'] = 1
        self.word2idx[u'<eos>'] = 2
        self.word2idx[u'<unk>'] = 3

    def build_vocab(self, corpus_path_list, min_count=None, max_size=None):
        '''

            corpus_path_list, the path of corpus list
                each line includes multi-column sequence (separated by '\t')
                Words in each sequence are separated by space ' '.
            min_count, 
                according to min_count, this fun should remove words 
                who's number is lower than that in the process of building vocabulary.
                default is None, maintaining all words.
                (int, option 5, 10, ...)
        '''
        # count word
        word_count = {}
        for corpus_path in corpus_path_list:
            # check there exists the corpus_path or not.
            if not os.path.exists(corpus_path):
                logger.info('The path {} doese not exist for building vocabulary'.format(corpus_path))
                continue
            with io.open(corpus_path, "r", encoding='utf-8') as f:
                for line in f:
                    words = line.strip().replace('\t', ' ').split()
                    for word in words:
                        word_count[word] = word_count.get(word, 0) + 1
                        # or try ... except ... 
        # cut with min_count
        if min_count is None:
            word_list = word_count.items()
        else:
            word_list = [(word, count) for word, count in word_count.items() if count > min_count]
        # sort word
        ranked_word_list = sorted(word_list, key=lambda d:d[1], reverse=True)
        if max_size is not None:
            ranked_word_list = ranked_word_list[:max_size]

        # build word2idx and idx2word
        self.init_vocab()
        for word, _ in ranked_word_list:
            self.word2idx[word] = len(self.word2idx)
        logger.info("original vocab {}; pruned to {} with min_count {}".
              format(len(word_count), len(self.word2idx), min_count))
        self.idx2word = {v: k for k, v in self.word2idx.items()}
    
    def load_vocab(self, vocab_path):
        if isinstance(vocab_path, dict):
            word2idx_t = vocab_path
        else:
            word2idx_t = json.load(open(vocab_path, "r"))
        self.word2idx = {}
        for k, v in word2idx_t.items():
            # self.word2idx[k] = v
            try:
                self.word2idx[k.encode('utf-8')] = v
            except:
                self.word2idx[k] = v
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        logger.info("Load vocabulary from {}".format(vocab_path))
    
    def save_vocab(self, vocab_path):
        with open(vocab_path, "w") as fout, open(vocab_path+'.txt', 'w') as ft:
            json.dump(self.word2idx, fout) 
            for k, v in self.word2idx.items():
                ft.write('{}\t{}\n'.format(k, v))
        logger.info("Save vocabulary to {}".format(vocab_path))

    @property
    def unkid(self):
        """return the id of unknown word
        """
        return self.word2idx.get(UNK, 0)

    @property
    def padid(self):
        """return the id of padding
        """
        return self.word2idx.get(PAD, 0)

    @property
    def sosid(self):
        """return the id of padding
        """
        return self.word2idx.get(SOS, 0)

    @property
    def eosid(self):
        """return the id of padding
        """
        return self.word2idx.get(EOS, 0)
