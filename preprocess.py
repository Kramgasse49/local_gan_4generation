# -*- coding:utf-8 -*-
import argparse
import numpy as np 
import os, sys, io
import torch

from tool.vocab import Vocab 
from tool.datasetbase import DataSet 
from tool.preprocess_opt import preprocess_opts
from tool.func_utils import line_to_id, load_w2v_txt

def build_save_text_dataset_in_shards(corpus_list, opt, corpus_type):
    corpus_size = os.path.getsize(corpus_list[0])
    if corpus_size > 10 * (1024**2) and opt.max_shard_size == 0:
        print("Warning. The corpus %s is larger than 10M bytes, you can "
              "set '-max_shard_size' to process it by small shards "
              "to use less memory." % src_corpus)
    # 
    if opt.max_shard_size != 0:
        print(' * divide corpus into shards and build dataset separately'
              '(shard_size = %d bytes).' % opt.max_shard_size)
    # load vocab 
    vocab = torch.load(opt.vocab_path)
    # open files
    f_list = [io.open(corpus, "r", encoding="utf-8") for corpus in corpus_list]
    examples, line_index, last_pos, out_index = ([], 0, 0, 1)

    while True:
        if opt.max_shard_size != 0 and line_index % 64 == 0:
            cur_pos = f_list[0].tell()
            if cur_pos >= last_pos + opt.max_shard_size:
                last_pos = cur_pos
                torch.save(DataSet(examples), 
                       '{}.{}.{}.pt'.format(opt.save_data, corpus_type, out_index))
                print('Saving {}.{}.{}.pt'.format(opt.save_data, corpus_type, out_index))
                examples = []
                out_index += 1
        lines = [ f.readline() for f in f_list]
        if lines[0] == '':
            break
        temp = []
        for i, line in enumerate(lines[:-1]):
            w_num, wids = line_to_id(line.strip(), 
                    vocab, max_len=opt.seq_length, pre_trunc=True)
            temp.extend([w_num, wids])
        if opt.with_label:
            examples.append( temp + [float(lines[-1])] )
        else:
            w_num, wids = line_to_id(lines[-1].strip(), 
                    vocab, max_len=opt.seq_length, pre_trunc=False)
            examples.append( temp + [w_num, wids])
        line_index += 1
    if len(examples) > 0:
        torch.save(DataSet(examples), 
                   '{}.{}.{}.pt'.format(opt.save_data, corpus_type, out_index))
        print('Saving {}.{}.{}.pt'.format(opt.save_data, corpus_type, out_index))
    # 
##################################################################################
def build_save_vocab(opt):
    if os.path.exists(opt.vocab_path):
        print("vocabulary has exists, {}".format(opt.vocab_path))
        return 
    # build vocabulary
	print('Building Vocabulary ... ...')
    corpus_list = opt.train_corpus_path
    if opt.with_label:
        corpus_list = corpus_list[:-1]
    vocab = Vocab()
    vocab.build_vocab(corpus_list,
                      min_count=opt.words_min_frequency, 
                      max_size=opt.vocab_size)
    torch.save(vocab, opt.vocab_path)

    # format pretrain word2vec 
    if opt.pre_word_vecs_path is not None and os.path.exists(opt.pre_word_vecs_path):
        word2id_pretrain, embedding_lists = load_w2v_txt(opt.pre_word_vecs_path)
        embed_lens = [len(embed) for embed in embedding_lists]
        print(max(embed_lens), min(embed_lens))
        W = []
        rev_vocab = vocab.idx2word
        not_in_pretrained = 0
        embed_dim = len(embedding_lists[0])
        for idx in range(len(rev_vocab)):
            if rev_vocab[idx] in word2id_pretrain:
                W.append(embedding_lists[word2id_pretrain[rev_vocab[idx]]])
            else:
                not_in_pretrained += 1
                if idx == 0:
                    W.append((np.random.randn(embed_dim)*0.).tolist())
                else:
                    W.append(np.random.randn(embed_dim).tolist())
        #
        print("Dim: {},Vocab_size:{}, Not in Glove Number:{}".format(embed_dim, len(rev_vocab),
                            not_in_pretrained))
        #
        word_embeddings = np.asarray(W)
        print(word_embeddings.shape)
        np.save(opt.vocab_path+'w2v.npy', word_embeddings)
    # end if   

def build_save_dataset(corpus_type, opt):
    assert corpus_type in ['train', 'valid', 'test']

    if corpus_type == 'train':
        corpus_list = opt.train_corpus_path
    elif corpus_type == 'valid':
        corpus_list = opt.valid_corpus_path
    else:
        corpus_list = opt.valid_corpus_path

    # Currently we only do preprocess sharding for corpus: data_type=='text'.
    return build_save_text_dataset_in_shards(corpus_list, opt, corpus_type)

def main():
    # get optional parameters
    parser = argparse.ArgumentParser(description='preprocess.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    preprocess_opts(parser)
    opt = parser.parse_args()

    # 
    print("Building & saving vocabulary...")
    build_save_vocab(opt)

    print("Building & saving training data...")
    build_save_dataset('train', opt)

    print("Building & saving validation data...")
    build_save_dataset('valid', opt)

    # print("Building & saving test data...")
    # build_save_dataset('test', opt)


if __name__ == "__main__":
    main()
