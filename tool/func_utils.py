#!/usr/bin/env python
import io
import numpy as np 

############################## transformation between words and ids #########
id_to_words = lambda idxs, idx2word: [idx2word.get(idx) for idx in idxs]

words_to_id = lambda words, vocab: [vocab.word2idx.get(word, vocab.unkid)]

def line_to_id(line, vocab, max_len=None, pre_trunc=True):
    """Mapping words in line into index ids.
       Parameters:
         :param line, text, contains several words seperated by the space.
         :param vocab, an object {vocba.Vocab}
         :param max_len, the maximum number of words should be maintained.
         :param pre_trunc, bool, if true, pruning the left part of a line,
                                    false, keeping the left part.
    """
    wids = [vocab.word2idx.get(w, vocab.unkid) for w in line.strip().split()]
    if max_len is None:
        return len(wids), wids
    else:
        if pre_trunc:
            wids = wids[-max_len:]
        else:
            wids = wids[:max_len]
        return len(wids), wids
    # 

def check_unk(wids_list, unkid):
    """Check whether words in one batch are mostly unknown,
       if that, report this exception.
    """
    word_num = sum( [len(wids) for wids in wids_list] )
    unk_num = sum( [wids.count(unkid) for wids in wids_list] )
    return word_num, unk_num 

def text2id(source_file, out_path, vocab):
    '''
        each line contains only one-column containing word list.
    '''
    logger.info("Unknown Word ID: {}".format(vocab.unkid))
    with io.open(source_file, encoding='utf-8') as fin, open(out_path,'w', encoding='utf-8') as fout:
        for line in fin:
            words = line.strip().split()
            wids = words_to_id(words, vocab)
            out_line = ' '.join(map(str, wids))
            fout.write('{}\n'.format(out_line))
    # end fun
############################## Util Functions ###############################
to_utf8 = lambda x: x.encode('utf-8') 
def load_w2v_txt(fpath):
    """
    """
    # 
    word2id_dict = {}
    embedding_lists = []
    
    # load
    with io.open(fpath, 'r', encoding='utf-8') as f:
        line_no = 0
        for line in f:
            items = line.split()
            word2id_dict[items[0].strip()] = len(word2id_dict)
            embedding_lists.append( map(float, items[1:]) )
            line_no += 1
        print("Read {} line".format(line_no))
    # 
    return word2id_dict, embedding_lists

#################################### Sentence Padding #########################
def sentence_pad(sents, max_len=None, eos_id=None, sos_id=None):
    """Padding sentences to the same length, for paralleling computation,
        Inputs:
          sents, sentence list, each item is a word_id list.
          max_len, pre-defined max_length
          eos_id, id of the sentence end
          sos_id, id of the beginnig 
        Returns,
         sent_matrix, [sample_num, max_len], a numpy.narray
    """
    sample_num = len(sents)
    # get max_len of the input sentences 
    if max_len is None:
        lens = [len(sent) for sent in sents]
        max_len = max(lens)
    if eos_id is not None:
        max_len += 1
    if sos_id is not None:
        max_len += 1
    # format 
    X =(np.zeros((sample_num, max_len))).astype(np.int32)
    for ind, words in enumerate(sents):
        if sos_id is not None:
            words = [sos_id] + words 
        if eos_id is not None:
            words.append(eos_id)
        sent_len = len(words)
        X[ind, :sent_len] = words 
    return X 

