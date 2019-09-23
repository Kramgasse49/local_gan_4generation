import torch 
from torch.autograd import Variable
import numpy as np 
import io 
from func_utils import line_to_id
from func_utils import sentence_pad 

class DataSet(object):
    '''
        subset of a very large dataset,
        #properties
        examples, sample list, each sample is also a list including multi-columns.
                  for a classification task, the last element of the sample features is the label
                  # [ first_col_len, first_col, second_col_len, second_col, ...  [label] ]
        batch_size,
        with_label,
    '''
    def __init__(self, examples):
        self.examples = examples

    def set_property(self, batch_size=128, 
                     with_label=False, rank=True,
                     eos_id=None, sos_id=None):
        """
            this method should be called before iteration, 
            batch_size, 
            with_label, 
            rank, according to the sequential length, rerank one batch
        """
        self.batch_size = batch_size
        self.with_label = with_label
        self.rank = rank 
        self.sample_num = len(self.examples)
        self.eos_id = eos_id
        self.sos_id = sos_id

    def __iter__(self):
        for ind in range(0, self.sample_num, self.batch_size):
            batch = self.examples[ind:ind+self.batch_size]
            yield self.format(batch)

    def format(self, batch):
        # rank
        if self.rank:
            batch = sorted(batch, key=lambda d:d[0], reverse=True)
        t = list(zip(*batch))
        # with label
        if self.with_label:
            label = t[-1]
            t = t[:-1]
        #
        outputs = None 
        for i in range(0, len(t)-1, 2):
            txt_len, txt = (t[i], t[i+1])
            max_len = max(txt_len)
            txt = sentence_pad(txt, max_len=max_len, eos_id=self.eos_id, sos_id=self.sos_id)
            # 
            txt = Variable(torch.from_numpy(txt.T.astype(np.int64)))
            txt_len = torch.LongTensor(txt_len).view(-1)
            if self.eos_id is not None:
               txt_len += 1
            if self.sos_id is not None:
                txt_len += 1
            if outputs is None:
                outputs = (txt, txt_len)
            else:
                outputs = outputs + (txt, txt_len)
        if self.with_label:
            outputs = outputs + (Variable(torch.FloatTensor(label).view(-1)), )

        return outputs


class TestDataset(object):
    """
    """
    def __init__(self, vocab, srcfiles, batch_size, 
                       max_len=None, pre_trunc=True, 
                       with_label=True,
                       eos_id=None, sos_id=None):
        """
        """
        self.vocab = vocab
        self.srcfiles = [io.open(srcfile, 'r', encoding='utf-8') for srcfile in srcfiles]
        self.batch_size = batch_size
        self.max_len = max_len
        self.pre_trunc = pre_trunc 
        self.with_label = with_label
        self.unk_ids = vocab.unkid
        self.eos_id = eos_id
        self.sos_id = sos_id

    def __iter__(self):
        """
        """
        batch_samples = []
        examples = []
        for lines in zip(*self.srcfiles):
            temp = []
            for i, line in enumerate(lines[:-1]):
                w_num, wids = line_to_id(line.strip(), self.vocab, 
                              max_len=self.max_len, pre_trunc=self.pre_trunc)
                temp.extend([w_num, wids])
            if self.with_label:
                examples.append( temp + [float(lines[-1].strip())] )
            else:
                w_num, wids = line_to_id(lines[-1].strip(), self.vocab, 
                              max_len=self.max_len, pre_trunc=False)
                examples.append( temp + [w_num, wids])
            if len(examples) == self.batch_size:
                yield self.format(examples)
                examples = []
        if len(examples) > 0:
            yield self.format(examples)

    def format(self, batch):
        # rank
        t = list(zip(*batch))
        # with label
        if self.with_label:
            label = t[-1]
            t = t[:-1]
        #
        outputs = None 
        for i in range(0, len(t)-1, 2):
            txt_len, txt = (t[i], t[i+1])
            max_len = max(txt_len)
            txt = sentence_pad(txt, max_len=max_len, eos_id=self.eos_id, sos_id=self.sos_id)
            # to Variable 
            txt = Variable(torch.from_numpy(txt.T.astype(np.int64)))
            txt_len = torch.LongTensor(txt_len).view(-1)
            if self.eos_id is not None:
               txt_len += 1
            if self.sos_id is not None:
                txt_len += 1
            if outputs is None:
                outputs = (txt, txt_len)
            else:
                outputs = outputs + (txt, txt_len)
        if self.with_label:
            outputs = outputs + (Variable(torch.FloatTensor(label).view(-1)), )
        return outputs