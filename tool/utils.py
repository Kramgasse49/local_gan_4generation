#!/usr/bin/env python
import io
import torch 
from datasetbase import DataSet 
import glob

def lazily_load_dataset(corpus_type, opt):
    """
    Dataset generator. Don't do extra stuff here, like printing,
    because they will be postponed to the first loading time.

    Args:
        corpus_type: 'train' or 'valid'
    Returns:
        A list of dataset, the dataset(s) are lazily loaded.
    """
    assert corpus_type in ["train", "valid"]

    def lazy_dataset_loader(pt_file, corpus_type):
        dataset = torch.load(pt_file)
        print('Loading %s dataset from %s, number of examples: %d' %
              (corpus_type, pt_file, len(dataset.examples)))
        return dataset

    # Sort the glob output by file name (by increasing indexes).
    pts = sorted(glob.glob(opt.data + '.' + corpus_type + '.[0-9]*.pt'))
    if pts:
        for pt in pts:
            dataset = lazy_dataset_loader(pt, corpus_type)
            dataset.set_property(batch_size=opt.batch_size, 
                                 with_label=False, rank=False,
                                 eos_id=2, sos_id=1)
            yield dataset
    else:
        # Only one onmt.io.*Dataset, simple!
        pt = opt.data + '.' + corpus_type + '.pt'
        dataset = lazy_dataset_loader(pt, corpus_type)
        dataset.set_batch_size(opt.batch_size)
        yield dataset