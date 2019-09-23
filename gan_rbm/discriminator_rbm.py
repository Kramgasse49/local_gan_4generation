# coding:utf-8
import torch
import torch.nn as nn

class Discriminator(nn.Module):
    """Discriminator """
    def __init__(self, emb_dim, rq_rbm, qr_rbm, 
        hidden_size,class_num=2, dropout_p=0.5):
        super(Discriminator, self).__init__()
        self.class_num = class_num
        self.hidden_size = hidden_size
        self.rq_rbm = rq_rbm # TextCNN(emb_dim, filter_num, filter_sizes)
        self.qr_rbm = qr_rbm # TextCNN(emb_dim, filter_num, filter_sizes)
        self.judger = nn.Sequential(
                        nn.Linear((self.rq_rbm.h_size + self.qr_rbm.h_size), hidden_size),
                        nn.ReLU(),
                        nn.Dropout(p=dropout_p),
                        nn.Linear(hidden_size, 1),
                    )

    def forward(self, query, response):
        """
        Args:
            - **query** [B, max_len, emb_dim]
        Output:
            The probability of real
        """
        rq = self.rq_rbm.rq_sample_hr(query, response) # [B, T, D] -> [B, all_features]
        qr = self.qr_rbm.rq_sample_hr(response, query)
        inputs = torch.cat((rq, qr), 1)
        return self.judger(inputs)

