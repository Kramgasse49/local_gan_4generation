
This directory contains the implementation of LocalGAN.

1, pretrain Seq2Seq
python train_s2s.py \
       -vocab_path vocab_path \
       -word_vec_size 300 \
       -encoder_type rnn \
       -decoder_type std -enc_layers 1 -dec_layers 1 \
       -hidden_size 256 -rnn_type GRU -rnn_size 256 \ 
       -data data_path \
       -save_model model_path \
       -gpuid 1 -batch_size 256 
       -optim adam -learning_rate 0.0005 \
       -pre_word_vecs embedding_path \ 
       -report_every 500

2, Pretraining DBM
(1) normalization, normalizing the sentence vector to [0, 1]
python train_norm.py \ 
    -tqf=train_query_path \
    -trf=train_response_path.txt \
    -vqf=valid_query_path.txt \
    -vrf=valid_response_path.txt \ 
    -vf=vocab_path \ 
    --data_name=data_name \ 
    --embedfile embedding_path 
    
    the embedding and the vocab_path should be corresponding with the pretraind S2S.
 
(2) pretraining DBM
python pretrain_qr_dbm.py \
    -tqf=train_query_path \
    -trf=train_response_path.txt \
    -vqf=valid_query_path.txt \
    -vrf=valid_response_path.txt \ 
    -vf=vocab_path \ 
    --data_name=data_name \ 
    --embedfile embedding_path 
    -m pretrain \
    --num_epoch 5 \
    --print_every 500

3, training LocalGAN
python train_dbm.py \
    -vocab_path vocab_path \
    -word_vec_size 300 \
    -encoder_type rnn \
    -decoder_type std \
    -enc_layers 1 \
    -dec_layers 1 \
    -hidden_size 128 \
    -rnn_type GRU \
    -rnn_size 256 \
    -data data_path \
    -save_model model_path \
    -gpuid 3 \
    -batch_size 128 \
    -optim adam \
    -learning_rate 0.001 \
    -report_every 50 \
    -train_from pretrained_s2s_path \
    -gan_loss_type LOG \
    -d_lr 0.00001 \
    -dec_lr 0.00001 \
    -rbm_path pretrained_dbm_path \
    -rbm_rq_prefix rq_2 \
    -rbm_qr_prefix qr_2 \
    -epochs 13 \
    -query_norm_path normalization_model_query \
    -reply_norm_path normalization_model_response \
    -d_train_ratio 5 \
    -g_train_ratio 1 

