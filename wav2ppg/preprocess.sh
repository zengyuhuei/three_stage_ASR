#!/usr/bin/env bash

. ./cmd.sh
. ./path.sh || exit 1;

dataset=matbn_new
stage=5
chain_dir=exp/chain/cnn_tdnnf_sp_1b/
gmm_dir=exp/tri3b
lang_dir=data/lang
dict_dir=data/local/dict
tree_dir=exp/chain/tree_sp
nj=16

. utils/parse_options.sh

if [ $stage -le 0 ]; then
  # data preparation
  #./local/tcc300_data_prep.sh /mnt/hdd18.2t/dataset/TCC300/tcc300 data/tcc300_new
  # 重新斷詞
  for type in test; do
    echo "normalizing transcription for $type."
    sed -i 's/UK/UNK/g' data/$dataset/$type/text
    mkdir -p data/$dataset/$type/tmp
    if [ ! -f data/$dataset/$type/tmp/text.backup ]; then
      cp data/$dataset/$type/text data/$dataset/$type/tmp/text.backup
    fi
    cat data/$dataset/$type/tmp/text.backup | tr "[:lower:]" "[:upper:]" > data/$dataset/$type/text.raw

    cut -d " " -f 2- data/$dataset/$type/text.raw | \
    sed 's/\([A-Z]\+\)/ \1 /g' | tr " " "\n" | sort -u | grep [A-Z] \
    > data/$dataset/$type/tmp/vocab-en

    comm -23 data/$dataset/$type/tmp/vocab-en $dict_dir/cmudict/words.txt > data/$dataset/$type/tmp/oov-en

    g2p.py --model=conf/g2p_model --apply data/$dataset/$type/tmp/oov-en \
    | tr "\t" " " > data/$dataset/$type/tmp/lexicon-en-oov-g2p.txt || exit 1;

    cat data/$dataset/$type/tmp/lexicon-en-oov-g2p.txt $dict_dir/cmudict/lexicon-letter.txt |\
    sort -u > data/$dataset/$type/tmp/lexicon-en.txt

    python local/generate_newword_lexicon.py \
    $dict_dir/cedict/ch-char-dict.txt \
    data/$dataset/$type/tmp/lexicon-en.txt \
    data/$dataset/$type/tmp/oov-en \
    | sort  -u > data/$dataset/$type/tmp/lexicon-en-oov.txt

    cut -d " " -f 1 $dict_dir/lexicon.txt | cat - data/$dataset/$type/tmp/vocab-en \
    > data/$dataset/$type/tmp/user_dict
    cat data/$dataset/$type/text.raw |\
    python local/character_tokenizer_keep_vocab.py --vocab data/$dataset/$type/tmp/user_dict \
    > data/$dataset/$type/text
    sed -i 's/着/著/g' data/$dataset/$type/text
  done
fi

if [ $stage -le 1 ]; then
  echo "$0: creating MFCC features" 
  for type in test; do
    steps/make_mfcc_pitch_online.sh --nj 16 data/$dataset/$type
    utils/fix_data_dir.sh data/$dataset/$type
    steps/compute_cmvn_stats.sh data/$dataset/$type
  done
fi

if [ $stage -le 2 ]; then
  echo "$0: creating high-resolution MFCC features"
  for type in test; do
    utils/copy_data_dir.sh data/$dataset/$type data/$dataset/${type}_hires
    steps/make_mfcc.sh --nj $nj --mfcc-config conf/mfcc_hires.conf \
        --cmd "$train_cmd" data/$dataset/${type}_hires || exit 1;
    steps/compute_cmvn_stats.sh data/$dataset/${type}_hires || exit 1;
    utils/fix_data_dir.sh data/$dataset/${type}_hires
  done
fi

if [ $stage -le 3 ]; then  
  echo "$0: punctuating" 
  for type in test_hires; do
    ./local/phone_embedding/re_punctuate.sh data/$dataset/$type/text data/$dataset/$type/punctuated_text
  done
fi

if [ $stage -le 4 ]; then  
  echo "$0: Creating ctm" 
  for type in test; do
    # get alignment
    steps/align_fmllr.sh --nj $nj --beam 100 --retry-beam 10000 --cmd run.pl data/$dataset/$type $lang_dir $gmm_dir $gmm_dir/${dataset}_${type}_ali
    # get ctm 
    ./steps/get_train_ctm.sh --cmd run.pl --stage 0 --use-segments false data/$dataset/$type $lang_dir $gmm_dir/${dataset}_${type}_ali 
  done
fi

if [ $stage -le 5 ]; then  
  echo "$0: Creating phone post" 
  for type in test_hires dev_hires train_hires; do
    #get phone post
    ./steps/nnet3/chain/get_phone_post.sh --use-gpu true --nj $nj \
       $tree_dir $chain_dir $lang_dir data/$dataset/$type $chain_dir/${dataset}_${type}_phone_post
  done
fi

if [ $stage -le 6 ]; then  
  echo "$0: Creating word frame position" 
  for type in test_hires dev_hires train_hires; do
    python ./local/phone_embedding/word_frame_pos.py --dataset $dataset
  done
fi
