#!/bin/sh
## current working directory
#$ -cwd
#$ -l f_node=1
#$ -l h_rt=24:00:00
#$ -N train_transformer
#$ -m abe
#$ -M kopamaru@gmail.com
#$ -o o.train_transformer
#$ -e e.train_transformer

## Initialize module command (don't remove)
. /etc/profile.d/modules.sh

module load cuda/9.0.176
module load cudnn/7.3

source ~/fairseq_modified/venv/bin/activate

python ~/fairseq_modified/train.py /gs/hs0/tga-nlp-titech/matsumaru/data/jnc_fairseq_3snt_100k_test_bin/ \
--arch transformer_wmt_en_de \
--max-epoch 15 \
--lr 0.0005 --min-lr 1e-09 \
--optimizer adam --adam-betas '(0.9, 0.98)' \
--update-freq 2 \
--max-tokens 3584 \
--dropout 0.1 \
--clip-norm 0.0 \
--lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
--save-dir /gs/hs0/tga-nlp-titech/matsumaru/entasum/fairseq_model/jnc_3snt_transformer_100k_test_wmtset_d01_upfreq2
