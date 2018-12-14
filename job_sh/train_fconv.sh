#!/bin/sh
## current working directory
#$ -cwd
#$ -l f_node=1
#$ -l h_rt=24:00:00
#$ -N train_fconv
#$ -m abe
#$ -M kopamaru@gmail.com
#$ -o o.train_fconv
#$ -e e.train_fconv

## Initialize module command (don't remove)
. /etc/profile.d/modules.sh

module load cuda/9.0.176
module load cudnn/7.3

source ~/fairseq_modified/venv/bin/activate

python ~/fairseq_modified/train.py /gs/hs0/tga-nlp-titech/matsumaru/data/jnc_fairseq_3snt_bin/ \
--arch fconv \
--max-epoch 15 \
--lr 0.25 \
--clip-norm 0.1 \
--save-dir /gs/hs0/tga-nlp-titech/matsumaru/entasum/fairseq_model/jnc_3snt_fconv
