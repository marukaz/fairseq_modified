#!/bin/sh
## current working directory
#$ -cwd
#$ -l q_node=1
#$ -l h_rt=24:00:00
#$ -N beam
#$ -m abe
#$ -M kopamaru@gmail.com
#$ -o o.beam
#$ -e e.beam

## Initialize module command (don't remove)
. /etc/profile.d/modules.sh

module load cuda/9.0.176
module load cudnn/7.3

source ~/fairseq_modified/venv/bin/activate

beam=31; subset="train"; \
python ~/fairseq_modified/generate.py /gs/hs0/tga-nlp-titech/takase/headline/data/jnc_preprocessed4fairseq_bin/jnc_1snt_spm_headline/ \
--path /gs/hs0/tga-nlp-titech/takase/headline/exp/jnc_spm_fairseq_transformer_wmtendesetting/jnc_spm_1snt_dropout0.1_gpu4_updatefreq2/checkpoint_best.pt \
--gen-subset train \
--batch-size 64 \
--beam ${beam} \
--nbest ${beam} | tee ~/fairseq_modified/gen/beam${beam}_${subset}_wmt_d01_gpu4_updatefreq2.json
