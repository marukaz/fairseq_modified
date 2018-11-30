#!/bin/sh
## current working directory
#$ -cwd
#$ -l f_node=1
#$ -l h_rt=01:00:00
#$ -N sample_top2
#$ -m abe
#$ -M kopamaru@gmail.com
#$ -o o.sample
#$ -e e.sample

## Initialize module command (don't remove)
. /etc/profile.d/modules.sh

module load cuda/9.0.176
module load cudnn/7.3

source ~/fairseq/venv/bin/activate

beam=30; top=2; \
python ~/fairseq/generate.py /gs/hs0/tga-nlp-titech/takase/headline/data/jnc_preprocessed4fairseq_bin/jnc_1snt_spm_headline/ \
--path /gs/hs0/tga-nlp-titech/takase/headline/exp/jnc_spm_fairseq_transformer_wmtendesetting/jnc_spm_1snt_dropout0.1_gpu4_updatefreq2/checkpoint_best.pt \
--batch-size 64 \
--beam ${beam} \
--nbest ${beam} \
--sampling-topk ${top} \
--sampling | tee ~/fairseq/gen/sample_${beam}_top${top}.out
