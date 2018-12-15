#!/bin/sh
## current working directory
#$ -cwd
#$ -l f_node=1
#$ -l h_rt=24:00:00
#$ -N beam_3snt
#$ -m abe
#$ -M kopamaru@gmail.com
#$ -o o.beam_3snt
#$ -e e.beam_3snt

## Initialize module command (don't remove)
. /etc/profile.d/modules.sh

module load cuda/9.0.176
module load cudnn/7.3

source ~/fairseq_modified/venv/bin/activate

beam=63; \
python ~/fairseq_modified/generate.py /gs/hs0/tga-nlp-titech/matsumaru/data/jnc_fairseq_3snt_bin/ \
--path /gs/hs0/tga-nlp-titech/matsumaru/entasum/fairseq_model/jnc_3snt_transformer_wmtset_d01_upfreq2/checkpoint_best.pt \
--batch-size 32 \
--beam ${beam} \
--nbest ${beam} > /gs/hs0/tga-nlp-titech/matsumaru/entasum/fairseq_model/jnc_3snt_transformer_wmtset_d01_upfreq2_gen/beam${beam}_snt3_wmt_d01_gpu4_updatefreq2.out
