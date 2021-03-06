#!/bin/sh
## current working directory
#$ -cwd
#$ -l f_node=1
#$ -l h_rt=24:00:00
#$ -N beam5_3snt
#$ -m abe
#$ -M kopamaru@gmail.com
#$ -o o.beam5_3snt
#$ -e e.beam5_3snt

## Initialize module command (don't remove)
. /etc/profile.d/modules.sh

module load cuda/9.0.176
module load cudnn/7.3

source ~/fairseq_modified/venv/bin/activate

beam=5; subset="test"; \
path="/gs/hs0/tga-nlp-titech/matsumaru/entasum/fairseq_model/jnc_3snt_transformer_100k_test_wmtset_d01_upfreq2"; \
python ~/fairseq_modified/generate.py /gs/hs0/tga-nlp-titech/matsumaru/data/jnc_fairseq_3snt_reference_entail_test_bin \
--path ${path}/checkpoint_best.pt \
--gen-subset ${subset} \
--batch-size 128 \
--beam ${beam} \
--nbest ${beam} > ${path}_gen/beam${beam}_from_${subset}_reference_entail_test.out
