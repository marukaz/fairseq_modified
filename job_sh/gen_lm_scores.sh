#!/bin/sh
## current working directory
#$ -cwd
#$ -l f_node=1
#$ -l h_rt=24:00:00
#$ -N test3k_dbs_3snt
#$ -m abe
#$ -M kopamaru@gmail.com
#$ -o o.3k_no_prefix_dbs_3snt
#$ -e e.3k_no_prefix_dbs_3snt

## Initialize module command (don't remove)
. /etc/profile.d/modules.sh

module load cuda/9.0.176
module load cudnn/7.3

source ~/fairseq_modified/venv/bin/activate

path="/gs/hs0/tga-nlp-titech/matsumaru/entasum/fairseq_model/jnc_tgt_transformer_lm_200k_test"; \
python ~/fairseq_modified/eval_lm_scores.py /gs/hs0/tga-nlp-titech/matsumaru/data/jnc_fairseq_dbs63_tf_3ktest_bin/ \
--path ${path}/checkpoint_best.pt \
> ${path}_gen/jnc_fairseq_dbs63_tf_3ktest.out
