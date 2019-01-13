#!/bin/sh
## current working directory
#$ -cwd
#$ -l f_node=1
#$ -l h_rt=24:00:00
#$ -N lm_scores
#$ -m abe
#$ -M kopamaru@gmail.com
#$ -o o.lm_scores
#$ -e e.lm_scores

## Initialize module command (don't remove)
. /etc/profile.d/modules.sh

module load cuda/9.0.176
module load cudnn/7.3

source ~/fairseq_modified/venv/bin/activate

data="dbs31_100k_test_1-25000"
path="/gs/hs0/tga-nlp-titech/matsumaru/entasum/fairseq_model/jnc_tgt_transformer_lm_100k_test"; \
python ~/fairseq_modified/eval_lm_scores.py /gs/hs0/tga-nlp-titech/matsumaru/data/${data}_bin \
--path ${path}/checkpoint_best.pt \
--sample-break-mode eos \
> ${path}_gen/${data}.out
