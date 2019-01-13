#!/bin/sh
## current working directory
#$ -cwd
#$ -l f_node=1
#$ -l h_rt=24:00:00
#$ -N dbs_3snt
#$ -m abe
#$ -M kopamaru@gmail.com
#$ -o o.no_prefix_dbs_3snt
#$ -e e.no_prefix_dbs_3snt

## Initialize module command (don't remove)
. /etc/profile.d/modules.sh

module load cuda/9.0.176
module load cudnn/7.3

source ~/fairseq_modified/venv/bin/activate

range="1-25000" \
beam=31; subset="test"; prefix=0; \
path="/gs/hs0/tga-nlp-titech/matsumaru/entasum/fairseq_model/jnc_3snt_transformer_100k_test_wmtset_d01_upfreq2"; \
python ~/fairseq_modified/generate.py /gs/hs0/tga-nlp-titech/matsumaru/data/100k_test_${range}_bin/ \
--path ${path}/checkpoint_best.pt \
--add-gold \
--gen-subset ${subset} \
--batch-size 63 \
--beam ${beam} \
--nbest ${beam} \
--prefix-size ${prefix} \
--diverse-beam-groups ${beam} \
> ${path}_gen/dbs${beam}_from_${subset}_range_${range}.out
