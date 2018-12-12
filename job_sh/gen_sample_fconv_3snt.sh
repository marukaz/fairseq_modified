#!/bin/sh
## current working directory
#$ -cwd
#$ -l f_node=1
#$ -l h_rt=24:00:00
#$ -N fconv_test200k_dbs_3snt
#$ -m abe
#$ -M kopamaru@gmail.com
#$ -o o.200k_fconv_dbs_3snt
#$ -e e.200k_fconv_dbs_3snt

## Initialize module command (don't remove)
. /etc/profile.d/modules.sh

module load cuda/9.0.176
module load cudnn/7.3

source ~/fairseq_modified/venv/bin/activate

beam=63; subset="test"; top=5; prefix=0; path="/gs/hs0/tga-nlp-titech/matsumaru/entasum/fairseq_model/jnc_3snt_fconv_200k_test"; \
python ~/fairseq_modified/generate.py /gs/hs0/tga-nlp-titech/matsumaru/data/jnc_fairseq_3snt_200k_test_bin/ \
--path ${path}/checkpoint_best.pt \
--gen-subset ${subset} \
--batch-size 32 \
--beam ${beam} \
--nbest ${beam} \
--prefix-size ${prefix} \
--sampling \
--sampling-topk ${top} \
> ${path}_gen/sample${beam}_top${top}_prefix${prefix}.out
