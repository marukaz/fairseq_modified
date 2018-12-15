#!/bin/sh
## current working directory
#$ -cwd
#$ -l f_node=1
#$ -l h_rt=24:00:00
#$ -N samp_fconv
#$ -m abe
#$ -M kopamaru@gmail.com
#$ -o o.samp_fconv
#$ -e e.samp_fconv

## Initialize module command (don't remove)
. /etc/profile.d/modules.sh

module load cuda/9.0.176
module load cudnn/7.3

source ~/fairseq_modified/venv/bin/activate

beam=128; subset="test"; top=3; prefix=2; path="/gs/hs0/tga-nlp-titech/matsumaru/entasum/fairseq_model/jnc_3snt_fconv_200k_test"; \
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
