#!/bin/bash
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

set -e
s=${1:-en}
t=${2:-si}
# s=${1:-si}
# t=${2:-en}
# t=${2:-es}

base=cc
center="True"
trueval="True"
lr=10
niter=20

echo "Example based on the ${s}->${t} alignment"

if [ ! -d data/ ]; then
  mkdir -p data;
fi

if [ ! -d res/ ]; then
  mkdir -p res;
fi

# dico_train=data/${s}-${t}.0-5000.txt
# dico_train=data/available-new-${s}-${t}-0-5000.txt

# dico_train=data/${s}-${t}-5000-random.txt
# dico_train=data/available-new-${s}-${t}-0-5000.txt
# dico_train=data/en-si-5000-v2-dict-random.txt
#dico_train=data/maximum_trainset_5000.txt
dico_train=data/maximum_trainset_5000_200k.txt
# dico_train=data/maximum_trainset_5000_80k.txt
# dico_train=data/maximum_trainset_10000_80k.txt
# dico_train=data/maximum_trainset_5000_100k.txt

# if [ ! -f "${dico_train}" ]; then
#   DICO=$(basename -- "${dico_train}")
#   wget -c "https://dl.fbaipublicfiles.com/arrival/dictionaries/${DICO}" -P data/
# fi

# dico_test=data/${s}-${t}.5000-6500.txt
# dico_test=data/en-si-1500-v2-dict-random.txt
#dico_test=data/maximum_testset_1500.txt
dico_test=data/maximum_testset_1500_200k.txt
# dico_test=data/maximum_testset_1500_80k.txt
# dico_test=data/maximum_testset_3000_80k.txt
# dico_test=data/maximum_testset_1500_100k.txt
# dico_test=data/available-new-${s}-${t}-5000-6500.txt

# dico_test=data/${s}-${t}-1500-random.txt
# dico_test=data/available-new-${s}-${t}-5000-6500.txt

# if [ ! -f "${dico_test}" ]; then
#   DICO=$(basename -- "${dico_test}")
#   wget -c "https://dl.fbaipublicfiles.com/arrival/dictionaries/${DICO}" -P data/
# fi

src_emb_gz=data/${base}.${s}.300.vec.gz
src_emb=data/${base}.${s}.300.vec
if [ ! -f "${src_emb}" ]; then
  EMB=$(basename -- "${src_emb_gz}")
  wget -c "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/${EMB}" -P data/
  gzip -dkv ${src_emb_gz}
#   wget -c "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/${EMB}" -P data/
fi

tgt_emb_gz=data/${base}.${t}.300.vec.gz
tgt_emb=data/${base}.${t}.300.vec
if [ ! -f "${tgt_emb}" ]; then
  EMB=$(basename -- "${tgt_emb_gz}")
  wget -c "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/${EMB}" -P data/
  gzip -dkv ${tgt_emb_gz}
#   wget -c "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/${EMB}" -P data/
fi

#output=res/${base}.${s}-${t}.300-july-12-100k-ex1.vec
output=res/${base}.${s}-${t}.300-lr-${lr}-niter-${niter}-center-${center}-200k.vec
# output=res/${base}.${s}-${t}.300-july-04-test1-50k.vec
src_mat=${output}-mat

# python3 align_si.py --src_emb "${src_emb}" --tgt_emb "${tgt_emb}" \
#   --dico_train "${dico_train}" --dico_test "${dico_test}" --output "${output}" \
#   --lr 25 --niter 10 --maxload 100000 --maxneg 100000
# python3 eval_si.py --src_emb "${output}" --tgt_emb "${tgt_emb}" \
#   --dico_test "${dico_test}" --maxload 100000

##############################################################
# train the alignment on the trainset
#if [ ${center}==${trueval} ]; then
#echo "Centering alignment..."
# train the alignment on the trainset
#python3 align_si.py --src_emb "${src_emb}" --tgt_emb "${tgt_emb}" \
#   --dico_train "${dico_train}" --dico_test "${dico_test}" --output "${output}" \
#   --lr ${lr} --niter ${niter} --maxload 200000 --maxneg 200000 --center

# evaluate aligned model on test set
#python3 eval_si.py --src_emb "${output}" --tgt_emb "${tgt_emb}" \
#   --dico_test "${dico_test}" --maxload 200000 --center

# # evaluate aligned model on train set
# python3 eval_si.py --src_emb "${output}" --tgt_emb "${tgt_emb}" \
#   --dico_test "${dico_train}" --maxload 200000 --center ${center}
#else
# train the alignment on the trainset
#python3 align_si.py --src_emb "${src_emb}" --tgt_emb "${tgt_emb}" \
#   --dico_train "${dico_train}" --dico_test "${dico_test}" --output "${output}" \
#   --lr ${lr} --niter ${niter} --maxload 200000 --maxneg 200000

# evaluate aligned model on test set
#python3 eval_si.py --src_emb "${output}" --tgt_emb "${tgt_emb}" \
#   --dico_test "${dico_test}" --maxload 200000

# # evaluate aligned model on train set
# python3 eval_si.py --src_emb "${output}" --tgt_emb "${tgt_emb}" \
#   --dico_test "${dico_train}" --maxload 200000
#fi
###############################################################

# # # evaluate aligned model on test set (using alignment matrix)
# python3 eval_si.py --src_emb "${src_emb}" --tgt_emb "${tgt_emb}" \
#   --dico_test "${dico_test}" --src_mat ${src_mat} --maxload 100000

# # # evaluate aligned model on train set (using alignment matrix)
# python3 eval_si.py --src_emb "${src_emb}" --tgt_emb "${tgt_emb}" \
#   --dico_test "${dico_train}" --src_mat ${src_mat}  --maxload 100000
###############################################################

# evaluate unaligned model on test set
# python3 eval_si.py --src_emb "${src_emb}" --tgt_emb "${tgt_emb}" \
#   --dico_test "${dico_test}" --maxload 50000

# evaluate unaligned model on train set
# python3 eval_si.py --src_emb "${src_emb}" --tgt_emb "${tgt_emb}" \
#   --dico_test "${dico_train}" --maxload 50000


##################################### Eval Plot
if [ ${center}==${trueval} ]; then
echo "Centering alignment..."
python3 plot_eval_results.py --src_emb "${src_emb}" --tgt_emb "${tgt_emb}" \
  --dico_test "${dico_test}" --src_mat ${src_mat}  --maxload 200000 --top_k 10 --fig_prefix "test" --center

python3 plot_eval_results.py --src_emb "${src_emb}" --tgt_emb "${tgt_emb}" \
  --dico_test "${dico_train}" --src_mat ${src_mat}  --maxload 200000 --top_k 10 --fig_prefix "train" --center
else
python3 plot_eval_results.py --src_emb "${src_emb}" --tgt_emb "${tgt_emb}" \
  --dico_test "${dico_test}" --src_mat ${src_mat}  --maxload 200000 --top_k 10 --fig_prefix "test"

python3 plot_eval_results.py --src_emb "${src_emb}" --tgt_emb "${tgt_emb}" \
  --dico_test "${dico_train}" --src_mat ${src_mat}  --maxload 200000 --top_k 10 --fig_prefix "train"
fi
