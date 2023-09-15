#!/bin/bash
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

set -e
s=${1:-en}
t=${2:-si}
#s=${1:-si}
#t=${2:-en}
# t=${2:-es}

base="cc"
#base="wiki"
center="True"
lr=10
niter=20
maxload=200000
topk=10
patience=10

train="False"
evaluate="False"
plot="True"

model="none"
#model="spectral"

echo "Example based on the ${s}->${t} alignment"

if [ ! -d data/ ]; then
  mkdir -p data;
fi

if [ ! -d res/ ]; then
  mkdir -p res;
fi

#dico_train=data/wiki_maximum_trainset_5000_200k.txt
#dico_test=data/wiki_maximum_testset_1500_200k.txt

dico_train="data/${base}_${s}-${t}_trainset_1500_5000_200k.txt"
dico_test="data/${base}_${s}-${t}_testset_1500_5000_200k.txt"
#dico_test="data/cc_${s}-${t}_testset_1500_5000_200k.txt"

#dico_train=data/maximum_trainset_5000_200k.txt
#dico_test=data/maximum_testset_1500_200k.txt

# dico_train=data/${s}-${t}.0-5000.txt
# dico_train=data/available-new-${s}-${t}-0-5000.txt

# dico_train=data/${s}-${t}-5000-random.txt
# dico_train=data/available-new-${s}-${t}-0-5000.txt
# dico_train=data/en-si-5000-v2-dict-random.txt
#dico_train=data/maximum_trainset_5000.txt
#dico_train=data/maximum_trainset_5000_200k.txt
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
#dico_test=data/maximum_testset_1500_200k.txt
# dico_test=data/maximum_testset_1500_80k.txt
# dico_test=data/maximum_testset_3000_80k.txt
#dico_test=data/maximum_testset_1500_100k.txt
# dico_test=data/available-new-${s}-${t}-5000-6500.txt

# dico_test=data/${s}-${t}-1500-random.txt
# dico_test=data/available-new-${s}-${t}-5000-6500.txt

# if [ ! -f "${dico_test}" ]; then
#   DICO=$(basename -- "${dico_test}")
#   wget -c "https://dl.fbaipublicfiles.com/arrival/dictionaries/${DICO}" -P data/
# fi

if [ $base == "cc" ]; then

  src_emb_gz=data/${base}.${s}.300.vec.gz
  src_emb=data/${base}.${s}.300.vec
  if [ ! -f "${src_emb}" ]; then
    EMB=$(basename -- "${src_emb_gz}")
    wget -c "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/${EMB}" -P data/
    gzip -dkv ${src_emb_gz}
  fi

  tgt_emb_gz=data/${base}.${t}.300.vec.gz
  tgt_emb=data/${base}.${t}.300.vec
  if [ ! -f "${tgt_emb}" ]; then
    EMB=$(basename -- "${tgt_emb_gz}")
    wget -c "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/${EMB}" -P data/
    gzip -dkv ${tgt_emb_gz}
  fi  

else

  src_emb_gz=data/${base}.${s}.vec.gz
  src_emb=data/${base}.${s}.vec
  if [ ! -f "${src_emb}" ]; then
    EMB=$(basename -- "${src_emb_gz}")
    wget -c "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/${EMB}" -P data/
    #curl -Lo data/wiki.${s}.vec https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.${s}.vec
    gzip -dkv ${src_emb_gz}
  fi

  tgt_emb_gz=data/${base}.${t}.vec.gz
  tgt_emb=data/${base}.${t}.vec
  if [ ! -f "${tgt_emb}" ]; then
    EMB=$(basename -- "${tgt_emb_gz}")
    wget -c "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/${EMB}" -P data/
    #curl -Lo data/wiki.${t}.vec https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.${t}.vec
    gzip -dkv ${tgt_emb_gz}
  fi

fi

#output=res/${base}.${s}-${t}.300-july-12-100k-ex1.vec
#output=res/${base}.${s}-${t}.300-lr-${lr}-niter-${niter}-center-${center}-spectral-${model}-top-10-rcsls-200k-freq.vec  # working
#output=res/${base}.${s}-${t}.300-july-04-test1-50k.vec


output=res/cc.en-si.300-lr-10-niter-20-center-True-top-10-rcsls-200k-freq-wo-spectral.vec
src_mat=${output}-mat


#resume_src_mat=res/cc.en-si.300-lr-10-niter-20-center-True-top-10-rcsls-200k-freq-with-spectral.vec-mat
resume_src_mat="no"

# python3 align_si.py --src_emb "${src_emb}" --tgt_emb "${tgt_emb}" \
#   --dico_train "${dico_train}" --dico_test "${dico_test}" --output "${output}" \
#   --lr 25 --niter 10 --maxload 100000 --maxneg 100000
# python3 eval_si.py --src_emb "${output}" --tgt_emb "${tgt_emb}" \
#   --dico_test "${dico_test}" --maxload 100000

##############################################################
if [ ${train} == "True" ]; then
  # train the alignment on the trainset
  if [ ${center}=="True" ]; then
    echo "Centering alignment..."
    # train the alignment on the trainset
    python3 align_si.py --src_emb "${src_emb}" --tgt_emb "${tgt_emb}" \
      --dico_train "${dico_train}" --dico_test "${dico_test}" --output "${output}" --model "${model}" --patience ${patience} \
      --lr ${lr} --niter ${niter} --maxload ${maxload} --maxneg ${maxload} --center --knn ${topk} --src_mat ${resume_src_mat}

    # evaluate aligned model on test set
    #python3 eval_si.py --src_emb "${output}" --tgt_emb "${tgt_emb}" \
    #  --dico_test "${dico_test}" --maxload ${maxload} --center --top_k ${topk}

    # evaluate aligned model on train set
    #python3 eval_si.py --src_emb "${output}" --tgt_emb "${tgt_emb}" \
    #  --dico_test "${dico_train}" --maxload ${maxload} --center --top_k ${topk}
  else
    # train the alignment on the trainset
    python3 align_si.py --src_emb "${src_emb}" --tgt_emb "${tgt_emb}" \
      --dico_train "${dico_train}" --dico_test "${dico_test}" --output "${output}" --model "${model}" --patience ${patience} \
      --lr ${lr} --niter ${niter} --maxload ${maxload} --maxneg ${maxload} --knn ${topk}  --src_mat ${resume_src_mat}

    # evaluate aligned model on test set
    #python3 eval_si.py --src_emb "${output}" --tgt_emb "${tgt_emb}" \
    #  --dico_test "${dico_test}" --maxload ${maxload} --top_k ${topk}

    # evaluate aligned model on train set
    #python3 eval_si.py --src_emb "${output}" --tgt_emb "${tgt_emb}" \
    #  --dico_test "${dico_train}" --maxload ${maxload} --top_k ${topk}
  fi
fi
###############################################################
topks=(1 5 10)
for topk in "${topks[@]}"
do
echo "Evaluating for top_k: $topk ...."

if [ ${evaluate} == "True" ]; then
  if [ ${center}=="True" ]; then
    echo "Centering alignment..."
    # # evaluate aligned model on test set (using alignment matrix)
    python3 eval_si.py --src_emb "${src_emb}" --tgt_emb "${tgt_emb}" \
      --dico_test "${dico_test}" --src_mat ${src_mat} --maxload ${maxload} --center --top_k ${topk}

    # # evaluate aligned model on train set (using alignment matrix)
    #python3 eval_si.py --src_emb "${src_emb}" --tgt_emb "${tgt_emb}" \
    #  --dico_test "${dico_train}" --src_mat ${src_mat}  --maxload ${maxload} --center --top_k ${topk}
  else
    # # evaluate aligned model on test set (using alignment matrix)
    python3 eval_si.py --src_emb "${src_emb}" --tgt_emb "${tgt_emb}" \
      --dico_test "${dico_test}" --src_mat ${src_mat} --maxload ${maxload} --top_k ${topk}

    # # evaluate aligned model on train set (using alignment matrix)
    #python3 eval_si.py --src_emb "${src_emb}" --tgt_emb "${tgt_emb}" \
    #  --dico_test "${dico_train}" --src_mat ${src_mat}  --maxload ${maxload} --top_k ${topk}
  fi
fi

done # end of for-loop
###############################################################

# evaluate unaligned model on test set
# python3 eval_si.py --src_emb "${src_emb}" --tgt_emb "${tgt_emb}" \
#   --dico_test "${dico_test}" --maxload ${maxload} --top_k ${topk}

# evaluate unaligned model on train set
# python3 eval_si.py --src_emb "${src_emb}" --tgt_emb "${tgt_emb}" \
#   --dico_test "${dico_train}" --maxload ${maxload} --top_k ${topk}


##################################### Eval Plot
if [ ${plot} == "True" ]; then
  if [ ${center}=="True" ]; then
    echo "Centering alignment..."
    python3 plot_eval_results.py --src_emb "${src_emb}" --tgt_emb "${tgt_emb}" \
      --dico_test "${dico_test}" --src_mat ${src_mat}  --maxload ${maxload} --top_k ${topk} --fig_prefix "test" --center

    #python3 plot_eval_results.py --src_emb "${src_emb}" --tgt_emb "${tgt_emb}" \
    #  --dico_test "${dico_train}" --src_mat ${src_mat}  --maxload ${maxload} --top_k ${topk} --fig_prefix "train" --center
  else
    python3 plot_eval_results.py --src_emb "${src_emb}" --tgt_emb "${tgt_emb}" \
      --dico_test "${dico_test}" --src_mat ${src_mat}  --maxload ${maxload} --top_k ${topk} --fig_prefix "test"

    #python3 plot_eval_results.py --src_emb "${src_emb}" --tgt_emb "${tgt_emb}" \
    #  --dico_test "${dico_train}" --src_mat ${src_mat}  --maxload ${maxload} --top_k ${topk} --fig_prefix "train"
  fi
fi

