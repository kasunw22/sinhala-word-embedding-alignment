#binn/bash

s=${1:-en}
t=${2:-si}

base="wiki"

#retrieval="nn"
#retrieval="invnn"
#retrieval="invsoftmax"
retrieval="csls"

train="False"
evaluate="True"

if [[ ${s} == "si" || ${t} == "si" ]]; then

  dico_train="data/${base}_${s}-${t}_trainset_1500_5000_200k.txt"
  dico_test="data/${base}_${s}-${t}_testset_1500_5000_200k.txt"
else

  dico_train=data/${s}-${t}.0-5000.txt
  dico_test=data/${s}-${t}.5000-6500.txt
fi

if [ ! -f "${dico_train}" ]; then
  DICO=$(basename -- "${dico_train}")
  wget -c "https://dl.fbaipublicfiles.com/arrival/dictionaries/${DICO}" -P data/
fi

if [ ! -f "${dico_test}" ]; then
  DICO=$(basename -- "${dico_test}")
  wget -c "https://dl.fbaipublicfiles.com/arrival/dictionaries/${DICO}" -P data/
fi



if [ $base == "cc" ]; then

  src_emb_gz=data/${base}.${s}.300.vec.gz
  src_emb=data/${base}.${s}.300.vec
  src_emb_mapped=data/${base}.${s}.mapped_${s}_${t}.300.vec
  if [ ! -f "${src_emb}" ]; then
    EMB=$(basename -- "${src_emb_gz}")
    wget -c "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/${EMB}" -P data/
    gzip -dkv ${src_emb_gz}
  fi

  tgt_emb_gz=data/${base}.${t}.300.vec.gz
  tgt_emb=data/${base}.${t}.300.vec
  tgt_emb_mapped=data/${base}.${t}.mapped_${s}_${t}.300.vec
  if [ ! -f "${tgt_emb}" ]; then
    EMB=$(basename -- "${tgt_emb_gz}")
    wget -c "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/${EMB}" -P data/
    gzip -dkv ${tgt_emb_gz}
  fi

else

  src_emb=data/${base}.${s}.vec
  src_emb_mapped=data/${base}.${s}.mapped_${s}_${t}.vec
  if [ ! -f "${src_emb}" ]; then
    curl -Lo ${src_emb} https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.${s}.vec
  fi

  tgt_emb=data/${base}.${t}.vec
  tgt_emb_mapped=data/${base}.${t}.mapped_${s}_${t}.vec
  if [ ! -f "${tgt_emb}" ]; then
    curl -Lo ${tgt_emb} https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.${t}.vec
  fi

fi


if [ ${train} == "True" ]; then
  echo "[INFO] Aligning the embeddings..."
  python map_embeddings.py --supervised ${dico_train} ${src_emb} ${tgt_emb} ${src_emb_mapped} ${tgt_emb_mapped} --verbose
fi

if [ ${evaluate} == "True" ]; then
  echo "[INFO] Evaluating the aligned embeddings using BLI..."
  python eval_translation.py ${src_emb_mapped} ${tgt_emb_mapped} -d ${dico_test} --retrieval ${retrieval}
fi