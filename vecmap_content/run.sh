#binn/bash

s=${1:-en}
#s=${1:-si}

#t=${2:-ta}
#t=${2:-si}
#t=${2:-en}
#t=${2:-fr}
#t=${2:-es}
#t=${2:-tr}
#t=${2:-it}
#t=${2:-de}
t=${2:-ru}

base="wiki"
#base="cc"

retrieval="nn"
#retrieval="invnn"
#retrieval="invsoftmax"
#retrieval="csls"

alignment="sup"  # supervised
#alignment="unsup"  # unsupervised

train="True"
evaluate="True"

if [[ ${s} == "si" || ${t} == "si" ]]; then
  #dico_train=data/wiki_maximum_trainset_5000_200k.txt
  #dico_test=data/wiki_maximum_testset_1500_200k.txt

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
  src_emb_mapped=data/${base}.${s}.mapped_${s}_${t}_${alignment}.300.vec
  if [ ! -f "${src_emb}" ]; then
    EMB=$(basename -- "${src_emb_gz}")
    wget -c "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/${EMB}" -P data/
    gzip -dkv ${src_emb_gz}
  fi

  tgt_emb_gz=data/${base}.${t}.300.vec.gz
  tgt_emb=data/${base}.${t}.300.vec
  tgt_emb_mapped=data/${base}.${t}.mapped_${s}_${t}_${alignment}.300.vec
  if [ ! -f "${tgt_emb}" ]; then
    EMB=$(basename -- "${tgt_emb_gz}")
    wget -c "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/${EMB}" -P data/
    gzip -dkv ${tgt_emb_gz}
  fi  

else

  #src_emb_gz=data/${base}.${s}.vec.gz
  src_emb=data/${base}.${s}.vec
  src_emb_mapped=data/${base}.${s}.mapped_${s}_${t}_${alignment}.vec
  if [ ! -f "${src_emb}" ]; then
    #EMB=$(basename -- "${src_emb_gz}")
    #wget -c "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/${EMB}" -P data/
    curl -Lo ${src_emb} https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.${s}.vec
    #gzip -dkv ${src_emb_gz}
  fi

  #tgt_emb_gz=data/${base}.${t}.vec.gz
  tgt_emb=data/${base}.${t}.vec
  tgt_emb_mapped=data/${base}.${t}.mapped_${s}_${t}_${alignment}.vec
  if [ ! -f "${tgt_emb}" ]; then
    #EMB=$(basename -- "${tgt_emb_gz}")
    #wget -c "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/${EMB}" -P data/
    curl -Lo ${tgt_emb} https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.${t}.vec
    #gzip -dkv ${tgt_emb_gz}
  fi

fi


echo "[INFO] Direction ${s} --> ${t}"
echo "[INFO] src embeddings: ${src_emb}"
echo "[INFO] tgt embeddings: ${tgt_emb}"

if [ ${train} == "True" ]; then
  if [ ${alignment} == "sup" ]; then
    echo "[INFO] Supervised Embedding Alignment..."
    echo "[INFO] train set: ${dico_train}"
    python map_embeddings.py --supervised ${dico_train} ${src_emb} ${tgt_emb} ${src_emb_mapped} ${tgt_emb_mapped} --verbose
  else
    echo "[INFO] Unsupervised Embedding Alignment..."
    python map_embeddings.py --unsupervised ${src_emb} ${tgt_emb} ${src_emb_mapped} ${tgt_emb_mapped} --verbose
   fi
fi

if [ ${evaluate} == "True" ]; then
  echo "[INFO] Evaluating the aligned embeddings using BLI ${s} --> ${t} ${retrieval} retrieval..."
  echo "[INFO] test set: ${dico_test}"
  python eval_translation.py ${src_emb_mapped} ${tgt_emb_mapped} -d ${dico_test} --retrieval ${retrieval}
fi
